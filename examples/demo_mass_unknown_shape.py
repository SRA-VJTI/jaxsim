"""
Recover the mass and shape of an unknown object.
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm, trange

from gradsim.bodies import RigidBody
from gradsim.forces import ConstantForce
from gradsim.renderutils import SoftRenderer, TriangleMesh
from gradsim.simulator import Simulator
from gradsim.utils import meshutils


def _adam_init(params):
    return {k: {"m": jnp.zeros_like(v), "v": jnp.zeros_like(v), "t": 0}
            for k, v in params.items()}


def _adam_step(params, grads, state, lr, beta1=0.5, beta2=0.99, eps=1e-8):
    new_params, new_state = {}, {}
    for k in params:
        t = state[k]["t"] + 1
        m = beta1 * state[k]["m"] + (1 - beta1) * grads[k]
        v = beta2 * state[k]["v"] + (1 - beta2) * grads[k] ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        new_params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        new_state[k] = {"m": m, "v": v, "t": t}
    return new_params, new_state


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expid",
        type=str,
        default="default",
        help="Unique string identifier for experiments.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="cache/mass_unknown_shape",
        help="Directory to store logs in.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (for repeatability)"
    )
    parser.add_argument(
        "--infile",
        type=str,
        default=Path("sampledata/cube.obj"),
        help="Path to input mesh (.obj) file.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=Path("sampledata/sphere.obj"),
        help="Path to input mesh (.obj) file.",
    )
    parser.add_argument(
        "--simsteps",
        type=int,
        default=20,
        help="Number of steps to run simulation for.",
    )
    parser.add_argument(
        "--shapeepochs",
        type=int,
        default=100,
        help="Number of epochs to run shape optimization for.",
    )
    parser.add_argument(
        "--massepochs",
        type=int,
        default=100,
        help="Number of epochs to run mass optimization for.",
    )
    parser.add_argument(
        "--compare-every",
        type=int,
        default=10,
        help="Apply loss every `--compare-every` frames.",
    )
    parser.add_argument(
        "--uniform-density",
        action="store_true",
        help="Whether to treat the object as having uniform density.",
    )
    parser.add_argument(
        "--force-magnitude",
        type=float,
        default=10.0,
        help="Magnitude of external force.",
    )
    parser.add_argument("--log", action="store_true", help="Save log files.")

    args = parser.parse_args()

    if args.compare_every >= args.simsteps:
        raise ValueError(
            f"Arg --compare-every cannot be greater than or equal to {args.simsteps}."
        )

    # Load GT body (from a triangle mesh obj file).
    mesh_gt = TriangleMesh.from_obj(args.infile)
    vertices_gt = meshutils.normalize_vertices(mesh_gt.vertices[None, :])
    faces_gt = mesh_gt.faces[None, :]
    textures_gt = jnp.concatenate(
        (
            jnp.ones((1, faces_gt.shape[1], 2, 1), dtype=jnp.float32),
            jnp.ones((1, faces_gt.shape[1], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces_gt.shape[1], 2, 1), dtype=jnp.float32),
        ),
        axis=-1,
    )
    mass_per_vertex = 1.0 / vertices_gt.shape[1]
    masses_gt = mass_per_vertex * jnp.ones(vertices_gt.shape[1], dtype=jnp.float32)
    body_gt = RigidBody(vertices_gt[0], masses=masses_gt)

    gravity = ConstantForce(
        direction=jnp.array([0.0, -1.0, 0.0]),
        magnitude=args.force_magnitude,
    )

    body_gt.add_external_force(gravity, application_points=[0, 1])

    sim_gt = Simulator([body_gt])

    renderer = SoftRenderer(camera_mode="look_at")
    camera_distance = 8.0
    elevation = 30.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    img_gt = []

    for i in trange(args.simsteps):
        sim_gt.step()
        rgba = renderer.forward(
            body_gt.get_world_vertices()[None, :], faces_gt, textures_gt
        )
        img_gt.append(rgba)

    # Load the template mesh (usually a sphere).
    mesh = TriangleMesh.from_obj(args.template)
    vertices = meshutils.normalize_vertices(mesh.vertices[None, :])
    faces = mesh.faces[None, :]
    textures = jnp.concatenate(
        (
            jnp.ones((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            jnp.ones((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces.shape[1], 2, 1), dtype=jnp.float32),
        ),
        axis=-1,
    )

    mass_per_vertex_est = 0.5 / vertices.shape[1]
    masses_est = mass_per_vertex_est * jnp.ones(vertices.shape[1], dtype=jnp.float32)

    # Phase 1: mass optimization
    if args.uniform_density:
        params_mass = {"update": jnp.zeros(1, dtype=jnp.float32)}
    else:
        params_mass = {"update": jnp.zeros(masses_est.shape, dtype=jnp.float32)}
    opt_state_mass = _adam_init(params_mass)

    masslosses = []
    est_masses = None
    initial_imgs = []

    def mass_loss_fn(params_m):
        if args.uniform_density:
            masses_cur = jnp.maximum(masses_est + params_m["update"], 0.0).repeat(vertices.shape[1])
        else:
            masses_cur = jnp.maximum(masses_est + params_m["update"], 0.0)
        body = RigidBody(vertices[0], masses=masses_cur)
        body.add_external_force(gravity, application_points=[0, 1])
        sim_est = Simulator([body])
        img_est_inner = []
        for t in range(args.simsteps):
            sim_est.step()
            rgba = renderer.forward(
                body.get_world_vertices()[None, :], faces, textures
            )
            img_est_inner.append(rgba)
        return sum(
            jnp.mean((est - gt) ** 2)
            for est, gt in zip(
                img_est_inner[:: args.compare_every], img_gt[:: args.compare_every]
            )
        ) / len(img_est_inner[:: args.compare_every])

    mass_grad_fn = jax.value_and_grad(mass_loss_fn)

    for i in trange(args.massepochs):
        loss_val, grads = mass_grad_fn(params_mass)
        lr = 5e-1
        if i >= 80:
            lr = 5e-1 * 0.25
        elif i >= 40:
            lr = 5e-1 * 0.5
        params_mass, opt_state_mass = _adam_step(params_mass, grads, opt_state_mass, lr=lr)

        if args.uniform_density:
            masses_cur = jnp.maximum(masses_est + params_mass["update"], 0.0).repeat(vertices.shape[1])
        else:
            masses_cur = jnp.maximum(masses_est + params_mass["update"], 0.0)

        tqdm.write(f"Mass Loss: {float(loss_val):.5f}, Mass (est): {float(masses_cur.mean()):.5f}")
        masslosses.append(float(loss_val))
        est_masses = np.array(masses_cur)

    # Phase 2: shape optimization
    params_shape = {"update": jnp.zeros(vertices.shape, dtype=jnp.float32)}
    opt_state_shape = _adam_init(params_shape)
    shapelosses = []

    if args.uniform_density:
        masses_final = jnp.maximum(masses_est + params_mass["update"], 0.0).repeat(vertices.shape[1])
    else:
        masses_final = jnp.maximum(masses_est + params_mass["update"], 0.0)

    def shape_loss_fn(params_s):
        vertices_cur = vertices + params_s["update"]
        body = RigidBody(vertices_cur[0], masses=masses_final)
        body.add_external_force(gravity, application_points=[0, 1])
        sim_est = Simulator([body])
        img_est_inner = []
        for t in range(args.simsteps):
            sim_est.step()
            rgba = renderer.forward(
                body.get_world_vertices()[None, :], faces, textures
            )
            img_est_inner.append(rgba)
        return sum(
            jnp.mean((est - gt) ** 2)
            for est, gt in zip(
                img_est_inner[:: args.compare_every], img_gt[:: args.compare_every]
            )
        ) / len(img_est_inner[:: args.compare_every])

    shape_grad_fn = jax.value_and_grad(shape_loss_fn)

    for i in trange(args.shapeepochs):
        loss_val, grads = shape_grad_fn(params_shape)
        params_shape, opt_state_shape = _adam_step(params_shape, grads, opt_state_shape, lr=1e-2)
        tqdm.write(
            f"Shape Loss: {float(loss_val):.5f}, Mass (est): {float(masses_final.mean()):.5f}"
        )
        shapelosses.append(float(loss_val))

    # Save viz, if specified.
    if args.log:
        logdir = Path(args.logdir) / args.expid
        logdir.mkdir(exist_ok=True)

        np.savetxt(logdir / "masslosses.txt", masslosses)
        np.savetxt(logdir / "shapelosses.txt", shapelosses)
        np.savetxt(logdir / "masses.txt", est_masses)
        shape = np.array((vertices + params_shape["update"])[0])
        np.savetxt(logdir / "shape.txt", shape)
