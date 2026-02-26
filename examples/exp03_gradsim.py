"""
Recover the mass and shape of an unknown object.
"""

import argparse
import json
import math
import os

import imageio
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm, trange

from gradsim.bodies import RigidBody
from gradsim.forces import ConstantForce
from gradsim.renderutils import SoftRenderer, TriangleMesh
from gradsim.renderutils.dibr.renderer import Renderer as DIBRenderer
from gradsim.renderutils.dibr.utils.sphericalcoord import get_spherical_coords_x
from gradsim.simulator import Simulator
from gradsim.utils import meshutils
from gradsim.utils.logging import write_imglist_to_gif


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


def write_meshes_to_dir(vertices_list, faces, dirname):
    os.makedirs(dirname, exist_ok=True)
    for i, verts in enumerate(vertices_list):
        np.savetxt(os.path.join(dirname, f"vertices_{i:02d}.txt"), verts)
    np.savetxt(os.path.join(dirname, "faces.txt"), faces)


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
        default=os.path.join("cache", "exp03-rigidshape"),
        help="Directory to store logs in.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (for repeatability)"
    )
    parser.add_argument(
        "--infile",
        type=str,
        default=os.path.join("cache", "dataset-rigid-shape", "obj", "apple.obj"),
        help="Path to input mesh (.obj) file.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=os.path.join("sampledata", "sphere.obj"),
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
        default=1,
        help="Apply loss every `--compare-every` frames.",
    )
    parser.add_argument(
        "--non-uniform-density",
        action="store_true",
        help="Whether to treat the object as having non-uniform density.",
    )
    parser.add_argument(
        "--force-magnitude",
        type=float,
        default=1000.0,
        help="Magnitude of external force.",
    )
    parser.add_argument("--log", action="store_true", help="Save log files.")
    parser.add_argument(
        "--log-every", type=int, default=10, help="How frequently to log gifs."
    )

    args = parser.parse_args()

    if args.compare_every >= args.simsteps:
        raise ValueError(
            f"Arg --compare-every cannot be greater than or equal to {args.simsteps}."
        )

    logdir = os.path.join(args.logdir, args.expid)
    if args.log:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    print("Setting up the expt...")

    mesh_gt = TriangleMesh.from_obj(args.infile)
    vertices_gt = meshutils.normalize_vertices(mesh_gt.vertices[None, :])
    faces_gt = mesh_gt.faces[None, :]
    textures_gt = jnp.stack(
        (
            jnp.ones((1, vertices_gt.shape[-2]), dtype=jnp.float32),
            jnp.ones((1, vertices_gt.shape[-2]), dtype=jnp.float32),
            jnp.zeros((1, vertices_gt.shape[-2]), dtype=jnp.float32),
        ),
        axis=-1,
    )

    position_gt = jnp.array([0.0, 2.0, 0.0], dtype=jnp.float32)
    orientation_gt = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    mass_per_vertex = 1.0
    masses_gt = mass_per_vertex * jnp.ones(vertices_gt.shape[1], dtype=jnp.float32)
    body_gt = RigidBody(
        vertices_gt[0],
        masses=masses_gt,
        position=position_gt,
        orientation=orientation_gt,
    )

    gravity_gt = ConstantForce(
        direction=jnp.array([0.0, -1.0, 0.0]),
        magnitude=args.force_magnitude,
    )

    inds = jnp.argmin(vertices_gt, axis=-2)[0]
    application_points = list(np.array(inds).flatten())
    body_gt.add_external_force(gravity_gt, application_points=application_points)

    sim_gt = Simulator([body_gt])

    renderer = DIBRenderer(128, 128, mode="VertexColor")
    camera_distance = 8.0
    elevation = 30.0
    azimuth = 30.0
    renderer.set_look_at_parameters([90 - azimuth], [elevation], [camera_distance])

    imgs_gt = []

    print("Running GT simulation...")
    for i in trange(args.simsteps):
        sim_gt.step()
        img_gt, alpha_gt, _ = renderer.forward(
            points=[
                body_gt.get_world_vertices()[None, :], faces_gt[0]
            ],
            colors_bxpx3=textures_gt,
        )
        rgba = jnp.concatenate((img_gt, alpha_gt), axis=-1)
        imgs_gt.append(rgba)

    if args.log:
        write_imglist_to_gif(imgs_gt, os.path.join(logdir, "gt.gif"), imgformat="dibr")

    # Load the template mesh (usually a sphere).
    mesh = TriangleMesh.from_obj(args.template)
    vertices = meshutils.normalize_vertices(mesh.vertices[None, :])
    faces = mesh.faces[None, :]
    textures = jnp.stack(
        (
            jnp.ones((1, vertices.shape[-2]), dtype=jnp.float32),
            jnp.ones((1, vertices.shape[-2]), dtype=jnp.float32),
            jnp.zeros((1, vertices.shape[-2]), dtype=jnp.float32),
        ),
        axis=-1,
    )

    mass_per_vertex_est = 0.8
    masses_est = mass_per_vertex_est * jnp.ones(vertices.shape[1], dtype=jnp.float32)

    gravity = ConstantForce(
        direction=jnp.array([0.0, -1.0, 0.0]),
        magnitude=args.force_magnitude,
    )

    args.shapeepochs = 130

    inds_est = jnp.argmin(vertices, axis=-2)[0]
    application_points_est = list(np.array(inds_est).flatten())

    # Shape optimization only (mass fixed)
    params_shape = {"update": jnp.zeros(vertices.shape, dtype=jnp.float32)}
    opt_state_shape = _adam_init(params_shape)
    masslosses = []
    initial_imgs = []

    def shape_loss_fn(params_s):
        vertices_cur = vertices + params_s["update"]
        masses_cur = jnp.ones_like(masses_est)
        body = RigidBody(
            vertices_cur[0],
            masses=masses_cur,
            position=position_gt,
            orientation=orientation_gt,
        )
        body.add_external_force(gravity, application_points=application_points_est)
        sim_est = Simulator([body])
        imgs_est_inner = []
        vertices_est_inner = []
        for t in range(args.simsteps):
            sim_est.step()
            img, alpha, _ = renderer.forward(
                points=[body.get_world_vertices()[None, :], faces[0]],
                colors_bxpx3=textures,
            )
            rgba = jnp.concatenate((img, alpha), axis=-1)
            imgs_est_inner.append(rgba)
            vertices_est_inner.append(np.array(body.get_world_vertices()))
        loss = sum(
            jnp.mean((est - gt) ** 2)
            for est, gt in zip(imgs_est_inner[::5], imgs_gt[::5])
        ) / len(imgs_est_inner[::5])
        return loss, imgs_est_inner, vertices_est_inner

    shape_grad_fn = jax.value_and_grad(lambda p: shape_loss_fn(p)[0])

    for i in trange(args.shapeepochs):
        loss_val, grads = shape_grad_fn(params_shape)
        params_shape, opt_state_shape = _adam_step(params_shape, grads, opt_state_shape, lr=5e-2)

        masses_cur = jnp.ones_like(masses_est)
        tqdm.write(
            f"Total Loss: {float(loss_val):.5f}, "
            f"Mass error: {float(jnp.abs(masses_gt.mean() - masses_cur.mean())):.5f}, "
            f"Mass (est): {float(masses_cur.mean()):.5f}"
        )

        masslosses.append(float(loss_val))
        est_masses = np.array(masses_est)

        if args.log and i % args.log_every == 0:
            _, imgs_est_log, vertices_est_log = shape_loss_fn(params_shape)
            write_imglist_to_gif(
                imgs_est_log, os.path.join(logdir, f"shape_{i:05d}.gif"), imgformat="dibr"
            )
            write_meshes_to_dir(
                vertices_est_log,
                np.array(faces[0]),
                os.path.join(logdir, f"vertices_{i:05d}"),
            )
