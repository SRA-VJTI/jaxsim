"""
Recover the mass of an object with known shape.
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
        default="cache/mass_known_shape",
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
        "--simsteps",
        type=int,
        default=20,
        help="Number of steps to run simulation for.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to run optimization for.",
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

    mesh = TriangleMesh.from_obj(args.infile)
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
    masses_gt = jnp.arange(vertices.shape[1], dtype=jnp.float32)
    body_gt = RigidBody(vertices[0], masses=masses_gt)

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
            body_gt.get_world_vertices()[None, :], faces, textures
        )
        img_gt.append(rgba)

    if args.uniform_density:
        print("Using uniform density assumption...")
        masses_est = 0.15 * jnp.ones(1, dtype=jnp.float32)
    else:
        print("Assuming nonuniform density...")
        masses_est = 0.15 * jnp.ones(vertices.shape[1], dtype=jnp.float32)

    params = {"update": jnp.zeros_like(masses_est)}
    opt_state = _adam_init(params)

    losses = []
    est_masses = None
    initial_imgs = []

    def loss_fn(params):
        update = params["update"]
        if args.uniform_density:
            masses_cur = jnp.maximum(masses_est + update, 0.0).repeat(vertices.shape[1])
        else:
            masses_cur = jnp.maximum(masses_est + update, 0.0)
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
        return (
            sum(
                jnp.mean((est - gt) ** 2)
                for est, gt in zip(
                    img_est_inner[:: args.compare_every], img_gt[:: args.compare_every]
                )
            )
            / len(img_est_inner[:: args.compare_every]),
            img_est_inner,
        )

    grad_fn = jax.value_and_grad(lambda p: loss_fn(p)[0])

    img_est = None
    for i in trange(args.epochs):
        loss_val, grads = grad_fn(params)
        lr = 5e-1
        if i >= 80:
            lr = 5e-1 * 0.25
        elif i >= 40:
            lr = 5e-1 * 0.5
        params, opt_state = _adam_step(params, grads, opt_state, lr=lr)

        if args.uniform_density:
            masses_cur = jnp.maximum(masses_est + params["update"], 0.0).repeat(vertices.shape[1])
        else:
            masses_cur = jnp.maximum(masses_est + params["update"], 0.0)

        tqdm.write(
            f"Loss: {float(loss_val):.5f}, "
            f"Mass (err): {float(jnp.abs(masses_cur - masses_gt).mean()):.5f}"
        )
        losses.append(float(loss_val))
        est_masses = np.array(masses_cur)

        # Compute img_est for logging (at first and last epoch)
        if i == 0 or args.log:
            _, img_est = loss_fn(params)
            if i == 0:
                initial_imgs = list(img_est)

    # Save viz, if specified.
    if args.log:
        logdir = Path(args.logdir) / args.expid
        logdir.mkdir(exist_ok=True)

        if img_est is None:
            _, img_est = loss_fn(params)

        initwriter = imageio.get_writer(logdir / "init.gif", mode="I")
        gtwriter = imageio.get_writer(logdir / "gt.gif", mode="I")
        estwriter = imageio.get_writer(logdir / "est.gif", mode="I")
        for gtimg, estimg, initimg in zip(img_gt, img_est, initial_imgs):
            gtimg = np.array(gtimg[0]).transpose(1, 2, 0)
            estimg = np.array(estimg[0]).transpose(1, 2, 0)
            initimg = np.array(initimg[0]).transpose(1, 2, 0)
            gtwriter.append_data((255 * gtimg).astype(np.uint8))
            estwriter.append_data((255 * estimg).astype(np.uint8))
            initwriter.append_data((255 * initimg).astype(np.uint8))
        gtwriter.close()
        estwriter.close()
        initwriter.close()

        np.savetxt(logdir / "losses.txt", losses)
        np.savetxt(logdir / "masses.txt", est_masses)
