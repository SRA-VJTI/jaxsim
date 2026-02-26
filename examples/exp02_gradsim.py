"""
Estimate mass of a known rigid shape (primitives: cube, cylinder, sphere, etc.)

Use the first-cut dataset that Vikram generated.
"""

import argparse
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm, trange

from gradsim.assets.primitives import INT_TO_PRIMITIVE, get_primitive_obj
from gradsim.bodies import RigidBody
from gradsim.engines import SemiImplicitEulerWithContacts
from gradsim.forces import ConstantForce
from gradsim.renderutils import SoftRenderer, TriangleMesh
from gradsim.simulator import Simulator
from gradsim.utils import meshutils
from gradsim.utils.h5 import HDF5Dataset


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
        help="Unique string identifier for the experiment.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=os.path.join("cache", "exp01"),
        help="Directory to store logs, for multiple runs of this experiment.",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default=os.path.join("cache", "dataset-rigid-exp2"),
        help="Directory containing the HDF5 dataset for the experiment.",
    )
    parser.add_argument(
        "--optiters",
        type=int,
        default=30,
        help="Number of iterations to optimize each object mass.",
    )
    parser.add_argument(
        "--compare-every",
        type=int,
        default=5,
        help="Number of frames after which to compute pixel-wise MSE loss.",
    )
    parser.add_argument("--log", action="store_true", help="Save log files.")

    args = parser.parse_args()

    sim_duration = 1.0
    fps = 30
    sim_substeps = 32
    sim_dtime = (1 / 30) / sim_substeps
    sim_steps = int(sim_duration / sim_dtime)
    render_every = sim_substeps

    logdir = os.path.join(args.logdir, args.expid)
    if args.log:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    true_masses = []
    predicted_masses = []
    true_restitutions = []
    predicted_restitutions = []

    dataset = HDF5Dataset(args.datadir, read_only_seqs=False)

    for idx, out in enumerate(dataset):

        starttime = time.time()

        print("Object:", idx)

        if idx > 0:
            continue

        (
            seqs,
            shape,
            init_pos,
            orientation,
            mass,
            fric,
            elas,
            color,
            scale,
            force_application_points,
            force_magnitude,
            force_direction,
            linear_velocity,
            angular_velocity,
        ) = out

        image_size = 256
        camera_mode = "look_at"
        camera_distance = 8.0
        elevation = 30.0
        azimuth = 0.0
        renderer = SoftRenderer(image_size=image_size, camera_mode=camera_mode)
        renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

        obj = get_primitive_obj(INT_TO_PRIMITIVE[shape[0]])
        mesh = TriangleMesh.from_obj(obj)
        vertices = meshutils.normalize_vertices(mesh.vertices)[None, :]
        faces = mesh.faces[None, :]
        textures = jnp.concatenate(
            (
                color[0][0] / 255.0 * jnp.ones((1, faces.shape[1], 2, 1), dtype=jnp.float32),
                color[0][1] / 255.0 * jnp.ones((1, faces.shape[1], 2, 1), dtype=jnp.float32),
                color[0][2] / 255.0 * jnp.ones((1, faces.shape[1], 2, 1), dtype=jnp.float32),
            ),
            axis=-1,
        )

        true_masses.append(mass[0])
        true_restitutions.append(float(elas[0]))

        masses_gt = float(mass[0]) * jnp.ones(vertices.shape[-2], dtype=jnp.float32)
        restitution_gt = float(elas[0])
        body_gt = RigidBody(
            vertices[0],
            masses=masses_gt,
            position=jnp.array(init_pos[0], dtype=jnp.float32),
            orientation=jnp.array(orientation[0], dtype=jnp.float32),
            friction_coefficient=float(fric[0]),
            restitution=restitution_gt,
        )

        gravity = ConstantForce(
            magnitude=10.0, direction=jnp.array([0, -1, 0], dtype=jnp.float32),
        )
        body_gt.add_external_force(gravity)

        sim_gt = Simulator(
            [body_gt], dtime=sim_dtime, engine=SemiImplicitEulerWithContacts()
        )

        imgs_gt = []
        for t in range(sim_steps):
            sim_gt.step()
            if t % render_every == 0:
                rgba = renderer.forward(
                    body_gt.get_world_vertices()[None, :], faces, textures
                )
                imgs_gt.append(rgba)

        masses_est_init = 0.2 * jnp.ones(vertices.shape[-2], dtype=jnp.float32)
        restitution_est_init = jnp.array([0.5], dtype=jnp.float32)

        params = {
            "mass_update": jnp.zeros(1, dtype=jnp.float32),
            "restitution_update": jnp.zeros(1, dtype=jnp.float32),
        }
        opt_state = _adam_init(params)

        losses = []
        est_masses = None
        est_restitution = None

        def loss_fn(params_inner):
            masses_cur = jnp.maximum(masses_est_init + params_inner["mass_update"][0], 0.0) * jnp.ones(
                vertices.shape[-2], dtype=jnp.float32
            )
            restitution_cur = restitution_est_init + params_inner["restitution_update"]
            body = RigidBody(
                vertices=vertices[0],
                masses=masses_cur,
                orientation=jnp.array(orientation[0], dtype=jnp.float32),
                friction_coefficient=float(fric[0]),
                restitution=restitution_cur[0],
            )
            body.add_external_force(gravity)
            sim_est = Simulator(
                [body], dtime=sim_dtime, engine=SemiImplicitEulerWithContacts()
            )
            imgs_est_inner = []
            for t in range(sim_steps):
                sim_est.step()
                if t % render_every == 0:
                    rgba = renderer.forward(
                        body.get_world_vertices()[None, :], faces, textures
                    )
                    imgs_est_inner.append(rgba)
            return sum(
                jnp.mean((est - gt) ** 2)
                for est, gt in zip(
                    imgs_est_inner[:: args.compare_every], imgs_gt[:: args.compare_every]
                )
            ) / len(imgs_est_inner[:: args.compare_every])

        grad_fn = jax.value_and_grad(loss_fn)

        for i in trange(args.optiters):
            loss_val, grads = grad_fn(params)
            lr = 1e-1
            if i >= 80:
                lr *= 0.25
            elif i >= 40:
                lr *= 0.5
            params, opt_state = _adam_step(params, grads, opt_state, lr=lr)
            losses.append(float(loss_val))
            masses_cur = np.array(
                jnp.maximum(masses_est_init + params["mass_update"][0], 0.0)
                * jnp.ones(vertices.shape[-2], dtype=jnp.float32)
            )
            restitution_cur = float(restitution_est_init[0] + params["restitution_update"][0])
            est_masses = masses_cur
            est_restitution = restitution_cur
            tqdm.write(
                f"Loss: {float(loss_val):.5f}, "
                f"Mass pred: {float(masses_cur.mean()):.5f} "
                f"Restitution pred: {restitution_cur:.5f}"
            )

        predicted_masses.append(float(est_masses.mean()))
        predicted_restitutions.append(est_restitution)

        print(f"Optimization time: {time.time() - starttime}")

    print("True masses:", true_masses)
    print("Predicted masses:", predicted_masses)

    if args.log:
        np.savetxt(os.path.join(logdir, "true_masses.txt"), true_masses)
        np.savetxt(os.path.join(logdir, "predicted_masses.txt"), predicted_masses)
        np.savetxt(os.path.join(logdir, "true_restitutions.txt"), true_restitutions)
        np.savetxt(os.path.join(logdir, "predicted_restitutions.txt"), predicted_restitutions)
