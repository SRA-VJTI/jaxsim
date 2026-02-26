import argparse
import json
import math
import os

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import trange

from argparsers import get_dflex_base_parser
from gradsim import dflex as df
from gradsim.renderutils import SoftRenderer
from gradsim.utils.logging import write_imglist_to_dir, write_imglist_to_gif


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


def read_tet_mesh(filepath):
    vertices = []
    faces = []
    with open(filepath, "r") as mesh:
        for line in mesh:
            data = line.split()
            if len(data) == 0:
                continue
            if data[0] == "v":
                vertices.append([float(d) for d in data[1:]])
            elif data[0] == "t":
                faces.append([int(d) for d in data[1:]])
    vertices = [tuple(v) for v in vertices]
    vertices = np.asarray(vertices).astype(np.float32)
    faces = [f for face in faces for f in face]
    return vertices, faces


if __name__ == "__main__":

    dflex_base_parser = get_dflex_base_parser()
    parser = argparse.ArgumentParser(
        parents=[dflex_base_parser], conflict_handler="resolve"
    )

    parser.add_argument(
        "--expid",
        type=str,
        default="default",
        help="Unique string identifier for this experiment.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=os.path.join("cache", "demo-fem"),
        help="Directory to store experiment logs in.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=os.path.join("sampledata", "tet", "icosphere.tet"),
        help="Path to input mesh file (.tet format).",
    )
    parser.add_argument(
        "--sim-duration",
        type=float,
        default=2.0,
        help="Duration of the simulation episode.",
    )
    parser.add_argument(
        "--physics-engine-rate",
        type=int,
        default=60,
        help="Number of physics engine `steps` per 1 second of simulator time.",
    )
    parser.add_argument(
        "--sim-substeps",
        type=int,
        default=32,
        help="Number of sub-steps to integrate, per 1 `step` of the simulation.",
    )
    parser.add_argument(
        "--epochs", type=int, default=40, help="Number of training iterations."
    )
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate.")
    parser.add_argument(
        "--method",
        type=str,
        default="gradsim",
        choices=["noisy-physics-only", "physics-only", "gradsim"],
        help="Method to use, to optimize for initial velocity."
    )
    parser.add_argument(
        "--compare-every",
        type=int,
        default=1,
        help="Interval at which video frames are compared.",
    )
    parser.add_argument("--log", action="store_true", help="Log experiment data.")

    args = parser.parse_args()
    print(args)

    logdir = os.path.join(args.logdir, args.expid)
    if args.log:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    sim_dt = (1.0 / args.physics_engine_rate) / args.sim_substeps
    sim_steps = int(args.sim_duration / sim_dt)
    sim_time = 0.0

    renderer = SoftRenderer(camera_mode="look_at")
    camera_distance = 8.0
    elevation = 30.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    render_steps = args.sim_substeps

    points, tet_indices = read_tet_mesh(args.mesh)

    r = df.quat_multiply(
        df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.0),
        df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 0.0),
    )

    # Use a deterministic initial x-velocity
    vx_init = 2.0

    imgs_gt = []
    particle_inv_mass_gt = None

    builder_gt = df.sim.ModelBuilder()
    builder_gt.add_soft_mesh(
        pos=(-2.0, 2.0, 0.0),
        rot=r,
        scale=1.0,
        vel=(vx_init, 0.0, 0.0),
        vertices=points,
        indices=tet_indices,
        density=10.0,
    )

    model_gt = builder_gt.finalize("cpu")

    model_gt.tet_kl = 1000.0
    model_gt.tet_km = 1000.0
    model_gt.tet_kd = 1.0

    model_gt.tri_ke = 0.0
    model_gt.tri_ka = 0.0
    model_gt.tri_kd = 0.0
    model_gt.tri_kb = 0.0

    model_gt.contact_ke = 1.0e4
    model_gt.contact_kd = 1.0
    model_gt.contact_kf = 10.0
    model_gt.contact_mu = 0.5

    model_gt.particle_radius = 0.05
    model_gt.ground = True

    particle_inv_mass_gt = model_gt.particle_inv_mass

    integrator = df.sim.SemiImplicitIntegrator()

    state_gt = model_gt.state()

    faces = model_gt.tri_indices
    textures = jnp.concatenate(
        (
            jnp.ones((1, faces.shape[-2], 2, 1), dtype=jnp.float32),
            jnp.ones((1, faces.shape[-2], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces.shape[-2], 2, 1), dtype=jnp.float32),
        ),
        axis=-1,
    )

    imgs_gt = []
    positions_gt = []
    for i in trange(0, sim_steps):
        state_gt = integrator.forward(model_gt, state_gt, sim_dt)
        sim_time += sim_dt

        if i % render_steps == 0:
            rgba = renderer.forward(
                state_gt.q[None, :],
                faces[None, :],
                textures,
            )
            imgs_gt.append(rgba)
            positions_gt.append(state_gt.q)

    if args.log:
        write_imglist_to_gif(
            imgs_gt,
            os.path.join(logdir, "gt.gif"),
            imgformat="rgba",
            verbose=False,
        )
        write_imglist_to_dir(
            imgs_gt, os.path.join(logdir, "gt"), imgformat="rgba",
        )
        np.savetxt(
            os.path.join(logdir, "mass_gt.txt"),
            np.array(particle_inv_mass_gt),
        )
        np.savetxt(
            os.path.join(logdir, "vertices.txt"),
            np.array(state_gt.q)
        )
        np.savetxt(
            os.path.join(logdir, "face.txt"),
            np.array(faces)
        )

    # Use deterministic initialization (instead of torch.rand_like)
    inv_mass_init = jnp.maximum(
        particle_inv_mass_gt + 0.1 * jnp.ones_like(particle_inv_mass_gt), 0.0
    )
    params = {"update": jnp.zeros_like(inv_mass_init)}
    opt_state = _adam_init(params)

    save_gif_every = 1

    losses = []
    inv_mass_errors = []
    mass_errors = []

    def loss_fn(params_inner):
        inv_mass_cur = jnp.maximum(inv_mass_init + params_inner["update"], 0.0)

        builder = df.sim.ModelBuilder()
        builder.add_soft_mesh(
            pos=(-2.0, 2.0, 0.0),
            rot=r,
            scale=1.0,
            vel=(vx_init, 0.0, 0.0),
            vertices=points,
            indices=tet_indices,
            density=10.0,
        )

        model = builder.finalize("cpu")

        model.tet_kl = 1000.0
        model.tet_km = 1000.0
        model.tet_kd = 1.0

        model.tri_ke = 0.0
        model.tri_ka = 0.0
        model.tri_kd = 0.0
        model.tri_kb = 0.0

        model.contact_ke = 1.0e4
        model.contact_kd = 1.0
        model.contact_kf = 10.0
        model.contact_mu = 0.5

        model.particle_radius = 0.05
        model.ground = True
        model.particle_inv_mass = inv_mass_cur

        integrator2 = df.sim.SemiImplicitIntegrator()
        state2 = model.state()
        faces2 = model.tri_indices
        textures2 = jnp.concatenate(
            (
                jnp.ones((1, faces2.shape[-2], 2, 1), dtype=jnp.float32),
                jnp.ones((1, faces2.shape[-2], 2, 1), dtype=jnp.float32),
                jnp.zeros((1, faces2.shape[-2], 2, 1), dtype=jnp.float32),
            ),
            axis=-1,
        )

        imgs_inner = []
        positions_inner = []
        for i in range(0, sim_steps):
            state2 = integrator2.forward(model, state2, sim_dt)
            if i % render_steps == 0:
                rgba = renderer.forward(
                    state2.q[None, :],
                    faces2[None, :],
                    textures2,
                )
                imgs_inner.append(rgba)
                positions_inner.append(state2.q)

        if args.method == "gradsim":
            return sum(
                jnp.mean((est - gt) ** 2)
                for est, gt in zip(
                    imgs_inner[::args.compare_every],
                    imgs_gt[::args.compare_every]
                )
            ) / len(imgs_inner[::args.compare_every])
        elif args.method in ("physics-only", "noisy-physics-only"):
            return sum(
                jnp.mean((est - gt) ** 2)
                for est, gt in zip(
                    positions_inner[::args.compare_every],
                    positions_gt[::args.compare_every]
                )
            ) / len(positions_inner[::args.compare_every])

    grad_fn = jax.value_and_grad(loss_fn)

    try:
        for e in range(args.epochs):
            loss_val, grads = grad_fn(params)
            params, opt_state = _adam_step(params, grads, opt_state, lr=args.lr)

            inv_mass_cur = jnp.maximum(inv_mass_init + params["update"], 0.0)
            inv_mass_err = float(jnp.mean((inv_mass_cur - particle_inv_mass_gt) ** 2))
            mass_err = float(jnp.mean(
                (1.0 / (inv_mass_cur + 1e-6) - 1.0 / (particle_inv_mass_gt + 1e-6)) ** 2
            ))

            print(
                f"[EPOCH: {e:03d}] "
                f"Loss: {float(loss_val):.5f} (Inv) Mass err: {inv_mass_err:.5f} "
                f"Mass err: {mass_err:.5f}"
            )

            losses.append(float(loss_val))
            inv_mass_errors.append(inv_mass_err)
            mass_errors.append(mass_err)

            if args.log and ((e % save_gif_every == 0) or (e == args.epochs - 1)):
                # Re-run forward to get images for logging
                builder3 = df.sim.ModelBuilder()
                builder3.add_soft_mesh(
                    pos=(-2.0, 2.0, 0.0),
                    rot=r,
                    scale=1.0,
                    vel=(vx_init, 0.0, 0.0),
                    vertices=points,
                    indices=tet_indices,
                    density=10.0,
                )
                model3 = builder3.finalize("cpu")
                model3.tet_kl = 1000.0
                model3.tet_km = 1000.0
                model3.tet_kd = 1.0
                model3.tri_ke = 0.0
                model3.tri_ka = 0.0
                model3.tri_kd = 0.0
                model3.tri_kb = 0.0
                model3.contact_ke = 1.0e4
                model3.contact_kd = 1.0
                model3.contact_kf = 10.0
                model3.contact_mu = 0.5
                model3.particle_radius = 0.05
                model3.ground = True
                model3.particle_inv_mass = jnp.maximum(inv_mass_init + params["update"], 0.0)
                integrator3 = df.sim.SemiImplicitIntegrator()
                state3 = model3.state()
                faces3 = model3.tri_indices
                textures3 = jnp.concatenate(
                    (
                        jnp.ones((1, faces3.shape[-2], 2, 1), dtype=jnp.float32),
                        jnp.ones((1, faces3.shape[-2], 2, 1), dtype=jnp.float32),
                        jnp.zeros((1, faces3.shape[-2], 2, 1), dtype=jnp.float32),
                    ),
                    axis=-1,
                )
                imgs_log = []
                for i in range(0, sim_steps):
                    state3 = integrator3.forward(model3, state3, sim_dt)
                    if i % render_steps == 0:
                        rgba = renderer.forward(state3.q[None, :], faces3[None, :], textures3)
                        imgs_log.append(rgba)
                write_imglist_to_gif(
                    imgs_log, os.path.join(logdir, f"{e:05d}.gif"), imgformat="rgba"
                )
                write_imglist_to_dir(
                    imgs_log, os.path.join(logdir, f"{e:05d}"), imgformat="rgba"
                )
                np.savetxt(
                    os.path.join(logdir, f"mass_{e:05d}.txt"),
                    np.array(jnp.maximum(inv_mass_init + params["update"], 0.0)),
                )

    except KeyboardInterrupt:
        pass

    if args.log:
        np.savetxt(os.path.join(logdir, "losses.txt"), losses)
        np.savetxt(os.path.join(logdir, "inv_mass_errors.txt"), inv_mass_errors)
        np.savetxt(os.path.join(logdir, "mass_errors.txt"), mass_errors)
