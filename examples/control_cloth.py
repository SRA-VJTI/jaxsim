import argparse
import json
import math
import os

import imageio
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm, trange

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


if __name__ == "__main__":

    dflex_base_parser = get_dflex_base_parser()
    parser = argparse.ArgumentParser(
        parents=[dflex_base_parser], conflict_handler="resolve"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Random seed"
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
        default=os.path.join("cache", "control-cloth"),
        help="Directory to store experiment logs in.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=os.path.join("sampledata", "usd", "prop.usda"),
        help="Path to input mesh file (.usda or .tet format).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=os.path.join("sampledata", "target_img_cloth.png"),
        help="Path to target image.",
    )
    parser.add_argument(
        "--sim-duration",
        type=float,
        default=1.5,
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
        default=16,
        help="Number of sub-steps to integrate, per 1 `step` of the simulation.",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training iterations."
    )
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate.")
    parser.add_argument(
        "--method",
        type=str,
        default="gradsim",
        choices=["random", "physics-only", "gradsim"],
        help="Method to use, to optimize for initial velocity."
    )
    parser.add_argument("--log", action="store_true", help="Log experiment data.")

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)

    logdir = os.path.join(args.logdir, args.expid)
    if args.log:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    sim_dt = (1.0 / args.physics_engine_rate) / args.sim_substeps
    sim_steps = int(args.sim_duration / sim_dt)
    sim_time = 0.0

    # Use deterministic values instead of random sampling
    height = 2.5
    clothdims = 16

    print(f"Cloth dims: {clothdims}")

    initial_vel = (0.0, 0.0, 0.0)
    if args.method == "random":
        initial_vel = (1.5, 0.0, 0.0)

    builder = df.sim.ModelBuilder()
    builder.add_cloth_grid(
        pos=(-5.0, height, 0.0),
        rot=df.quat_from_axis_angle((1.0, 0.5, 0.0), math.pi * 0.5),
        vel=initial_vel,
        dim_x=clothdims,
        dim_y=clothdims,
        cell_x=0.125,
        cell_y=0.125,
        mass=2.0,
    )

    model = builder.finalize("cpu")
    model.tri_lambda = 10000.0
    model.tri_ka = 10000.0
    model.tri_kd = 100.0
    model.tri_lift = 10.0
    model.tri_drag = 5.0

    model.contact_ke = 1.0e4
    model.contact_kd = 1000.0
    model.contact_kf = 1000.0
    model.contact_mu = 0.5

    model.particle_radius = 0.01
    model.ground = False

    integrator = df.sim.SemiImplicitIntegrator()

    renderer = SoftRenderer(camera_mode="look_at")
    camera_distance = 15.0
    elevation = 0.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    faces = model.tri_indices
    textures = jnp.concatenate(
        (
            jnp.zeros((1, faces.shape[-2], 2, 1), dtype=jnp.float32),
            jnp.ones((1, faces.shape[-2], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces.shape[-2], 2, 1), dtype=jnp.float32),
        ),
        axis=-1,
    )

    # Use deterministic target position instead of random
    target_position = jnp.array([2.0, -3.0, 0.0], dtype=jnp.float32)
    target_image = None

    builder_gt = df.sim.ModelBuilder()
    builder_gt.add_cloth_grid(
        pos=(float(target_position[0]), float(target_position[1]), float(target_position[2])),
        rot=df.quat_from_axis_angle((1.0, 0.5, 0.0), math.pi * 0.5),
        vel=initial_vel,
        dim_x=clothdims,
        dim_y=clothdims,
        cell_x=0.125,
        cell_y=0.125,
        mass=1.0,
    )

    model_gt = builder_gt.finalize("cpu")
    model_gt.tri_lambda = 10000.0
    model_gt.tri_ka = 10000.0
    model_gt.tri_kd = 100.0
    model_gt.tri_lift = 10.0
    model_gt.tri_drag = 5.0

    model_gt.contact_ke = 1.0e4
    model_gt.contact_kd = 1000.0
    model_gt.contact_kf = 1000.0
    model_gt.contact_mu = 0.5

    model_gt.particle_radius = 0.01
    model_gt.ground = False
    state_gt = model_gt.state()
    target_image = renderer.forward(
        state_gt.q[None, :],
        faces[None, :],
        textures,
    )
    if args.log:
        imageio.imwrite(
            os.path.join(logdir, "target_image.png"),
            (np.array(target_image[0]).transpose(1, 2, 0) * 255).astype(np.uint8),
        )
        np.savetxt(
            os.path.join(logdir, "target_position.txt"),
            np.array(target_position),
        )

    # Initial velocity parameter
    params = {"velocity": jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)}
    opt_state = _adam_init(params)

    render_every = 60

    losses = []
    position_errors = []
    initial_velocities = []

    def loss_fn(params_inner):
        vel = params_inner["velocity"]
        state_inner = model.state()
        state_inner.u = state_inner.u + vel

        loss_inner = 0.0
        imgs_inner = []

        for i in range(0, sim_steps):
            state_inner = integrator.forward(model, state_inner, sim_dt)

            if i % render_every == 0 or i == sim_steps - 1:
                rgba = renderer.forward(
                    state_inner.q[None, :],
                    faces[None, :],
                    textures,
                )
                imgs_inner.append(rgba)

            com_pos = jnp.mean(state_inner.q, axis=0) + 1e-16

            if i % render_every == 0 or i == sim_steps - 1:
                if args.method == "physics-only":
                    loss_inner = loss_inner + jnp.mean((com_pos - target_position) ** 2)
                if args.method == "gradsim":
                    loss_inner = loss_inner + jnp.mean(
                        (imgs_inner[-1] + 1e-16 - target_image) ** 2
                    ) + 1e-5

        return loss_inner, imgs_inner, state_inner

    grad_fn = jax.value_and_grad(lambda p: loss_fn(p)[0])

    try:
        for e in trange(args.epochs):

            initial_velocities.append(list(np.array(params["velocity"])))

            if args.method != "random":
                loss_val, grads = grad_fn(params)
                params, opt_state = _adam_step(params, grads, opt_state, lr=args.lr)
            else:
                loss_val = 0.0

            # Run forward for logging/metrics
            _, imgs, state_end = loss_fn(params)

            position_error = float(jnp.mean((jnp.mean(state_end.q, axis=0) - target_position) ** 2))

            tqdm.write(f"Loss: {float(loss_val):.5}")
            tqdm.write(f"Position error: {position_error:.5}")

            losses.append(float(loss_val))
            position_errors.append(position_error)

            if args.log:
                write_imglist_to_gif(
                    imgs,
                    os.path.join(logdir, f"{e:02d}.gif"),
                    imgformat="rgba",
                    verbose=False,
                )
                write_imglist_to_dir(
                    imgs, os.path.join(logdir, f"{e:02d}"), imgformat="rgba",
                )
                imageio.imwrite(
                    os.path.join(logdir, f"last_frame_{e:02d}.png"),
                    (np.array(imgs[-1][0]).transpose(1, 2, 0) * 255).astype(np.uint8),
                )

    except Exception as exc:
        if args.log:
            with open(os.path.join(logdir, "exceptions.txt"), "w") as f:
                f.write("Exception occured!\n")

    if args.log:
        np.savetxt(os.path.join(logdir, "losses.txt"), losses)
        np.savetxt(os.path.join(logdir, "position_errors.txt"), position_errors)
        np.savetxt(os.path.join(logdir, "initial_velocities.txt"), np.array(initial_velocities))
