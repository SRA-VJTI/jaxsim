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
try:
    from pxr import Usd, UsdGeom
except ImportError:
    pass


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


def write_meshes_to_file(vertices_across_time, faces, dirname):
    os.makedirs(dirname, exist_ok=True)
    for i, vertices in enumerate(vertices_across_time):
        np.savetxt(os.path.join(dirname, f"{i:03d}.txt"), vertices)
    np.savetxt(os.path.join(dirname, "faces.txt"), faces)


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
        default=os.path.join("cache", "control-walker"),
        help="Directory to store experiment logs in.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=os.path.join("sampledata", "usd", "walker.usda"),
        help="Path to input mesh file (.usda or .tet format).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=os.path.join("sampledata", "target_img_walker.png"),
        help="Path to target image.",
    )
    parser.add_argument(
        "--sim-duration",
        type=float,
        default=5.0,
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
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate.")
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

    logdir = os.path.join(args.logdir, args.expid)
    if args.log:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)

    sim_dt = (1.0 / args.physics_engine_rate) / args.sim_substeps
    sim_steps = int(args.sim_duration / sim_dt)
    sim_time = 0.0

    train_rate = 0.001

    phase_count = 4

    builder = df.sim.ModelBuilder()

    walker = Usd.Stage.Open(args.mesh)
    mesh = UsdGeom.Mesh(walker.GetPrimAtPath("/Grid/Grid"))

    points = mesh.GetPointsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()

    for p in points:
        builder.add_particle(tuple(p), (0.0, 0.0, 0.0), 1.0)

    for t in range(0, len(indices), 3):
        i = indices[t + 0]
        j = indices[t + 1]
        k = indices[t + 2]

        builder.add_triangle(i, j, k)

    model = builder.finalize("cpu")

    model.tri_ke = 10000.0
    model.tri_ka = 10000.0
    model.tri_kd = 100.0
    model.tri_lift = 0.0
    model.tri_drag = 0.0

    model.contact_ke = 1.0e4
    model.contact_kd = 1000.0
    model.contact_kf = 1000.0
    model.contact_mu = 0.5

    model.particle_radius = 0.01

    # Replace torch.nn.Sequential with weight matrix W
    params = {"W": jnp.zeros((phase_count, model.tri_count), dtype=jnp.float32)}
    opt_state = _adam_init(params)

    activation_strength = 0.2
    activation_penalty = 0.1

    integrator = df.sim.SemiImplicitIntegrator()

    renderer = SoftRenderer(camera_mode="look_at")
    camera_distance = 8.0
    elevation = 30.0
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

    target_image = imageio.imread(os.path.join("sampledata", "target_img_walker.png"))
    target_image = (jnp.array(target_image, dtype=jnp.float32) / 255.0).transpose(2, 0, 1)[None, :]

    render_every = 60 * 4

    losses = []
    position_xs = []

    def loss_fn(params_inner):
        sim_time_inner = 0.0
        state_inner = model.state()
        loss_inner = 0.0
        imgs_inner = []
        vertices_inner = []

        for i in range(0, sim_steps):
            # Build sinusoidal phase inputs
            phases = jnp.array([
                math.sin(20.0 * (sim_time_inner + 0.5 * p * math.pi))
                for p in range(phase_count)
            ], dtype=jnp.float32)

            model.tri_activations = jnp.tanh(phases @ params_inner["W"]) * activation_strength
            state_inner = integrator.forward(model, state_inner, sim_dt)

            sim_time_inner += sim_dt

            if i % render_every == 0 or i == sim_steps - 1:
                rgba = renderer.forward(
                    state_inner.q[None, :],
                    faces[None, :],
                    textures,
                )
                imgs_inner.append(rgba)
                vertices_inner.append(np.array(state_inner.q))

            com_vel = jnp.mean(state_inner.u, axis=0)

            if args.method == "physics-only":
                loss_inner = (
                    loss_inner
                    - com_vel[0]
                    + jnp.linalg.norm(model.tri_activations) * activation_penalty
                )

        if args.method == "gradsim":
            loss_inner = jnp.mean((imgs_inner[-1] - target_image) ** 2)

        return loss_inner, imgs_inner, vertices_inner, state_inner

    grad_fn = jax.value_and_grad(lambda p: loss_fn(p)[0])

    for e in trange(args.epochs):

        if args.method != "random":
            loss_val, grads = grad_fn(params)
            params, opt_state = _adam_step(params, grads, opt_state, lr=args.lr)
        else:
            loss_val = 0.0

        _, imgs, vertices, state_end = loss_fn(params)

        tqdm.write(f"Loss: {float(loss_val):.5}")
        tqdm.write(f"Iter: {e:03d}, Loss: {float(loss_val):.5}")

        losses.append(float(loss_val))
        position_xs.append(float(state_end.q.mean(axis=0)[0]))

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
            write_meshes_to_file(
                vertices,
                np.array(faces),
                os.path.join(logdir, f"vertices_{e:05d}")
            )

    if args.log:
        np.savetxt(os.path.join(logdir, "losses.txt"), losses)
        np.savetxt(os.path.join(logdir, "position_xs.txt"), position_xs)
