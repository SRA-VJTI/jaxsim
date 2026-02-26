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
        default=os.path.join("cache", "control-fem"),
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
        default=os.path.join("sampledata", "target_img_fem_gear.png"),
        help="Path to target image.",
    )
    parser.add_argument(
        "--sim-duration",
        type=float,
        default=3.0,
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
        "--epochs", type=int, default=20, help="Number of training iterations."
    )
    parser.add_argument("--lr", type=float, default=10, help="Learning rate.")
    parser.add_argument(
        "--method",
        type=str,
        choices=["random", "physics-only", "noisy-physics-only", "gradsim"],
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

    phase_count = 8
    phase_step = math.pi / phase_count * 2.0
    phase_freq = 2.5

    builder = df.sim.ModelBuilder()

    if args.mesh[-4:] == "usda":
        mesh = Usd.Stage.Open(args.mesh)
        geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/mesh"))
        points = geom.GetPointsAttr().Get()
        tet_indices = geom.GetPrim().GetAttribute("tetraIndices").Get()
    elif args.mesh[-3:] == "tet":
        points, tet_indices = read_tet_mesh(args.mesh)

    r = df.quat_multiply(
        df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.0),
        df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 0.0),
    )

    TET_KL = 1000.0
    TET_KM = 1000.0
    TET_KD = 1.0

    builder.add_soft_mesh(
        pos=(-4.0, 2.0, 0.0),
        rot=r,
        scale=1.0,
        vel=(1.5, 0.0, 0.0),
        vertices=points,
        indices=tet_indices,
        density=1.0,
        k_mu=TET_KM,
        k_lambda=TET_KL,
        k_damp=TET_KD,
    )

    model = builder.finalize("cpu")

    model.tet_kl = TET_KL
    model.tet_km = TET_KM
    model.tet_kd = TET_KD

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

    # Replace torch.nn.Sequential with weight matrix W
    # network: Linear(phase_count, tet_count, bias=False) + Tanh
    # forward: jnp.tanh(phases @ W) * activation_strength
    params = {"W": jnp.zeros((phase_count, model.tet_count), dtype=jnp.float32)}
    opt_state = _adam_init(params)

    activation_strength = 0.3
    activation_penalty = 0.0

    render_time = 0

    integrator = df.sim.SemiImplicitIntegrator()

    renderer = SoftRenderer(camera_mode="look_at")
    camera_distance = 13.0
    elevation = 30.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    faces = model.tri_indices
    textures = jnp.concatenate(
        (
            jnp.ones((1, faces.shape[-2], 2, 1), dtype=jnp.float32),
            jnp.ones((1, faces.shape[-2], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces.shape[-2], 2, 1), dtype=jnp.float32),
        ),
        axis=-1,
    )

    # Use deterministic target position
    target_position = jnp.array([
        float(np.random.rand() * 5.0 + 0.0) if args.method else 4.0,
        2.0,
        0.0,
    ], dtype=jnp.float32)
    target_image = None
    print("Target position:", target_position)

    gt_builder = df.sim.ModelBuilder()
    gt_builder.add_soft_mesh(
        pos=(float(target_position[0]), float(target_position[1]), float(target_position[2])),
        rot=r,
        scale=1.0,
        vel=(1.5, 0.0, 0.0),
        vertices=points,
        indices=tet_indices,
        density=1.0,
        k_mu=TET_KM,
        k_lambda=TET_KL,
        k_damp=TET_KD,
    )
    gt_model = gt_builder.finalize("cpu")
    gt_model.tet_kl = model.tet_kl
    gt_model.tet_km = model.tet_km
    gt_model.tet_kd = model.tet_kd
    gt_model.tri_ke = model.tri_ke
    gt_model.tri_ka = model.tri_ka
    gt_model.tri_kd = model.tri_kd
    gt_model.tri_kb = model.tri_kb
    gt_model.contact_ke = model.contact_ke
    gt_model.contact_kd = model.contact_kd
    gt_model.contact_kf = model.contact_kf
    gt_model.contact_mu = model.contact_mu
    gt_model.particle_radius = model.particle_radius
    gt_model.ground = model.ground
    gt_state = gt_model.state()
    target_image = renderer.forward(
        gt_state.q[None, :],
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

    losses = []
    position_errors = []

    render_every = 60

    def loss_fn(params_inner):
        sim_time_inner = 0.0
        state_inner = model.state()
        imgs_inner = []

        for i in range(0, sim_steps):
            phases = jnp.array([
                math.sin(phase_freq * (sim_time_inner + p * phase_step))
                for p in range(phase_count)
            ], dtype=jnp.float32)

            if args.method == "random":
                model.tet_activations = jnp.array(
                    (np.random.rand(model.tet_count) * 2 - 1) * activation_strength,
                    dtype=jnp.float32,
                )
            else:
                model.tet_activations = jnp.tanh(phases @ params_inner["W"]) * activation_strength

            state_inner = integrator.forward(model, state_inner, sim_dt)
            sim_time_inner += sim_dt

            if i % render_every == 0:
                rgba = renderer.forward(
                    state_inner.q[None, :],
                    faces[None, :],
                    textures,
                )
                imgs_inner.append(rgba)

        loss_inner = jnp.mean((imgs_inner[-1] - target_image) ** 2)
        return loss_inner, imgs_inner, state_inner

    grad_fn = jax.value_and_grad(lambda p: loss_fn(p)[0])

    try:
        for e in range(args.epochs):

            if args.method == "random":
                loss_val = 0.0
                _, imgs, state_end = loss_fn(params)
            elif args.method in ("physics-only", "noisy-physics-only"):
                # Use position-based loss
                def pos_loss_fn(p):
                    _, _, s = loss_fn(p)
                    pos_err = jnp.mean((jnp.mean(s.q, axis=0) - target_position) ** 2)
                    if args.method == "noisy-physics-only":
                        noise = jnp.array(
                            np.random.rand(3) * 0.1,
                            dtype=jnp.float32,
                        )
                        pos_err = jnp.mean((jnp.mean(s.q, axis=0) - (target_position + noise * target_position)) ** 2)
                    return pos_err
                pos_grad_fn = jax.value_and_grad(pos_loss_fn)
                loss_val, grads = pos_grad_fn(params)
                params, opt_state = _adam_step(params, grads, opt_state, lr=args.lr)
                _, imgs, state_end = loss_fn(params)
            else:
                loss_val, grads = grad_fn(params)
                params, opt_state = _adam_step(params, grads, opt_state, lr=args.lr)
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

    except KeyboardInterrupt:
        pass

    if args.log:
        np.savetxt(os.path.join(logdir, "losses.txt"), losses)
        np.savetxt(os.path.join(logdir, "position_errors.txt"), position_errors)
