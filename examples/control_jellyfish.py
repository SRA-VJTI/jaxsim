import argparse
import math
import os

import imageio
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm, trange

from gradsim import dflex as df
from gradsim.renderutils import SoftRenderer
from gradsim.utils.logging import write_imglist_to_gif
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", action="store_true", help="Only run inference.")

    args = parser.parse_args()

    sim_duration = 1  # seconds
    sim_substeps = 16
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    train_iters = 200
    train_rate = 0.01

    phase_count = 8
    phase_step = math.pi / phase_count * 2.0
    phase_freq = 2.5

    r = df.quat_multiply(
        df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.0),
        df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5),
    )

    builder = df.sim.ModelBuilder()

    mesh = Usd.Stage.Open("cache/usdassets/jellyfish.usda")
    geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/Icosphere/Icosphere"))

    points = geom.GetPointsAttr().Get()
    indices = geom.GetFaceVertexIndicesAttr().Get()
    counts = geom.GetFaceVertexCountsAttr().Get()

    face_materials = [-1] * len(counts)
    face_subsets = UsdGeom.Subset.GetAllGeomSubsets(geom)

    for i, s in enumerate(face_subsets):
        face_subset_indices = s.GetIndicesAttr().Get()

        for f in face_subset_indices:
            face_materials[f] = i

    active_material = 0
    active_scale = []

    def add_edge(f0, f1):
        if (
            face_materials[f0] == active_material
            and face_materials[f1] == active_material
        ):
            active_scale.append(1.0)
        else:
            active_scale.append(0.0)

    builder.add_cloth_mesh(
        pos=(0.0, 2.5, 0.0),
        rot=r,
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=points,
        indices=indices,
        edge_callback=add_edge,
        density=100.0,
    )

    model = builder.finalize("cpu")
    model.tri_lambda = 5000.0
    model.tri_ka = 5000.0
    model.tri_kd = 100.0
    model.tri_lift = 1000.0
    model.tri_drag = 0.0

    model.edge_ke = 20.0
    model.edge_kd = 1.0

    model.contact_ke = 1.0e4
    model.contact_kd = 0.0
    model.contact_kf = 1000.0
    model.contact_mu = 0.5

    model.particle_radius = 0.01
    model.ground = False
    model.gravity = jnp.array((0.0, 0.0, 0.0))

    # Store rest angle before training
    rest_angle = model.edge_rest_angle

    # Replace torch.nn.Sequential with weight matrix W
    # network: Linear(phase_count, edge_count, bias=False) + Tanh
    # forward: jnp.tanh(phases @ W) * activation_strength * activation_scale
    params = {"W": jnp.zeros((phase_count, model.edge_count), dtype=jnp.float32)}
    opt_state = _adam_init(params)

    activation_strength = math.pi * 0.3
    activation_scale = jnp.array(active_scale)
    activation_penalty = 0.0

    integrator = df.sim.SemiImplicitIntegrator()

    render_time = 0

    renderer = SoftRenderer(camera_mode="look_at")
    camera_distance = 10.0
    elevation = 0.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    faces = model.tri_indices
    textures = jnp.concatenate(
        (
            jnp.zeros((1, faces.shape[-2], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces.shape[-2], 2, 1), dtype=jnp.float32),
            jnp.ones((1, faces.shape[-2], 2, 1), dtype=jnp.float32),
        ),
        axis=-1,
    )

    epochs = 100
    validate_every = 10

    if args.inference:
        epochs = 1
        sim_duration = 25
        sim_dt = (1.0 / 60.0) / sim_substeps
        sim_steps = int(sim_duration / sim_dt)

    render_every = 60
    print_every = 60 * 16

    def loss_fn(params_inner):
        sim_time_inner = 0.0
        # Reset rest angle for this forward pass
        model.edge_rest_angle = rest_angle
        state_inner = model.state()
        loss_inner = 0.0
        imgs_inner = []

        for i in range(0, sim_steps):
            # Build sinusoidal input phases
            phases = jnp.array([
                math.sin(phase_freq * (sim_time_inner + p * phase_step))
                for p in range(phase_count)
            ], dtype=jnp.float32)

            # Compute activations (rest angles)
            activation = jnp.tanh(phases @ params_inner["W"]) * activation_strength * activation_scale
            model.edge_rest_angle = rest_angle + activation

            state_inner = integrator.forward(model, state_inner, sim_dt)
            sim_time_inner += sim_dt * sim_substeps

            com_loss = jnp.mean(state_inner.u * model.particle_mass[:, None], axis=0)
            act_loss = jnp.linalg.norm(activation) * activation_penalty

            loss_inner = loss_inner - com_loss[1] - act_loss

            if i % render_every == 0 or i == sim_steps - 1:
                rgba = renderer.forward(
                    state_inner.q[None, :],
                    faces[None, :],
                    textures,
                )
                imgs_inner.append(rgba)

        return loss_inner, imgs_inner

    grad_fn = jax.value_and_grad(lambda p: loss_fn(p)[0])

    for e in trange(epochs):

        if not args.inference:
            loss_val, grads = grad_fn(params)
            params, opt_state = _adam_step(params, grads, opt_state, lr=train_rate)
        else:
            loss_val = 0.0

        _, imgs = loss_fn(params)

        tqdm.write(f"Loss: {float(loss_val):.5}")

        render_time += 1
        if args.inference:
            filename = os.path.join("cache", "jellyfish", "debug", "inference")
        else:
            filename = os.path.join("cache", "jellyfish", "debug", f"{render_time:02d}")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        write_imglist_to_gif(imgs, f"{filename}.gif", imgformat="rgba", verbose=False)

        if args.inference:
            savepath = os.path.join("cache", "jellyfish", "debug", "inference.png")
        else:
            savepath = os.path.join(
                "cache", "jellyfish", "debug", f"last_frame_{render_time:02d}.png"
            )
        imageio.imwrite(
            savepath,
            (np.array(imgs[-1][0]).transpose(1, 2, 0) * 255).astype(np.uint8),
        )
        # Save weights as numpy array (replaces torch.save)
        np.save("cache/jellyfish/debug/model_W.npy", np.array(params["W"]))
