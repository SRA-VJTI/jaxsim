import math
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import trange

from gradsim import dflex as df


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

    from gradsim.renderutils import SoftRenderer
    from gradsim.utils.logging import write_imglist_to_gif

    sim_duration = 1.5  # seconds
    sim_substeps = 32
    sim_dt = (1.0 / 60.0) / sim_substeps
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    renderer = SoftRenderer(camera_mode="look_at")
    camera_distance = 5.0
    elevation = 0.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    render_steps = 60 * 4

    height = 1.5

    particle_inv_mass_gt = None
    particle_velocity_gt = None

    builder = df.sim.ModelBuilder()
    builder.add_cloth_grid(
        pos=(-2.0, height, 0.0),
        rot=df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 1.04),
        vel=(1.0, 0.0, 0.0),
        dim_x=20,
        dim_y=10,
        cell_x=0.1,
        cell_y=0.1,
        mass=1.0,
    )

    attach0 = 0
    attach1 = 20

    anchor0 = builder.add_particle(
        pos=np.array(builder.particle_x[attach0]) - np.array([1.0, 0.0, 0.0]),
        vel=(0.0, 0.0, 0.0),
        mass=0.0,
    )
    anchor1 = builder.add_particle(
        pos=np.array(builder.particle_x[attach1]) + np.array([1.0, 0.0, 0.0]),
        vel=(0.0, 0.0, 0.0),
        mass=0.0,
    )

    builder.add_spring(anchor0, attach0, 10000.0, 1000.0, 0)
    builder.add_spring(anchor1, attach1, 10000.0, 1000.0, 0)

    model = builder.finalize("cpu")
    model.tri_lambda = 10000.0
    model.tri_ka = 10000.0
    model.tri_kd = 100.0

    model.contact_ke = 1.0e4
    model.contact_kd = 1000.0
    model.contact_kf = 1000.0
    model.contact_mu = 0.5

    model.particle_radius = 0.01
    model.ground = False

    numparticles = model.particle_inv_mass.size
    # Use deterministic initialization instead of torch.rand_like
    inv_mass_vals = jnp.linspace(0.1, 1.0, numparticles, dtype=jnp.float32)
    model.particle_inv_mass = inv_mass_vals.at[-1].set(0.0).at[-2].set(0.0)

    particle_inv_mass_gt = model.particle_inv_mass
    particle_velocity_gt = model.particle_v

    integrator = df.sim.SemiImplicitIntegrator()

    sim_time = 0.0
    state = model.state()

    faces = model.tri_indices
    textures = jnp.concatenate(
        (
            jnp.ones((1, faces.shape[-2], 2, 1), dtype=jnp.float32),
            jnp.ones((1, faces.shape[-2], 2, 1), dtype=jnp.float32),
            jnp.zeros((1, faces.shape[-2], 2, 1), dtype=jnp.float32),
        ),
        axis=-1,
    )

    # Run GT simulation
    imgs_gt = []
    for i in trange(0, sim_steps):
        state = integrator.forward(model, state, sim_dt)
        sim_time += sim_dt

        if i % render_steps == 0:
            rgba = renderer.forward(
                state.q[None, :],
                faces[None, :],
                textures,
            )
            imgs_gt.append(rgba)

    cloth_path = Path("cache/cloth")
    cloth_path.mkdir(exist_ok=True)

    write_imglist_to_gif(
        imgs_gt, cloth_path / "gt.gif", imgformat="rgba", verbose=False
    )

    # Velocity optimization
    velocity_init = -0.01 * jnp.ones_like(particle_velocity_gt)
    params = {"velocity_update": jnp.zeros_like(velocity_init)}
    opt_state = _adam_init(params)
    epochs = 50
    save_gif_every = 1
    compare_every = 1

    def loss_fn(params_inner):
        builder2 = df.sim.ModelBuilder()
        builder2.add_cloth_grid(
            pos=(-2.0, height, 0.0),
            rot=df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 1.04),
            vel=(1.0, 0.0, 0.0),
            dim_x=20,
            dim_y=10,
            cell_x=0.1,
            cell_y=0.1,
            mass=1.0,
        )

        anchor0_ = builder2.add_particle(
            pos=np.array(builder2.particle_x[attach0]) - np.array([1.0, 0.0, 0.0]),
            vel=(0.0, 0.0, 0.0),
            mass=0.0,
        )
        anchor1_ = builder2.add_particle(
            pos=np.array(builder2.particle_x[attach1]) + np.array([1.0, 0.0, 0.0]),
            vel=(0.0, 0.0, 0.0),
            mass=0.0,
        )

        builder2.add_spring(anchor0_, attach0, 10000.0, 1000.0, 0)
        builder2.add_spring(anchor1_, attach1, 10000.0, 1000.0, 0)

        model2 = builder2.finalize("cpu")
        model2.tri_lambda = 10000.0
        model2.tri_ka = 10000.0
        model2.tri_kd = 100.0
        model2.contact_ke = 1.0e4
        model2.contact_kd = 1000.0
        model2.contact_kf = 1000.0
        model2.contact_mu = 0.5
        model2.particle_radius = 0.01
        model2.ground = False

        model2.particle_v = velocity_init + params_inner["velocity_update"]

        integrator2 = df.sim.SemiImplicitIntegrator()
        state2 = model2.state()
        faces2 = model2.tri_indices
        textures2 = jnp.concatenate(
            (
                jnp.ones((1, faces2.shape[-2], 2, 1), dtype=jnp.float32),
                jnp.ones((1, faces2.shape[-2], 2, 1), dtype=jnp.float32),
                jnp.zeros((1, faces2.shape[-2], 2, 1), dtype=jnp.float32),
            ),
            axis=-1,
        )

        imgs_inner = []
        for i in range(0, sim_steps):
            state2 = integrator2.forward(model2, state2, sim_dt)
            if i % render_steps == 0:
                rgba = renderer.forward(
                    state2.q[None, :],
                    faces2[None, :],
                    textures2,
                )
                imgs_inner.append(rgba)

        return sum(
            jnp.mean((est - gt) ** 2)
            for est, gt in zip(imgs_inner[::compare_every], imgs_gt[::compare_every])
        ) / len(imgs_inner[::compare_every])

    grad_fn = jax.value_and_grad(loss_fn)

    for e in range(epochs):
        lr = 5e-2
        if e in [10, 15, 20, 30]:
            lr /= 2 ** ([10, 15, 20, 30].index(e) + 1)

        loss_val, grads = grad_fn(params)
        params, opt_state = _adam_step(params, grads, opt_state, lr=lr)
        print("Loss:", float(loss_val))

        if (e % save_gif_every == 0) or (e == epochs - 1):
            # Render with current params for visualization
            vel_cur = velocity_init + params["velocity_update"]
            builder3 = df.sim.ModelBuilder()
            builder3.add_cloth_grid(
                pos=(-2.0, height, 0.0),
                rot=df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 1.04),
                vel=(1.0, 0.0, 0.0),
                dim_x=20,
                dim_y=10,
                cell_x=0.1,
                cell_y=0.1,
                mass=1.0,
            )
            model3 = builder3.finalize("cpu")
            model3.particle_v = vel_cur
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
            imgs_viz = []
            for i in range(0, sim_steps):
                state3 = integrator3.forward(model3, state3, sim_dt)
                if i % render_steps == 0:
                    rgba = renderer.forward(state3.q[None, :], faces3[None, :], textures3)
                    imgs_viz.append(rgba)
            write_imglist_to_gif(
                imgs_viz, f"cache/cloth/{e:05d}.gif", imgformat="rgba", verbose=False
            )
