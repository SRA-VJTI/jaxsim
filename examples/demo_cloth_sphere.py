"""Cloth draping over a static sphere — lightweight CPU demo.

Physics: analytic sphere-particle projection constraint applied after every
integration substep.  No rigid-body collision pipeline needed — just compute
the distance from each cloth particle to the sphere center, and if it's inside
the sphere, push it out radially and zero the inward velocity component.

This gives correct, stable draping at a fraction of the cost.

Render: 128 × 128 px, 10 fps, ~60 frames GIF.
"""
import math
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from tqdm import trange

from jaxsim import dflex as df


# ── sphere mesh helper ────────────────────────────────────────────────────────

def make_sphere_mesh(center, radius, n_lat=14, n_lon=20):
    """UV sphere vertices + faces (outward winding)."""
    cx, cy, cz = center
    verts = []
    for i in range(n_lat + 1):
        phi = math.pi * i / n_lat          # 0 (north) → π (south)
        sp, cp = math.sin(phi), math.cos(phi)
        for j in range(n_lon + 1):
            theta = 2.0 * math.pi * j / n_lon
            verts.append([
                cx + radius * sp * math.cos(theta),
                cy + radius * cp,
                cz + radius * sp * math.sin(theta),
            ])
    s = n_lon + 1
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            v0 = i * s + j
            v1 = v0 + 1
            v2 = (i + 1) * s + j
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


# ── analytic sphere constraint ────────────────────────────────────────────────

def apply_sphere_constraint(q, u, center, radius, p_radius=0.02):
    """Project cloth particles outside the sphere; kill inward velocity."""
    sc    = jnp.array(center, dtype=jnp.float32)
    delta = q - sc[None, :]                                 # (N, 3)
    dist  = jnp.linalg.norm(delta, axis=-1, keepdims=True) # (N, 1)
    thresh = radius + p_radius
    d_hat  = delta / jnp.maximum(dist, 1e-8)
    inside = dist < thresh                                  # (N, 1) bool

    # position: push inside particles to sphere surface
    q_out = jnp.where(inside, sc[None, :] + d_hat * thresh, q)

    # velocity: zero the component pointing into the sphere
    v_rad = jnp.sum(u * d_hat, axis=-1, keepdims=True)     # radial component
    u_out = jnp.where(inside & (v_rad < 0.0), u - d_hat * v_rad, u)

    return q_out, u_out


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    from jaxsim.renderutils import SoftRenderer
    from jaxsim.utils.logging import write_imglist_to_gif

    # ── sim parameters (lightweight for CPU) ─────────────────────────────────
    sim_duration = 2.0           # seconds — enough to watch cloth settle
    sim_substeps = 16
    sim_dt       = (1.0 / 60.0) / sim_substeps
    sim_steps    = int(sim_duration / sim_dt)
    render_every = 32            # one frame per 32 substeps → ~60 frames total

    # ── sphere ────────────────────────────────────────────────────────────────
    SPHERE_CENTER = (0.0, 0.6, 0.0)   # sits on the ground (y = 0)
    SPHERE_RADIUS = 0.6

    # ── cloth grid ────────────────────────────────────────────────────────────
    dim_x, dim_y = 10, 10
    cell         = 0.14              # 1.4 m × 1.4 m cloth
    start_y      = 2.0               # drop height

    rot       = df.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi * 0.5)
    start_pos = (-dim_x * cell / 2, start_y, dim_y * cell / 2)

    builder = df.sim.ModelBuilder()
    builder.add_cloth_grid(
        pos=start_pos,
        rot=rot,
        vel=(0.0, 0.0, 0.0),
        dim_x=dim_x,
        dim_y=dim_y,
        cell_x=cell,
        cell_y=cell,
        mass=0.3,
    )

    model = builder.finalize("cpu")

    # cloth material — soft enough to drape, stiff enough not to explode
    model.tri_ke  = 2500.0
    model.tri_ka  = 2500.0
    model.tri_kd  = 45.0
    model.edge_ke = 80.0       # gentle bending resistance
    model.edge_kd = 0.0

    # ground (y = 0) contact
    model.contact_ke = 5000.0
    model.contact_kd = 500.0
    model.contact_kf = 500.0
    model.contact_mu = 0.6
    model.particle_radius = 0.02
    model.ground = True

    integrator = df.sim.SemiImplicitIntegrator()
    state      = model.state()

    # ── renderer ──────────────────────────────────────────────────────────────
    renderer = SoftRenderer(
        camera_mode="look_at",
        image_size=512,
        bg_color=[0.0, 0.0, 0.0],   # black → output is premultiplied alpha
        light_intensity_ambient=0.35,
        light_intensity_directional=0.75,
        light_direction=[0.5, 1.2, -0.5],
        light_color_directional=[1.0, 0.95, 0.88],
        light_color_ambient=[0.5, 0.6, 0.95],
        anti_aliasing=False,
    )
    renderer.set_eye_from_angles(distance=5.0, elevation=28.0, azimuth=42.0)

    # ── sphere mesh (static — vertices never change) ───────────────────────
    sph_verts_np, sph_faces_np = make_sphere_mesh(SPHERE_CENTER, SPHERE_RADIUS, n_lat=6, n_lon=8)
    sph_verts_j  = jnp.array(sph_verts_np)
    sph_faces_j  = jnp.array(sph_faces_np)
    n_sph_faces  = sph_faces_np.shape[0]

    # blue sphere
    sph_tex = jnp.concatenate([
        jnp.ones((1, n_sph_faces, 2, 1), dtype=jnp.float32) * 0.15,
        jnp.ones((1, n_sph_faces, 2, 1), dtype=jnp.float32) * 0.40,
        jnp.ones((1, n_sph_faces, 2, 1), dtype=jnp.float32) * 0.90,
    ], axis=-1)

    # pre-batch sphere (never changes)
    sph_verts_batch = sph_verts_j[None, :]    # (1, V, 3)
    sph_faces_batch = sph_faces_j[None, :]    # (1, F, 3)

    # ── cloth mesh ────────────────────────────────────────────────────────────
    cloth_faces   = model.tri_indices          # (F, 3)
    n_cloth_faces = cloth_faces.shape[0]

    # orange cloth
    cloth_tex = jnp.concatenate([
        jnp.ones((1, n_cloth_faces, 2, 1), dtype=jnp.float32) * 0.95,
        jnp.ones((1, n_cloth_faces, 2, 1), dtype=jnp.float32) * 0.50,
        jnp.ones((1, n_cloth_faces, 2, 1), dtype=jnp.float32) * 0.05,
    ], axis=-1)

    cloth_faces_batch = cloth_faces[None, :]  # (1, F, 3)

    # ── simulate + capture ────────────────────────────────────────────────────
    imgs = []
    for i in trange(sim_steps, desc="Simulating"):
        state = integrator.forward(model, state, sim_dt)

        # analytic sphere collision: push particles off the sphere surface
        state.q, state.u = apply_sphere_constraint(
            state.q, state.u,
            SPHERE_CENTER, SPHERE_RADIUS,
            p_radius=model.particle_radius,
        )

        if i % render_every == 0:
            # Render each object with black background so output is premultiplied
            # alpha: rgba[:, :3] = color * alpha (no bg baked in).
            rgba_sph   = renderer.forward(sph_verts_batch, sph_faces_batch, sph_tex)
            rgba_cloth = renderer.forward(state.q[None, :], cloth_faces_batch, cloth_tex)

            sphere_a = rgba_sph[:, 3:4]    # (1, 1, H, W)
            cloth_a  = rgba_cloth[:, 3:4]  # (1, 1, H, W)

            # Porter-Duff "cloth over sphere" in premultiplied alpha space.
            # combined_premul = cloth_premul + sphere_premul * (1 - cloth_a)
            combined_premul = rgba_cloth[:, :3] + rgba_sph[:, :3] * (1.0 - cloth_a)
            combined_a      = cloth_a + sphere_a * (1.0 - cloth_a)

            # Composite over actual scene background.
            bg = jnp.array([0.10, 0.10, 0.14], dtype=jnp.float32)[None, :, None, None]
            rgb_final = combined_premul + bg * (1.0 - combined_a)
            rgba = jnp.concatenate([rgb_final, combined_a], axis=1)
            imgs.append(rgba)

    # ── save GIF ──────────────────────────────────────────────────────────────
    out_path = Path("cache/cloth_sphere")
    out_path.mkdir(parents=True, exist_ok=True)
    gif_path = out_path / "drop.gif"

    write_imglist_to_gif(imgs, str(gif_path), imgformat="rgba", fps=20, verbose=True)
