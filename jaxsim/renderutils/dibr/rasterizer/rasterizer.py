# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the MIT License (see original file for full text).
#
# Pure JAX Z-buffer rasterizer replacing the CUDA LinearRasterizer.
# Gradients flow automatically via JAX autodiff.

from __future__ import division, print_function

import jax
import jax.numpy as jnp


def _seg_sq(px_, py_, ax, ay, bx, by):
    """Squared distance from point (px_, py_) to segment [a, b]."""
    abx = bx - ax
    aby = by - ay
    apx = px_ - ax
    apy = py_ - ay
    denom = abx * abx + aby * aby + 1e-20
    t = jnp.clip((apx * abx + apy * aby) / denom, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    return (px_ - cx) ** 2 + (py_ - cy) ** 2


def linear_rasterizer(
    height,
    width,
    points3d_bxfx9,
    points2d_bxfx6,
    normalz_bxfx1,
    vertex_attr_bxfx3d,
    expand=0.02,
    knum=30,
    multiplier=1000.0,
    delta=7000.0,
):
    r"""Pure JAX differentiable Z-buffer rasterizer (replaces CUDA LinearRasterizer).

    Args:
        height (int): Output image height H.
        width (int): Output image width W.
        points3d_bxfx9 (jnp.ndarray): (B, F, 9) per-face 3-D vertices in
            camera space [v0.xyz, v1.xyz, v2.xyz].  z < 0 for visible objects
            (OpenGL convention: camera looks in -Z direction).
        points2d_bxfx6 (jnp.ndarray): (B, F, 6) per-face projected 2-D screen
            coords [v0.xy, v1.xy, v2.xy] in NDC-like space.  After ×
            ``multiplier`` they lie in pixel units centred at (0, 0).
        normalz_bxfx1 (jnp.ndarray): (B, F, 1) z-component of face normal.
            Positive ⇒ front-facing.
        vertex_attr_bxfx3d (jnp.ndarray): (B, F, 3·D) packed per-vertex
            attributes [v0_attrs(D), v1_attrs(D), v2_attrs(D)].
        expand (float): Unused (kept for API compatibility).
        knum (int): Unused (kept for API compatibility).
        multiplier (float): Scale factor for 2-D coords (default 1000).
        delta (float): Sharpness of soft-probability sigmoid (default 7000).

    Returns:
        imfeat_bxhxwxd (jnp.ndarray): (B, H, W, D) barycentric-interpolated
            vertex attributes at each pixel.
        improb_bxhxwx1 (jnp.ndarray): (B, H, W, 1) soft occupancy probability.
    """
    H, W = height, width
    B, F = points3d_bxfx9.shape[:2]
    D = vertex_attr_bxfx3d.shape[2] // 3

    # ── scale 2-D coords to pixel space (centred at 0) ──────────────────────
    pts2d = (multiplier * points2d_bxfx6).reshape(B, F, 3, 2)  # (B, F, 3, 2)

    # Per-face 2-D vertex positions: (B, F, 1, 1)
    v0x = pts2d[:, :, 0, 0, None, None]
    v0y = pts2d[:, :, 0, 1, None, None]
    v1x = pts2d[:, :, 1, 0, None, None]
    v1y = pts2d[:, :, 1, 1, None, None]
    v2x = pts2d[:, :, 2, 0, None, None]
    v2y = pts2d[:, :, 2, 1, None, None]

    # ── pixel grid (pixel centres; x right, y up) ───────────────────────────
    xs = jnp.linspace(-W / 2.0 + 0.5, W / 2.0 - 0.5, W)           # (W,)
    ys = jnp.linspace(H / 2.0 - 0.5, -H / 2.0 + 0.5, H)           # (H,) top→bottom
    px, py = jnp.meshgrid(xs, ys)                                    # (H, W)
    px = px[None, None]   # (1, 1, H, W)
    py = py[None, None]

    # ── inside / outside test for every face × pixel ─────────────────────────
    d0 = (v1x - v0x) * (py - v0y) - (v1y - v0y) * (px - v0x)      # (B, F, H, W)
    d1 = (v2x - v1x) * (py - v1y) - (v2y - v1y) * (px - v1x)
    d2 = (v0x - v2x) * (py - v2y) - (v0y - v2y) * (px - v2x)
    inside = (((d0 >= 0) & (d1 >= 0) & (d2 >= 0)) |
              ((d0 <= 0) & (d1 <= 0) & (d2 <= 0)))                  # (B, F, H, W)

    # ── front-face culling (normalz > 0 → front-facing) ──────────────────────
    is_front = (normalz_bxfx1[:, :, 0] > 0)[:, :, None, None]      # (B, F, 1, 1)
    inside_and_front = inside & is_front                             # (B, F, H, W)

    # ── Z-buffer: centroid depth (camera z — larger = closer) ────────────────
    z_face = (points3d_bxfx9[:, :, 2] +
              points3d_bxfx9[:, :, 5] +
              points3d_bxfx9[:, :, 8]) / 3.0                        # (B, F)
    z_face = z_face[:, :, None, None]                                # (B, F, 1, 1)

    depth_map = jnp.where(
        inside_and_front,
        jnp.broadcast_to(z_face, (B, F, H, W)),
        -jnp.inf
    )                                                                 # (B, F, H, W)

    best_face = jnp.argmax(depth_map, axis=1)                        # (B, H, W)
    any_coverage = jnp.any(inside_and_front, axis=1)                 # (B, H, W)

    # ── gather winning-face 2-D vertices ─────────────────────────────────────
    b_idx = jnp.arange(B)[:, None, None]                             # (B, 1, 1)
    pts2d_best = pts2d[b_idx, best_face]                             # (B, H, W, 3, 2)

    v0x_b = pts2d_best[..., 0, 0]   # (B, H, W)
    v0y_b = pts2d_best[..., 0, 1]
    v1x_b = pts2d_best[..., 1, 0]
    v1y_b = pts2d_best[..., 1, 1]
    v2x_b = pts2d_best[..., 2, 0]
    v2y_b = pts2d_best[..., 2, 1]

    # ── pixel coords broadcast to (B, H, W) ──────────────────────────────────
    px_g = jnp.broadcast_to(px[0, 0][None], (B, H, W))              # (B, H, W)
    py_g = jnp.broadcast_to(py[0, 0][None], (B, H, W))

    # ── barycentric coordinates at each pixel for the best face ──────────────
    denom = ((v1x_b - v0x_b) * (v2y_b - v0y_b) -
             (v1y_b - v0y_b) * (v2x_b - v0x_b) + 1e-20)
    bary1 = ((px_g - v0x_b) * (v2y_b - v0y_b) -
             (py_g - v0y_b) * (v2x_b - v0x_b)) / denom
    bary2 = ((v1x_b - v0x_b) * (py_g - v0y_b) -
             (v1y_b - v0y_b) * (px_g - v0x_b)) / denom
    bary0 = 1.0 - bary1 - bary2
    # Clamp and renormalise (gracefully handles pixels outside the triangle)
    bary0 = jnp.clip(bary0, 0.0, 1.0)
    bary1 = jnp.clip(bary1, 0.0, 1.0)
    bary2 = jnp.clip(bary2, 0.0, 1.0)
    bary_s = bary0 + bary1 + bary2 + 1e-20
    bary0 /= bary_s
    bary1 /= bary_s
    bary2 /= bary_s

    # ── interpolate vertex attributes ─────────────────────────────────────────
    attrs = vertex_attr_bxfx3d.reshape(B, F, 3, D)                  # (B, F, 3, D)
    attrs_best = attrs[b_idx, best_face]                             # (B, H, W, 3, D)

    imfeat = (bary0[..., None] * attrs_best[..., 0, :] +
              bary1[..., None] * attrs_best[..., 1, :] +
              bary2[..., None] * attrs_best[..., 2, :])              # (B, H, W, D)
    imfeat = jnp.where(any_coverage[..., None], imfeat, 0.0)

    # ── soft probability (sigmoid of signed distance in pixel space) ──────────
    d01 = _seg_sq(px_g, py_g, v0x_b, v0y_b, v1x_b, v1y_b)
    d12 = _seg_sq(px_g, py_g, v1x_b, v1y_b, v2x_b, v2y_b)
    d20 = _seg_sq(px_g, py_g, v2x_b, v2y_b, v0x_b, v0y_b)
    d_sq = jnp.minimum(jnp.minimum(d01, d12), d20)

    # Inside test for winning face
    d0b = (v1x_b - v0x_b) * (py_g - v0y_b) - (v1y_b - v0y_b) * (px_g - v0x_b)
    d1b = (v2x_b - v1x_b) * (py_g - v1y_b) - (v2y_b - v1y_b) * (px_g - v1x_b)
    d2b = (v0x_b - v2x_b) * (py_g - v2y_b) - (v0y_b - v2y_b) * (px_g - v2x_b)
    inside_best = (((d0b >= 0) & (d1b >= 0) & (d2b >= 0)) |
                   ((d0b <= 0) & (d1b <= 0) & (d2b <= 0)))

    dist_signed = jnp.where(inside_best, -d_sq, d_sq)
    improb = jax.nn.sigmoid(-dist_signed * delta)
    improb = jnp.where(any_coverage, improb, 0.0)
    improb_bxhxwx1 = improb[..., None]                              # (B, H, W, 1)

    return imfeat, improb_bxhxwx1


# Alias matching the original API
linear_rasterizer = linear_rasterizer
