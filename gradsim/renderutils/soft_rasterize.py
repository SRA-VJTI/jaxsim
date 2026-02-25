# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure JAX implementation of the SoftRasterizer (Liu et al., ICCV 2019).

Replaces the CUDA-only torch backend with differentiable JAX operations.
All gradients flow through automatically via JAX autodiff.
"""

import jax
import jax.numpy as jnp


# ── texture sampling helpers ──────────────────────────────────────────────────

def _sample_texture(textures, bary0, bary1, bary2, texture_type):
    """Sample face texture at barycentric coordinates.

    Args:
        textures: one of
            (B, F, 3)           – constant per-face RGB
            (B, F, 3, 3)        – per-vertex RGB (vertex type)
            (B, F, K, K, K, 3) – volumetric barycentric grid (surface type)
        bary0, bary1, bary2: (B, F, H, W) – clamped barycentric coords (sum=1)
        texture_type: "vertex" or "surface"

    Returns:
        colors: (B, F, H, W, 3)
    """
    ndim = textures.ndim

    if ndim == 3:
        # (B, F, 3): broadcast as constant per face
        return textures[:, :, None, None, :]                  # (B, F, 1, 1, 3)

    if ndim == 4 or (ndim >= 4 and texture_type == "vertex"):
        # (B, F, 3, C): per-vertex attribute, interpolate with barycentric
        # bary0/1/2: (B, F, H, W) — weights for vertices 0/1/2
        c0 = textures[:, :, 0, :][:, :, None, None, :]      # (B, F, 1, 1, C)
        c1 = textures[:, :, 1, :][:, :, None, None, :]
        c2 = textures[:, :, 2, :][:, :, None, None, :]
        return (bary0[..., None] * c0 +
                bary1[..., None] * c1 +
                bary2[..., None] * c2)                        # (B, F, H, W, C)

    if ndim == 6:
        # (B, F, K, K, K, 3): volumetric barycentric texture, trilinear interp
        B, F = textures.shape[:2]
        K   = textures.shape[2]
        H, W = bary0.shape[2], bary0.shape[3]

        # Scale barycentric coords to grid indices in [0, K-1]
        coords0 = jnp.clip(bary0 * (K - 1), 0.0, K - 1 - 1e-6)  # (B,F,H,W)
        coords1 = jnp.clip(bary1 * (K - 1), 0.0, K - 1 - 1e-6)
        coords2 = jnp.clip(bary2 * (K - 1), 0.0, K - 1 - 1e-6)

        i0 = jnp.floor(coords0).astype(jnp.int32)
        j0 = jnp.floor(coords1).astype(jnp.int32)
        k0 = jnp.floor(coords2).astype(jnp.int32)
        i1 = jnp.minimum(i0 + 1, K - 1)
        j1 = jnp.minimum(j0 + 1, K - 1)
        k1 = jnp.minimum(k0 + 1, K - 1)

        di = (coords0 - i0.astype(jnp.float32))[..., None]   # (B,F,H,W,1)
        dj = (coords1 - j0.astype(jnp.float32))[..., None]
        dk = (coords2 - k0.astype(jnp.float32))[..., None]

        # Batch index arrays for gather
        b_idx = jnp.arange(B)[:, None, None, None]
        f_idx = jnp.arange(F)[None, :, None, None]

        c000 = textures[b_idx, f_idx, i0, j0, k0]
        c001 = textures[b_idx, f_idx, i0, j0, k1]
        c010 = textures[b_idx, f_idx, i0, j1, k0]
        c011 = textures[b_idx, f_idx, i0, j1, k1]
        c100 = textures[b_idx, f_idx, i1, j0, k0]
        c101 = textures[b_idx, f_idx, i1, j0, k1]
        c110 = textures[b_idx, f_idx, i1, j1, k0]
        c111 = textures[b_idx, f_idx, i1, j1, k1]

        return (c000 * (1-di)*(1-dj)*(1-dk) +
                c001 * (1-di)*(1-dj)*dk     +
                c010 * (1-di)*dj*(1-dk)     +
                c011 * (1-di)*dj*dk         +
                c100 * di*(1-dj)*(1-dk)     +
                c101 * di*(1-dj)*dk         +
                c110 * di*dj*(1-dk)         +
                c111 * di*dj*dk)

    # Fallback: treat first 3 values as constant color
    flat = textures.reshape(textures.shape[0], textures.shape[1], -1)
    return flat[:, :, :3][:, :, None, None, :]


# ── main rasterizer ───────────────────────────────────────────────────────────

def soft_rasterize(
    face_vertices,
    textures,
    image_size=256,
    background_color=None,
    near=1.0,
    far=100.0,
    fill_back=True,
    eps=1e-3,
    sigma_val=1e-5,
    dist_func="euclidean",
    dist_eps=1e-4,
    gamma_val=1e-4,
    aggr_func_rgb="softmax",
    aggr_func_alpha="prod",
    texture_type="surface",
):
    r"""Pure JAX soft rasterizer (SoftRas, Liu et al. ICCV 2019).

    Implements the full forward pass with differentiable sigmoid-based soft
    occupancy, depth-weighted softmax colour aggregation, and product alpha.

    Args:
        face_vertices: (B, F, 3, 3) per-face vertices. Each row holds
            ``[x, y, z]`` after camera projection. ``x, y ∈ [-1, 1]`` NDC;
            ``z > 0`` is positive depth from the camera.
            Also accepts the legacy flat format (B, F, 9).
        textures: face texture array — shape depends on ``texture_type``:
            * ``"surface"``: (B, F, K, K, K, 3) volumetric barycentric grid.
            * ``"vertex"``:  (B, F, 3, 3) per-vertex RGB.
            * Constant:      (B, F, 3) per-face RGB.
        image_size (int): Square output resolution H = W (default: 256).
        background_color (list of 3 floats): RGB background (default: black).
        near (float): Near clipping depth (default: 1).
        far (float): Far clipping depth (default: 100).
        fill_back (bool): Whether back-facing triangles contribute. The inside
            test implemented here already handles both winding orders, so this
            flag is effectively always True (kept for API compatibility).
        eps (float): Rasterizer epsilon (kept for API compatibility).
        sigma_val (float): Sharpness of soft occupancy sigmoid (default: 1e-5).
        dist_func (str): Distance metric — only ``"euclidean"`` is implemented.
        dist_eps (float): Distance epsilon (kept for API compatibility).
        gamma_val (float): Sharpness of depth-weighted softmax (default: 1e-4).
        aggr_func_rgb (str): RGB aggregation — only ``"softmax"`` implemented.
        aggr_func_alpha (str): Alpha aggregation — only ``"prod"`` implemented.
        texture_type (str): ``"surface"`` or ``"vertex"`` (default: ``"surface"``).

    Returns:
        soft_colors (jnp.ndarray): (B, 4, H, W) RGBA image, float32.
            Channels 0-2 are RGB; channel 3 is alpha (opacity).
    """
    if background_color is None:
        background_color = [0.0, 0.0, 0.0]

    bg = jnp.array(background_color, dtype=jnp.float32)  # (3,)

    # Accept legacy flat format (B, F, 9)
    if face_vertices.ndim == 3:
        face_vertices = face_vertices.reshape(face_vertices.shape[0],
                                              face_vertices.shape[1], 3, 3)

    B, F = face_vertices.shape[:2]
    H = W = image_size

    # ── pixel grid in NDC (pixel centres, top→bottom) ──────────────────────
    xs = jnp.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W)    # (W,)
    ys = jnp.linspace( 1.0 - 1.0 / H, -1.0 + 1.0 / H, H)   # (H,) top→bottom
    px, py = jnp.meshgrid(xs, ys)                            # (H, W)
    px = px[None, None]   # (1, 1, H, W) — broadcasts over (B, F)
    py = py[None, None]

    # ── per-face vertex 2D coords: (B, F, 1, 1) ───────────────────────────
    v0x = face_vertices[:, :, 0, 0, None, None]
    v0y = face_vertices[:, :, 0, 1, None, None]
    v1x = face_vertices[:, :, 1, 0, None, None]
    v1y = face_vertices[:, :, 1, 1, None, None]
    v2x = face_vertices[:, :, 2, 0, None, None]
    v2y = face_vertices[:, :, 2, 1, None, None]

    # ── triangle edges: (B, F, 1, 1) ──────────────────────────────────────
    e01x = v1x - v0x;  e01y = v1y - v0y
    e12x = v2x - v1x;  e12y = v2y - v1y
    e20x = v0x - v2x;  e20y = v0y - v2y
    # v0→v2 used for barycentric denominator
    v0v2x = v2x - v0x; v0v2y = v2y - v0y

    # ── pixel offset from v0: broadcasts to (B, F, H, W) ──────────────────
    p0x = px - v0x   # (B, F, H, W)
    p0y = py - v0y

    # ── inside / outside test (handles both winding orders) ───────────────
    d0 = e01x * (py - v0y) - e01y * (px - v0x)   # (B, F, H, W)
    d1 = e12x * (py - v1y) - e12y * (px - v1x)
    d2 = e20x * (py - v2y) - e20y * (px - v2x)
    inside = ((d0 >= 0) & (d1 >= 0) & (d2 >= 0)) | \
             ((d0 <= 0) & (d1 <= 0) & (d2 <= 0))       # (B, F, H, W)

    # ── squared distance to nearest edge ──────────────────────────────────
    def _seg_sq(px_, py_, ax, ay, bx, by):
        """Squared distance from pixel to segment [a, b]."""
        abx, aby = bx - ax, by - ay
        apx, apy = px_ - ax, py_ - ay
        t = (apx * abx + apy * aby) / (abx * abx + aby * aby + 1e-20)
        t = jnp.clip(t, 0.0, 1.0)
        cx, cy = ax + t * abx, ay + t * aby
        return (px_ - cx) ** 2 + (py_ - cy) ** 2

    d01 = _seg_sq(px, py, v0x, v0y, v1x, v1y)    # (B, F, H, W)
    d12 = _seg_sq(px, py, v1x, v1y, v2x, v2y)
    d20 = _seg_sq(px, py, v2x, v2y, v0x, v0y)
    d_sq = jnp.minimum(jnp.minimum(d01, d12), d20)

    # Signed: negative inside (→ high sigma), positive outside (→ low sigma)
    dist_signed = jnp.where(inside, -d_sq, d_sq)           # (B, F, H, W)

    # ── soft occupancy ─────────────────────────────────────────────────────
    sigma = jax.nn.sigmoid(-dist_signed / sigma_val)        # (B, F, H, W)

    # ── depth for aggregation weight ───────────────────────────────────────
    # Centroid depth; smaller z = closer to camera = higher weight
    z_face = jnp.mean(face_vertices[:, :, :, 2], axis=2)   # (B, F)
    z_face = jnp.clip(z_face, near, far)
    z_b = z_face[:, :, None, None]                          # (B, F, 1, 1)

    # ── barycentric coordinates at each pixel ──────────────────────────────
    denom  = e01x * v0v2y - e01y * v0v2x + 1e-20           # (B, F, 1, 1)
    bary1  = (p0x * v0v2y - p0y * v0v2x) / denom           # (B, F, H, W)
    bary2  = (e01x * p0y  - e01y * p0x ) / denom
    bary0  = 1.0 - bary1 - bary2
    # Clamp + renormalise for pixels outside the triangle
    bary0  = jnp.clip(bary0, 0.0, 1.0)
    bary1  = jnp.clip(bary1, 0.0, 1.0)
    bary2  = jnp.clip(bary2, 0.0, 1.0)
    bary_s = bary0 + bary1 + bary2 + 1e-20
    bary0 /= bary_s;  bary1 /= bary_s;  bary2 /= bary_s

    # ── texture sampling ───────────────────────────────────────────────────
    face_colors = _sample_texture(textures, bary0, bary1, bary2, texture_type)
    # face_colors: (B, F, H, W, 3)  [or broadcasts correctly]
    # Ensure shape (B, F, H, W, 3) via expand if needed
    face_colors = jnp.broadcast_to(face_colors,
                                   (B, F, H, W, face_colors.shape[-1]))

    # ── RGB aggregation: depth-weighted softmax ────────────────────────────
    # w_f ∝ sigma_f * exp(-z_f / gamma)  (closer face → larger weight)
    log_w    = jnp.log(sigma + 1e-20) - z_b / gamma_val    # (B, F, H, W)
    log_w_mx = jnp.max(log_w, axis=1, keepdims=True)
    w        = jnp.exp(log_w - log_w_mx)
    w_norm   = w / (jnp.sum(w, axis=1, keepdims=True) + 1e-20)  # (B, F, H, W)
    rgb = jnp.sum(w_norm[..., None] * face_colors, axis=1)      # (B, H, W, 3)

    # ── alpha aggregation: product ─────────────────────────────────────────
    # alpha = 1 - prod_f(1 - sigma_f); use log-sum for stability
    log1ms = jnp.log(jnp.maximum(1.0 - sigma, 1e-20))           # (B, F, H, W)
    alpha  = 1.0 - jnp.exp(jnp.sum(log1ms, axis=1))            # (B, H, W)
    alpha  = jnp.clip(alpha, 0.0, 1.0)

    # ── composite over background ──────────────────────────────────────────
    a    = alpha[..., None]                                       # (B, H, W, 1)
    comp = rgb * a + bg[None, None, None, :] * (1.0 - a)        # (B, H, W, 3)
    rgba = jnp.concatenate([comp, a], axis=-1)                   # (B, H, W, 4)
    out  = jnp.transpose(rgba, (0, 3, 1, 2))                    # (B, 4, H, W)
    return out
