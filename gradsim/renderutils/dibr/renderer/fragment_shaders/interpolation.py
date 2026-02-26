# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Licensed under the MIT License.
#
# Pure JAX replacement for torch.nn.functional.grid_sample.

from __future__ import division, print_function

import jax.numpy as jnp


def texinterpolation(imtexcoord_bxhxwx2, texture_bx3xthxtw, filtering="nearest"):
    """Sample a texture image at UV coordinates.

    Replicates the OpenGL→PyTorch coordinate conversion from the original:
    - UV coords are wrapped to [0, 1] (tiling)
    - v-axis is flipped (UV v=0 → texture bottom row, v=1 → top row)

    Args:
        imtexcoord_bxhxwx2: (B, H, W, 2) UV coords in OpenGL convention
            (u horizontal, v vertical with 0 at bottom).
        texture_bx3xthxtw: (B, C, TH, TW) texture image (C channels).
        filtering: ``"nearest"`` or ``"bilinear"``.

    Returns:
        (B, H, W, C) sampled colours.
    """
    B, C, TH, TW = texture_bx3xthxtw.shape

    # Wrap UV to [0, 1) (tile)
    coords = jnp.remainder(imtexcoord_bxhxwx2, 1.0)               # (B, H, W, 2)
    u = coords[..., 0]                                              # (B, H, W)
    v = 1.0 - coords[..., 1]                                       # flip v: 0=top

    # (B, TH, TW, C) — easier indexing
    tex_bhwc = jnp.transpose(texture_bx3xthxtw, (0, 2, 3, 1))

    b_idx = jnp.arange(B)[:, None, None]                           # (B, 1, 1)

    if filtering == "nearest":
        col = jnp.clip((u * TW).astype(jnp.int32), 0, TW - 1)
        row = jnp.clip((v * TH).astype(jnp.int32), 0, TH - 1)
        return tex_bhwc[b_idx, row, col]                            # (B, H, W, C)

    # bilinear
    col_f = u * TW - 0.5
    row_f = v * TH - 0.5
    col0 = jnp.clip(jnp.floor(col_f).astype(jnp.int32), 0, TW - 1)
    col1 = jnp.clip(col0 + 1, 0, TW - 1)
    row0 = jnp.clip(jnp.floor(row_f).astype(jnp.int32), 0, TH - 1)
    row1 = jnp.clip(row0 + 1, 0, TH - 1)
    dc = jnp.clip(col_f - col0.astype(jnp.float32), 0.0, 1.0)[..., None]
    dr = jnp.clip(row_f - row0.astype(jnp.float32), 0.0, 1.0)[..., None]
    c00 = tex_bhwc[b_idx, row0, col0]
    c01 = tex_bhwc[b_idx, row0, col1]
    c10 = tex_bhwc[b_idx, row1, col0]
    c11 = tex_bhwc[b_idx, row1, col1]
    return (c00 * (1 - dc) * (1 - dr) +
            c01 * dc       * (1 - dr) +
            c10 * (1 - dc) * dr       +
            c11 * dc       * dr)
