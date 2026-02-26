# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Licensed under the MIT License.
#
# Pure JAX replacement — no torch / nn.Module dependency.

from __future__ import division, print_function

import jax.numpy as jnp

from ..rasterizer import linear_rasterizer
from ..utils import datanormalize
from .vertex_shaders.perpsective import perspective_projection


class VCRender:
    """Vertex-colour renderer (differentiable, pure JAX)."""

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, points, cameras, colors_bxpx3):
        points_bxpx3, faces_fx3 = points

        points3d_bxfx9, points2d_bxfx6, normal_bxfx3 = perspective_projection(
            points_bxpx3, faces_fx3, cameras
        )

        normalz_bxfx1 = normal_bxfx3[:, :, 2:3]
        normal1_bxfx3 = datanormalize(normal_bxfx3, axis=2)

        c0 = colors_bxpx3[:, faces_fx3[:, 0], :]
        c1 = colors_bxpx3[:, faces_fx3[:, 1], :]
        c2 = colors_bxpx3[:, faces_fx3[:, 2], :]
        mask = jnp.ones_like(c0[:, :, :1])
        color_bxfx12 = jnp.concatenate([c0, mask, c1, mask, c2, mask], axis=2)

        imfeat, improb_bxhxwx1 = linear_rasterizer(
            self.height, self.width,
            points3d_bxfx9, points2d_bxfx6, normalz_bxfx1, color_bxfx12,
        )

        imrender = imfeat[:, :, :, :3]
        return imrender, improb_bxhxwx1, normal1_bxfx3
