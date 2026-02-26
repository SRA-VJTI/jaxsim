# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Licensed under the MIT License.
#
# Pure JAX replacement — no torch / nn.Module dependency.

from __future__ import division, print_function

import jax.numpy as jnp

from ..rasterizer import linear_rasterizer
from ..utils import datanormalize
from .fragment_shaders.frag_tex import fragmentshader
from .vertex_shaders.perpsective import perspective_projection


class TexRender:
    """Lambertian texture renderer (differentiable, pure JAX)."""

    def __init__(self, height, width, filtering="nearest"):
        self.height = height
        self.width = width
        self.filtering = filtering

    def __call__(self, points, cameras, uv_bxpx2, texture_bx3xthxtw, ft_fx3=None):
        points_bxpx3, faces_fx3 = points
        if ft_fx3 is None:
            ft_fx3 = faces_fx3

        points3d_bxfx9, points2d_bxfx6, normal_bxfx3 = perspective_projection(
            points_bxpx3, faces_fx3, cameras
        )

        normalz_bxfx1 = normal_bxfx3[:, :, 2:3]
        normal1_bxfx3 = datanormalize(normal_bxfx3, axis=2)

        c0 = uv_bxpx2[:, ft_fx3[:, 0], :]
        c1 = uv_bxpx2[:, ft_fx3[:, 1], :]
        c2 = uv_bxpx2[:, ft_fx3[:, 2], :]
        mask = jnp.ones_like(c0[:, :, :1])
        uv_bxfx9 = jnp.concatenate([c0, mask, c1, mask, c2, mask], axis=2)

        imfeat, improb_bxhxwx1 = linear_rasterizer(
            self.height, self.width,
            points3d_bxfx9, points2d_bxfx6, normalz_bxfx1, uv_bxfx9,
        )

        imtexcoords = imfeat[:, :, :, :2]
        hardmask = imfeat[:, :, :, 2:3]

        imrender = fragmentshader(
            imtexcoords, texture_bx3xthxtw, hardmask, filtering=self.filtering
        )
        return imrender, improb_bxhxwx1, normal1_bxfx3
