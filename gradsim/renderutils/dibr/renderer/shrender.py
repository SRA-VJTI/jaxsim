# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Licensed under the MIT License.
#
# Pure JAX replacement — no torch / nn.Module dependency.

from __future__ import division, print_function

import jax.numpy as jnp

from ..rasterizer import linear_rasterizer
from ..utils import datanormalize
from .fragment_shaders.frag_shtex import fragmentshader
from .vertex_shaders.perpsective import perspective_projection


class SHRender:
    """Spherical-harmonics texture renderer (differentiable, pure JAX)."""

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.smooth = False

    def set_smooth(self, pfmtx):
        self.smooth = True
        self.pfmtx = pfmtx  # (P, F) numpy/jnp matrix

    def __call__(self, points, cameras, uv_bxpx2, texture_bx3xthxtw,
                 lightparam, ft_fx3=None):
        assert lightparam is not None, \
            "When using SH model, lightparam (B, 9) must be passed"

        points_bxpx3, faces_fx3 = points
        if ft_fx3 is None:
            ft_fx3 = faces_fx3

        points3d_bxfx9, points2d_bxfx6, normal_bxfx3 = perspective_projection(
            points_bxpx3, faces_fx3, cameras
        )

        normalz_bxfx1 = normal_bxfx3[:, :, 2:3]
        normal1_bxfx3 = datanormalize(normal_bxfx3, axis=2)

        bnum, fnum = normal1_bxfx3.shape[:2]

        if self.smooth:
            normal_bxpx3 = jnp.matmul(self.pfmtx, normal_bxfx3)
            n0 = normal_bxpx3[:, faces_fx3[:, 0], :]
            n1 = normal_bxpx3[:, faces_fx3[:, 1], :]
            n2 = normal_bxpx3[:, faces_fx3[:, 2], :]
            normal_bxfx9 = jnp.concatenate([n0, n1, n2], axis=2)
        else:
            normal_bxfx9 = jnp.tile(normal_bxfx3, (1, 1, 3))

        c0 = uv_bxpx2[:, ft_fx3[:, 0], :]
        c1 = uv_bxpx2[:, ft_fx3[:, 1], :]
        c2 = uv_bxpx2[:, ft_fx3[:, 2], :]
        mask = jnp.ones_like(c0[:, :, :1])
        uv_bxfx3x3 = jnp.concatenate([c0, mask, c1, mask, c2, mask], axis=2)
        uv_bxfx3x3 = uv_bxfx3x3.reshape(bnum, fnum, 3, -1)

        normal_bxfx3x3 = normal_bxfx9.reshape(bnum, fnum, 3, -1)
        feat = jnp.concatenate([normal_bxfx3x3, uv_bxfx3x3], axis=3)
        feat = feat.reshape(bnum, fnum, -1)

        imfeat, improb_bxhxwx1 = linear_rasterizer(
            self.height, self.width,
            points3d_bxfx9, points2d_bxfx6, normalz_bxfx1, feat,
        )

        imnormal_bxhxwx3 = imfeat[:, :, :, :3]
        imtexcoords = imfeat[:, :, :, 3:5]
        hardmask = imfeat[:, :, :, 5:]

        imnormal1_bxhxwx3 = datanormalize(imnormal_bxhxwx3, axis=3)
        imrender = fragmentshader(
            imnormal1_bxhxwx3, lightparam, imtexcoords, texture_bx3xthxtw, hardmask
        )
        return imrender, improb_bxhxwx1, normal1_bxfx3
