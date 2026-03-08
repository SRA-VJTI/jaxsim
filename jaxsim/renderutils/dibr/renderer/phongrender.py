# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Licensed under the MIT License.
#
# Pure JAX replacement — no torch / nn.Module dependency.

from __future__ import division, print_function

import numpy as np
import jax.numpy as jnp

from ..rasterizer import linear_rasterizer
from ..utils import datanormalize
from .fragment_shaders.frag_phongtex import fragmentshader
from .vertex_shaders.perpsective import perspective_projection


class PhongRender:
    """Phong texture renderer (differentiable, pure JAX)."""

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.smooth = False

    def set_smooth(self, pfmtx):
        self.smooth = True
        self.pfmtx = jnp.array(pfmtx)[None]   # (1, P, F)

    def __call__(self, points, cameras, uv_bxpx2, texture_bx3xthxtw,
                 lightdirect_bx3, material_bx3x3, shininess_bx1, ft_fx3=None):
        assert lightdirect_bx3 is not None, "Phong requires lightdirect_bx3"
        assert material_bx3x3 is not None, "Phong requires material_bx3x3"
        assert shininess_bx1 is not None, "Phong requires shininess_bx1"

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
            pfmtx = jnp.broadcast_to(self.pfmtx, (bnum,) + self.pfmtx.shape[1:])
            normal_bxpx3 = jnp.matmul(pfmtx, normal_bxfx3)
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
        eyedirect_bxfx3x3 = (-points3d_bxfx9).reshape(bnum, fnum, 3, 3)

        feat = jnp.concatenate([normal_bxfx3x3, eyedirect_bxfx3x3, uv_bxfx3x3], axis=3)
        feat = feat.reshape(bnum, fnum, -1)

        imfeature, improb_bxhxwx1 = linear_rasterizer(
            self.height, self.width,
            points3d_bxfx9, points2d_bxfx6, normalz_bxfx1, feat,
        )

        imnormal = imfeature[:, :, :, :3]
        imeye = imfeature[:, :, :, 3:6]
        imtexcoords = imfeature[:, :, :, 6:8]
        immask = imfeature[:, :, :, 8:9]

        imnormal1 = datanormalize(imnormal, axis=3)
        lightdirect_bx3 = datanormalize(lightdirect_bx3, axis=1)
        imeye1 = datanormalize(imeye, axis=3)

        imrender = fragmentshader(
            imnormal1, lightdirect_bx3, imeye1,
            material_bx3x3, shininess_bx1, imtexcoords, texture_bx3xthxtw, immask,
        )
        return imrender, improb_bxhxwx1, normal1_bxfx3
