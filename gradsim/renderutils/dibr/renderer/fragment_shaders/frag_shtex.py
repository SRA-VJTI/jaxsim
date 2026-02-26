# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Licensed under the MIT License.
#
# Pure JAX replacement — no torch dependency.

from __future__ import division, print_function

import jax.numpy as jnp

from .interpolation import texinterpolation


def fragmentshader(imnormal1_bxhxwx3, lightparam_bx9, imtexcoord_bxhxwx2,
                   texture_bx3xthxtw, improb_bxhxwx1):
    # Spherical harmonic basis evaluated at surface normals
    x = imnormal1_bxhxwx3[..., 0:1]
    y = imnormal1_bxhxwx3[..., 1:2]
    z = imnormal1_bxhxwx3[..., 2:3]

    band0  =  0.2820948 * jnp.ones_like(x)
    band10 = -0.3257350 * y
    band11 =  0.3257350 * z
    band12 = -0.3257350 * x
    band20 =  0.2731371 * (x * y)
    band21 = -0.2731371 * (y * z)
    band22 =  0.1365686 * (z * z) - 0.0788479
    band23 = -0.1931371 * (x * z)
    band24 =  0.1365686 * (x * x - y * y)

    bands = jnp.concatenate(
        [band0, band10, band11, band12, band20, band21, band22, band23, band24],
        axis=-1
    )                                                     # (B, H, W, 9)
    coef = jnp.sum(bands * lightparam_bx9[:, None, None, :], axis=-1, keepdims=True)

    texcolor_bxhxwx3 = texinterpolation(imtexcoord_bxhxwx2, texture_bx3xthxtw)
    color = coef * texcolor_bxhxwx3 * improb_bxhxwx1
    return jnp.clip(color, 0.0, 1.0)
