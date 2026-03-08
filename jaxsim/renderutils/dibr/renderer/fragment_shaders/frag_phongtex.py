# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Licensed under the MIT License.
#
# Pure JAX replacement — no torch dependency.

from __future__ import division, print_function

import jax.numpy as jnp

from .interpolation import texinterpolation


def fragmentshader(imnormal1_bxhxwx3, lightdirect1_bx3, eyedirect1_bxhxwx3,
                   material_bx3x3, shininess_bx1, imtexcoord_bxhxwx2,
                   texture_bx3xthxtw, improb_bxhxwx1):
    # Phong shading: ambient + Lambertian diffuse + specular
    lightdirect1_bx1x1x3 = lightdirect1_bx3[:, None, None, :]      # (B,1,1,3)

    # Lambertian term
    cosTheta = jnp.sum(imnormal1_bxhxwx3 * lightdirect1_bx1x1x3, axis=-1, keepdims=True)
    cosTheta = jnp.clip(cosTheta, 0.0, 1.0)

    # Specular (Phong reflection)
    reflect = -lightdirect1_bx1x1x3 + 2.0 * cosTheta * imnormal1_bxhxwx3
    cosAlpha = jnp.sum(reflect * eyedirect1_bxhxwx3, axis=-1, keepdims=True)
    cosAlpha = jnp.clip(cosAlpha, 1e-5, 1.0)
    cosAlpha = jnp.power(cosAlpha, shininess_bx1[:, None, None, :])

    # Material colours: row 0=ambient, row 1=diffuse, row 2=specular
    MatAmb = material_bx3x3[:, 0:1, :][:, None, :, :]              # (B,1,1,3)
    MatDif = material_bx3x3[:, 1:2, :][:, None, :, :] * cosTheta   # (B,H,W,3)
    MatSpe = material_bx3x3[:, 2:3, :][:, None, :, :] * cosAlpha

    texcolor = texinterpolation(imtexcoord_bxhxwx2, texture_bx3xthxtw)
    color = (MatAmb + MatDif) * texcolor + MatSpe
    color = color * improb_bxhxwx1
    return jnp.clip(color, 0.0, 1.0)
