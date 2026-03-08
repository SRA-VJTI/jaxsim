# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Licensed under the MIT License.
#
# Pure JAX replacement — no torch dependency.

from __future__ import division, print_function

import jax.numpy as jnp

from .interpolation import texinterpolation


def fragmentshader(imtexcoord_bxhxwx2, texture_bx3xthxtw, improb_bxhxwx1,
                   filtering="nearest"):
    texcolor_bxhxwx3 = texinterpolation(
        imtexcoord_bxhxwx2, texture_bx3xthxtw, filtering=filtering
    )
    color = texcolor_bxhxwx3 * improb_bxhxwx1
    return jnp.clip(color, 0.0, 1.0)
