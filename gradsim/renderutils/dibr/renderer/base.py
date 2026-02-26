# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Licensed under the MIT License.
#
# Pure JAX replacement — no torch / nn.Module / CUDA dependency.

from __future__ import division, print_function

import numpy as np
import jax.numpy as jnp

from ..utils import compute_camera_params, perspectiveprojectionnp
from .phongrender import PhongRender
from .shrender import SHRender
from .texrender import TexRender as Lambertian
from .vcrender import VCRender

renderers = {
    "VertexColor": VCRender,
    "Lambertian": Lambertian,
    "SphericalHarmonics": SHRender,
    "Phong": PhongRender,
}


class Renderer:
    """High-level differentiable renderer (pure JAX, no CUDA)."""

    def __init__(self, height, width, mode="VertexColor",
                 camera_center=None, camera_up=None, camera_fov_y=None):
        assert mode in renderers, \
            "mode {0} must be one of: {1}".format(mode, list(renderers))
        self.mode = mode
        self.renderer = renderers[mode](height, width)
        self.camera_center = (np.array([0, 0, 0], dtype=np.float32)
                              if camera_center is None else camera_center)
        self.camera_up = (np.array([0, 1, 0], dtype=np.float32)
                          if camera_up is None else camera_up)
        self.camera_fov_y = (49.13434207744484 * np.pi / 180.0
                             if camera_fov_y is None else camera_fov_y)
        self.camera_params = None

    def __call__(self, points, *args, **kwargs):
        if self.camera_params is None:
            print("Camera parameters not set — using defaults: "
                  "distance=1, elevation=30, azimuth=0")
            self.set_look_at_parameters([0], [30], [1])

        assert self.camera_params[0].shape[0] == points[0].shape[0], \
            "Camera batch size must equal points batch size"

        return self.renderer(points, self.camera_params, *args, **kwargs)

    def set_look_at_parameters(self, azimuth, elevation, distance):
        proj = perspectiveprojectionnp(self.camera_fov_y, 1.0)          # (3, 1)
        camera_projection_mtx = jnp.array(proj, dtype=jnp.float32)     # (3, 1)

        mats, shifts = [], []
        for a, e, d in zip(azimuth, elevation, distance):
            mat, pos = compute_camera_params(a, e, d)
            mats.append(mat)
            shifts.append(pos)

        camera_view_mtx = jnp.stack(mats)       # (B, 3, 3)
        camera_view_shift = jnp.stack(shifts)   # (B, 3)

        self.camera_params = [camera_view_mtx, camera_view_shift,
                              camera_projection_mtx]

    def set_camera_parameters(self, parameters):
        self.camera_params = parameters
