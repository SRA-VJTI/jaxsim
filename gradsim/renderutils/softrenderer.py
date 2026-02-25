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

"""JAX port of the SoftRenderer class (Liu et al., ICCV 2019).

Replaces all torch / CUDA calls with pure jax.numpy equivalents.
No GPU-specific API is required; JAX handles device placement automatically.
"""

import math
from typing import Optional, Union

import jax.numpy as jnp

from .lighting import compute_ambient_light, compute_directional_light
from .soft_rasterize import soft_rasterize


def _normalize(v, eps=1e-5):
    """L2-normalise vectors along the last axis."""
    n = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(n, eps)


def _avg_pool2d(x, kernel_size=2):
    """Average pooling with stride = kernel_size (no padding).

    Args:
        x: (B, C, H, W)
        kernel_size (int): pooling window (and stride).

    Returns:
        (B, C, H // kernel_size, W // kernel_size)
    """
    B, C, H, W = x.shape
    k = kernel_size
    return x.reshape(B, C, H // k, k, W // k, k).mean(axis=(3, 5))


class SoftRenderer:
    r"""A class implementing the \emph{Soft Renderer}
        from the following ICCV 2019 paper:
            Soft Rasterizer: A differentiable renderer for image-based 3D reasoning
            Shichen Liu, Tianye Li, Weikai Chen, and Hao Li
            Link: https://arxiv.org/abs/1904.01786

    .. note::
        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::
            @article{liu2019softras,
              title={Soft Rasterizer: A Differentiable Renderer for Image-based 3D Reasoning},
              author={Liu, Shichen and Li, Tianye and Chen, Weikai and Li, Hao},
              journal={The IEEE International Conference on Computer Vision (ICCV)},
              month = {Oct},
              year={2019}
            }

    """

    def __init__(
        self,
        image_size: int = 256,
        anti_aliasing: bool = True,
        bg_color=None,
        fill_back: bool = True,
        camera_mode: str = "look_at",
        K=None,
        rmat=None,
        tvec=None,
        perspective_distort: bool = True,
        sigma_val: float = 1e-5,
        dist_func: str = "euclidean",
        dist_eps: float = 1e-4,
        gamma_val: float = 1e-4,
        aggr_func_rgb: str = "softmax",
        aggr_func_alpha: str = "prod",
        texture_type: str = "surface",
        viewing_angle: float = 30.0,
        viewing_scale: float = 1.0,
        eye=None,
        camera_direction=None,
        near: float = 1.0,
        far: float = 100.0,
        light_mode: str = "surface",
        light_intensity_ambient: float = 0.5,
        light_intensity_directional: float = 0.5,
        light_color_ambient=None,
        light_color_directional=None,
        light_direction=None,
    ):
        r"""Initialise the SoftRenderer.

        Args:
            image_size (int): Square output image resolution (default: 256).
            anti_aliasing (bool): Render at 2× and downsample (default: True).
            bg_color: (3,) RGB background colour (default: black zeros).
            fill_back (bool): Rasterise back faces (default: True).
            camera_mode (str): ``"look_at"``, ``"look"``, or ``"projection"``.
            K: (4, 4) or (B, 4, 4) camera intrinsics (``"projection"`` mode).
            rmat: (4, 4) or (B, 4, 4) rotation matrix (``"projection"`` mode).
            tvec: (3,) or (B, 3) translation vector (``"projection"`` mode).
            perspective_distort (bool): Apply perspective divide (default: True).
            sigma_val (float): Soft-occupancy sharpness (default: 1e-5).
            dist_func (str): Distance function; only ``"euclidean"`` supported.
            dist_eps (float): Distance epsilon (API compat, unused).
            gamma_val (float): Softmax sharpness (default: 1e-4).
            aggr_func_rgb (str): RGB aggregation; only ``"softmax"`` supported.
            aggr_func_alpha (str): Alpha aggregation; only ``"prod"`` supported.
            texture_type (str): ``"surface"`` or ``"vertex"`` (default: ``"surface"``).
            viewing_angle (float): FOV half-angle in degrees (default: 30).
            viewing_scale (float): Scale for orthographic view (unused, kept for API).
            eye: (3,) camera position for look/look_at modes.
            camera_direction: (3,) look direction for look mode.
            near (float): Near clipping depth (default: 1).
            far (float): Far clipping depth (default: 100).
            light_intensity_ambient (float): Ambient intensity ∈ [0,1] (default: 0.5).
            light_intensity_directional (float): Directional intensity ∈ [0,1] (default: 0.5).
            light_color_ambient: (3,) ambient colour (default: white).
            light_color_directional: (3,) directional colour (default: white).
            light_direction: (3,) light direction (default: +Y).
        """
        self.image_size   = image_size
        self.anti_aliasing = anti_aliasing
        self.fill_back    = fill_back
        self.camera_mode  = camera_mode
        self.near         = near
        self.far          = far

        self.bg_color = (jnp.zeros(3, dtype=jnp.float32)
                         if bg_color is None else jnp.asarray(bg_color, dtype=jnp.float32))

        # Camera direction (for "look" mode)
        if camera_direction is None:
            self.camera_direction = jnp.array([0.0, 0.0, 1.0])
        else:
            self.camera_direction = jnp.asarray(camera_direction)

        if camera_mode == "projection":
            self.K    = jnp.eye(3)[None] if K    is None else jnp.asarray(K)
            self.rmat = jnp.eye(3)[None] if rmat is None else jnp.asarray(rmat)
            if tvec is None:
                t = jnp.zeros((1, 3))
                t = t.at[0, 2].set(-5.0)
                self.tvec = t
            else:
                self.tvec = jnp.asarray(tvec)

        elif camera_mode in ("look", "look_at"):
            self.perspective_distort = perspective_distort
            self.viewing_angle       = viewing_angle
            if eye is None:
                d = -(1.0 / math.tan(math.radians(viewing_angle)) + 1.0)
                self.eye = jnp.array([0.0, 0.0, d])
            else:
                self.eye = jnp.asarray(eye)
            self.camera_direction = jnp.array([0.0, 0.0, 1.0])

        # Soft-rasterizer parameters
        self.sigma_val       = sigma_val
        self.dist_func       = dist_func
        self.dist_eps        = dist_eps
        self.gamma_val       = gamma_val
        self.aggr_func_rgb   = aggr_func_rgb
        self.aggr_func_alpha = aggr_func_alpha
        self.texture_type    = texture_type
        self.rasterizer_eps  = 1e-3

        # Lighting
        self.light_intensity_ambient     = light_intensity_ambient
        self.light_intensity_directional = light_intensity_directional
        self.light_color_ambient = (jnp.ones(3)
                                    if light_color_ambient is None
                                    else jnp.asarray(light_color_ambient))
        self.light_color_directional = (jnp.ones(3)
                                        if light_color_directional is None
                                        else jnp.asarray(light_color_directional))
        self.light_direction = (jnp.array([0.0, 1.0, 0.0])
                                if light_direction is None
                                else jnp.asarray(light_direction))

    # ── public API ────────────────────────────────────────────────────────────

    def forward(self, vertices, faces, textures=None, mode=None,
                K=None, rmat=None, tvec=None):
        return self.render(vertices, faces, textures, mode, K, rmat, tvec)

    def render(self, vertices, faces, textures=None, mode=None,
               K=None, rmat=None, tvec=None):
        r"""Render RGB + alpha channels.

        Args:
            vertices: (B, V, 3) mesh vertices in world / object space.
            faces:    (B, F, 3) triangle face indices.
            textures: texture array (shape depends on ``texture_type``).
            mode (str): ``None`` (all), ``"rgb"``, ``"silhouette"``, ``"depth"``.
            K, rmat, tvec: per-call camera override for ``"projection"`` mode.

        Returns:
            (B, 4, H, W) RGBA image.
        """
        if mode not in ("depth", "silhouette"):
            textures = self.lighting(vertices, faces, textures)

        vertices = self.transform_to_camera_frame(vertices, K=K, rmat=rmat, tvec=tvec)
        vertices = self.project_to_image(vertices, K=K, rmat=rmat, tvec=tvec)
        images   = self.rasterize(vertices, faces, textures)
        return images

    # ── lighting ──────────────────────────────────────────────────────────────

    def lighting(self, vertices, faces, textures):
        r"""Apply ambient + directional lighting to textures.

        Args:
            vertices: (B, V, 3)
            faces:    (B, F, 3)
            textures: (B, F, 4, 4, 4, 3) or compatible shape.

        Returns:
            lit textures with same shape as input ``textures``.
        """
        faces_lighting = self.vertices_to_faces(vertices, faces)  # (B, F, 3, 3)
        ambient = compute_ambient_light(
            faces_lighting, textures,
            self.light_intensity_ambient,
            self.light_color_ambient,
        )
        directional = compute_directional_light(
            faces_lighting, textures,
            self.light_intensity_directional,
            self.light_color_directional,
            self.light_direction,
        )
        light = ambient + directional   # (B, F, 1, 3)
        # Expand light to broadcast with textures of any ndim
        # e.g. (B, F, 1, 3) → (B, F, 1, 1, 1, 3) for (B, F, K, K, K, 3) textures
        while light.ndim < textures.ndim:
            light = jnp.expand_dims(light, axis=-2)
        return light * textures

    def shading(self):
        raise NotImplementedError

    # ── camera transforms ─────────────────────────────────────────────────────

    def transform_to_camera_frame(self, vertices, K=None, rmat=None, tvec=None):
        r"""Transform vertices to the camera coordinate frame.

        Args:
            vertices: (B, V, 3)

        Returns:
            (B, V, 3) in camera space.
        """
        if self.camera_mode == "look_at":
            vertices = self.look_at(vertices, self.eye)
        elif self.camera_mode == "look":
            vertices = self.look(vertices, self.eye, self.camera_direction)
        elif self.camera_mode == "projection":
            _K    = self.K    if K    is None else jnp.asarray(K)
            _rmat = self.rmat if rmat is None else jnp.asarray(rmat)
            _tvec = self.tvec if tvec is None else jnp.asarray(tvec)
            vertices = self.perspective_projection(vertices, _K, _rmat, _tvec)
        return vertices

    def project_to_image(self, vertices, K=None, rmat=None, tvec=None):
        r"""Project vertices from camera space to image (NDC) space.

        Args:
            vertices: (B, V, 3) in camera space.

        Returns:
            (B, V, 3) with x, y in NDC and z as depth.
        """
        if self.camera_mode in ("look_at", "look"):
            vertices = self.perspective_distortion(vertices, angle=self.viewing_angle)
        elif self.camera_mode == "projection":
            pass  # already projected by transform_to_camera_frame
        return vertices

    # ── rasterization ─────────────────────────────────────────────────────────

    def rasterize(self, vertices, faces, textures):
        r"""Rasterise: vertices → pixel image.

        Args:
            vertices: (B, V, 3) projected vertices.
            faces:    (B, F, 3) face indices.
            textures: face texture array.

        Returns:
            (B, 4, H, W) RGBA.
        """
        face_vertices = self.vertices_to_faces(vertices, faces)   # (B, F, 3, 3)
        face_textures = textures
        if self.texture_type == "vertex":
            face_textures = self.textures_to_faces(textures, faces)

        size = self.image_size * (2 if self.anti_aliasing else 1)
        out  = soft_rasterize(
            face_vertices,
            face_textures,
            size,
            list(self.bg_color),
            self.near,
            self.far,
            self.fill_back,
            self.rasterizer_eps,
            self.sigma_val,
            self.dist_func,
            self.dist_eps,
            self.gamma_val,
            self.aggr_func_rgb,
            self.aggr_func_alpha,
            self.texture_type,
        )
        if self.anti_aliasing:
            out = _avg_pool2d(out, kernel_size=2)
        return out

    # ── camera methods ────────────────────────────────────────────────────────

    def look_at(self, vertices, eye, at=None, up=None):
        r"""Place camera at ``eye`` looking toward ``at``.

        Args:
            vertices: (B, V, 3)
            eye:      (3,) or (B, 3) camera position.
            at:       (3,) look-at target (default: origin).
            up:       (3,) up vector (default: +Y).

        Returns:
            (B, V, 3) vertices in camera space.
        """
        if at is None:
            at = jnp.zeros(3)
        if up is None:
            up = jnp.array([0.0, 1.0, 0.0])

        eye = jnp.asarray(eye, dtype=jnp.float32)
        at  = jnp.asarray(at,  dtype=jnp.float32)
        up  = jnp.asarray(up,  dtype=jnp.float32)

        B = vertices.shape[0]
        if eye.ndim == 1:
            eye = jnp.broadcast_to(eye[None], (B, 3))
        if at.ndim  == 1:
            at  = jnp.broadcast_to(at[None],  (B, 3))
        if up.ndim  == 1:
            up  = jnp.broadcast_to(up[None],  (B, 3))

        z_axis = _normalize(at - eye)                           # (B, 3)
        x_axis = _normalize(jnp.cross(up, z_axis))
        y_axis = _normalize(jnp.cross(z_axis, x_axis))

        # Rotation matrix R: (B, 3, 3), rows = [x_axis, y_axis, z_axis]
        R = jnp.concatenate(
            [x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]], axis=1
        )                                                       # (B, 3, 3)

        eye_v = eye[:, None, :]                                 # (B, 1, 3)
        vertices = vertices - eye_v                             # (B, V, 3)
        vertices = jnp.matmul(vertices, jnp.swapaxes(R, 1, 2))# (B, V, 3)
        return vertices

    def look(self, vertices, eye, direction=None, up=None):
        r"""Point camera in a given ``direction`` from ``eye``.

        Args:
            vertices:  (B, V, 3)
            eye:       (3,) or (B, 3) camera position.
            direction: (3,) viewing direction (default: +Z).
            up:        (3,) up vector (default: +Y).

        Returns:
            (B, V, 3) vertices in camera space.
        """
        if direction is None:
            direction = jnp.array([0.0, 0.0, 1.0])
        if up is None:
            up = jnp.array([0.0, 1.0, 0.0])

        eye       = jnp.asarray(eye, dtype=jnp.float32)
        direction = jnp.asarray(direction, dtype=jnp.float32)
        up        = jnp.asarray(up, dtype=jnp.float32)

        if eye.ndim       == 1: eye       = eye[None, :]
        if direction.ndim == 1: direction = direction[None, :]
        if up.ndim        == 1: up        = up[None, :]

        z_axis = _normalize(direction)
        x_axis = _normalize(jnp.cross(up, z_axis))
        y_axis = _normalize(jnp.cross(z_axis, x_axis))

        R = jnp.concatenate(
            [x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]], axis=1
        )

        eye_v = eye[:, None, :]
        vertices = vertices - eye_v
        vertices = jnp.matmul(vertices, jnp.swapaxes(R, 1, 2))
        return vertices

    def perspective_distortion(self, vertices, angle=30.0):
        r"""Perspective divide based on viewing angle.

        Args:
            vertices: (B, V, 3) in camera space (z > 0 = forward).
            angle:    Half-FOV in degrees (default: 30).

        Returns:
            (B, V, 3): x, y divided by (z * tan(angle)); z unchanged.
        """
        width = math.tan(math.radians(angle))
        z     = vertices[..., 2]                               # (B, V)
        xy    = vertices[..., :2] / (z * width)[..., None]    # (B, V, 2)
        return jnp.concatenate([xy, z[..., None]], axis=-1)   # (B, V, 3)

    def perspective_projection(self, vertices, K, rmat, tvec):
        r"""Full pinhole camera projection (``"projection"`` mode).

        Args:
            vertices: (B, V, 3) in world space.
            K:    (B, 3, 3) or (3, 3) intrinsics.
            rmat: (B, 3, 3) or (3, 3) rotation.
            tvec: (B, 3)    or (3,)   translation.

        Returns:
            (B, V, 3) projected vertices (x, y in image, z = depth).
        """
        # Ensure batch dims
        if rmat.ndim == 2: rmat = rmat[None]
        if tvec.ndim == 1: tvec = tvec[None]
        if K.ndim    == 2: K    = K[None]

        # X_cam = R @ (X - t)
        tvec_v = tvec[:, None, :]                               # (B, 1, 3)
        vertices = vertices - tvec_v                            # (B, V, 3)
        vertices = jnp.matmul(vertices, jnp.swapaxes(rmat, 1, 2))

        # Project: p = K @ X_cam
        # x_img = (K[0,0]*X + K[0,2]) / Z, similarly y
        X, Y, Z = vertices[..., 0], vertices[..., 1], vertices[..., 2]
        Z = jnp.maximum(Z, 1e-6)
        fx = K[:, 0:1, 0:1]; fy = K[:, 1:2, 1:2]
        cx = K[:, 0:1, 2:3]; cy = K[:, 1:2, 2:3]
        x_img = (fx[:, :, 0] * X + cx[:, :, 0]) / Z
        y_img = (fy[:, :, 0] * Y + cy[:, :, 0]) / Z
        return jnp.stack([x_img, y_img, Z], axis=-1)

    # ── mesh helpers ──────────────────────────────────────────────────────────

    def vertices_to_faces(self, vertices, faces):
        r"""Gather per-face vertices.

        Args:
            vertices: (B, V, 3)
            faces:    (B, F, 3)  integer vertex indices in [0, V).

        Returns:
            (B, F, 3, 3): per-face vertex positions.
        """
        B, V, _ = vertices.shape
        flat_v  = vertices.reshape(B * V, 3)
        offset  = (jnp.arange(B) * V)[:, None, None]           # (B, 1, 1)
        idx     = faces + offset                                 # (B, F, 3)
        return flat_v[idx]                                       # (B, F, 3, 3)

    def textures_to_faces(self, textures, faces):
        r"""Gather per-face textures (for ``texture_type="vertex"``).

        Args:
            textures: (B, V, ...) per-vertex texture attributes.
            faces:    (B, F, 3)  vertex indices.

        Returns:
            (B, F, 3, ...) per-face texture values.
        """
        B, V   = textures.shape[:2]
        rest   = textures.shape[2:]
        flat_t = textures.reshape(B * V, *rest)
        offset = (jnp.arange(B) * V)[:, None, None]            # (B, 1, 1)
        idx    = faces + offset                                  # (B, F, 3)
        return flat_t[idx]                                       # (B, F, 3, ...)

    # ── camera positioning helpers ─────────────────────────────────────────────

    def set_eye_from_angles(
        self,
        distance: Union[int, float],
        elevation: Union[int, float],
        azimuth: Union[int, float],
        degrees: bool = True,
    ):
        r"""Set camera eye position from spherical coordinates.

        Args:
            distance:  Distance from origin.
            elevation: Elevation angle.
            azimuth:   Azimuth angle.
            degrees:   Whether angles are in degrees (default: True).
        """
        if degrees:
            elevation = math.pi / 180.0 * elevation
            azimuth   = math.pi / 180.0 * azimuth
        self.eye = jnp.array([
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
            -distance * math.cos(elevation) * math.cos(azimuth),
        ], dtype=jnp.float32)
