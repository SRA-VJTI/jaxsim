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

import jax.numpy as jnp


def _is_array(x):
    return hasattr(x, "shape")


def compute_ambient_light(
    face_vertices,
    textures,
    ambient_intensity: float = 1.0,
    ambient_color=None,
):
    r"""Computes ambient lighting for a mesh.

    Args:
        face_vertices: (B, F, 3, 3) or (B, F, 9) per-face vertices.
        textures: face textures array.
        ambient_intensity: scalar in [0, 1].
        ambient_color: (3,) RGB, defaults to ones.

    Returns:
        light: (B, F, 1, 3) light tensor to elementwise-multiply with textures.
    """
    if not _is_array(face_vertices):
        raise TypeError(
            "Expected face_vertices to be an array. Got {0}.".format(type(face_vertices))
        )
    if not _is_array(textures):
        raise TypeError(
            "Expected textures to be an array. Got {0}.".format(type(textures))
        )
    if not isinstance(ambient_intensity, (float, int)):
        raise TypeError(
            "Expected ambient_intensity to be float. Got {0}.".format(type(ambient_intensity))
        )
    if ambient_color is None:
        ambient_color = jnp.ones(3, dtype=face_vertices.dtype)
    if not _is_array(ambient_color):
        raise TypeError(
            "Expected ambient_color to be an array. Got {0}.".format(type(ambient_color))
        )
    if ambient_color.ndim != 1 or ambient_color.shape != (3,):
        raise ValueError(
            "ambient_color must be shape (3,). Got {0}.".format(ambient_color.shape)
        )

    ambient_intensity = float(jnp.clip(ambient_intensity, 0.0, 1.0))

    batchsize = face_vertices.shape[0]
    num_faces = face_vertices.shape[1]

    light = jnp.zeros((batchsize, num_faces, 3), dtype=face_vertices.dtype)

    if ambient_intensity == 0:
        return light[:, :, None, :]

    # I = I_a * K_a  (constant everywhere)
    light = light + ambient_intensity * ambient_color[None, None, :]

    return light[:, :, None, :]


def apply_ambient_light(
    face_vertices,
    textures,
    ambient_intensity: float = 1.0,
    ambient_color=None,
):
    r"""Computes and applies ambient lighting to textures."""
    if ambient_color is None:
        ambient_color = jnp.ones(3, dtype=face_vertices.dtype)
    light = compute_ambient_light(face_vertices, textures, ambient_intensity, ambient_color)
    return textures * light


def compute_directional_light(
    face_vertices,
    textures,
    directional_intensity: float = 1.0,
    directional_color=None,
    direction=None,
):
    r"""Computes directional lighting for a mesh.

    Args:
        face_vertices: (B, F, 3, 3) per-face vertices.
        textures: face textures array.
        directional_intensity: scalar in [0, 1].
        directional_color: (3,) RGB, defaults to ones.
        direction: (3,) light direction, defaults to (0, 1, 0).

    Returns:
        light: (B, F, 1, 3) light tensor.
    """
    if not _is_array(face_vertices):
        raise TypeError(
            "Expected face_vertices to be an array. Got {0}.".format(type(face_vertices))
        )
    if not _is_array(textures):
        raise TypeError(
            "Expected textures to be an array. Got {0}.".format(type(textures))
        )
    if not isinstance(directional_intensity, (float, int)):
        raise TypeError(
            "Expected directional_intensity to be float. Got {0}.".format(type(directional_intensity))
        )
    if directional_color is None:
        directional_color = jnp.ones(3, dtype=face_vertices.dtype)
    if not _is_array(directional_color):
        raise TypeError(
            "Expected directional_color to be an array. Got {0}.".format(type(directional_color))
        )
    if direction is None:
        direction = jnp.array([0.0, 1.0, 0.0], dtype=face_vertices.dtype)
    if not _is_array(direction):
        raise TypeError(
            "Expected direction to be an array. Got {0}.".format(type(direction))
        )
    if directional_color.ndim != 1 or directional_color.shape != (3,):
        raise ValueError(
            "directional_color must be shape (3,). Got {0}.".format(directional_color.shape)
        )
    if direction.ndim != 1 or direction.shape != (3,):
        raise ValueError(
            "direction must be shape (3,). Got {0}.".format(direction.shape)
        )

    directional_intensity = float(jnp.clip(directional_intensity, 0.0, 1.0))

    batchsize = face_vertices.shape[0]
    num_faces = face_vertices.shape[1]

    light = jnp.zeros((batchsize, num_faces, 3), dtype=face_vertices.dtype)

    if directional_intensity == 0:
        return light[:, :, None, :]

    # Compute face normals from per-face vertex positions (B, F, 3, 3)
    v10 = face_vertices[:, :, 0] - face_vertices[:, :, 1]   # (B, F, 3)
    v12 = face_vertices[:, :, 2] - face_vertices[:, :, 1]   # (B, F, 3)
    normals = jnp.cross(v12, v10)                            # (B, F, 3)
    n_len = jnp.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / jnp.maximum(n_len, 1e-6)             # normalised

    # cos(angle between normal and light direction), clamped to [0, 1]
    dir_bcast = direction[None, None, :]                     # (1, 1, 3)
    cos = jnp.maximum(jnp.sum(normals * dir_bcast, axis=2), 0.0)  # (B, F)

    light = (light
             + directional_intensity
             * directional_color[None, None, :]
             * cos[:, :, None])

    return light[:, :, None, :]


def apply_directional_light(
    face_vertices,
    textures,
    directional_intensity: float = 1.0,
    directional_color=None,
    direction=None,
):
    r"""Computes and applies directional lighting to textures."""
    if directional_color is None:
        directional_color = jnp.ones(3, dtype=face_vertices.dtype)
    if direction is None:
        direction = jnp.array([0.0, 1.0, 0.0], dtype=face_vertices.dtype)
    light = compute_directional_light(
        face_vertices, textures, directional_intensity, directional_color, direction
    )
    return textures * light
