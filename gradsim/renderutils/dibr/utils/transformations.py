# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# Kornia components: Copyright (C) 2017-2019, Arraiy / Open Source Vision Foundation / Kornia authors.
# Licensed under the Apache License, Version 2.0.
#
# Pure JAX/NumPy replacement — no torch dependency.

import numpy as np
import jax.numpy as jnp


def _to_np_1d(theta):
    """Accept float, int, np.ndarray, or jnp array; return flat np.ndarray."""
    if isinstance(theta, (float, int)):
        return np.array([float(theta)], dtype=np.float64)
    arr = np.asarray(theta).ravel()
    return arr.astype(np.float64)


def rotx(theta, enc="rad"):
    r"""Returns a (B, 3, 3) rotation matrix about the X-axis."""
    theta_np = _to_np_1d(theta)
    n = len(theta_np)
    c, s = np.cos(theta_np), np.sin(theta_np)
    rx = np.zeros((n, 3, 3), dtype=np.float32)
    rx[:, 0, 0] = 1.0
    rx[:, 1, 1] = c
    rx[:, 2, 2] = c
    rx[:, 1, 2] = -s
    rx[:, 2, 1] = s
    return jnp.array(rx)


def roty(theta, enc="rad"):
    r"""Returns a (B, 3, 3) rotation matrix about the Y-axis."""
    theta_np = _to_np_1d(theta)
    n = len(theta_np)
    c, s = np.cos(theta_np), np.sin(theta_np)
    ry = np.zeros((n, 3, 3), dtype=np.float32)
    ry[:, 1, 1] = 1.0
    ry[:, 0, 0] = c
    ry[:, 2, 2] = c
    ry[:, 2, 0] = -s
    ry[:, 0, 2] = s
    return jnp.array(ry)


def rotz(theta, enc="rad"):
    r"""Returns a (B, 3, 3) rotation matrix about the Z-axis."""
    theta_np = _to_np_1d(theta)
    n = len(theta_np)
    c, s = np.cos(theta_np), np.sin(theta_np)
    rz = np.zeros((n, 3, 3), dtype=np.float32)
    rz[:, 2, 2] = 1.0
    rz[:, 0, 0] = c
    rz[:, 1, 1] = c
    rz[:, 0, 1] = -s
    rz[:, 1, 0] = s
    return jnp.array(rz)


def homogenize_points(pts):
    r"""Append a column of ones to convert pts to homogeneous coordinates."""
    if pts.ndim < 2:
        raise ValueError("Input must have at least 2 dims.")
    ones = jnp.ones(pts.shape[:-1] + (1,), dtype=pts.dtype)
    return jnp.concatenate([pts, ones], axis=-1)


def unhomogenize_points(pts):
    r"""Divide by last coordinate to convert from homogeneous to Euclidean."""
    if pts.ndim < 2:
        raise ValueError("Input must have at least 2 dims.")
    w = pts[..., -1:]
    eps = 1e-6
    scale = jnp.where(jnp.abs(w) > eps, 1.0 / w, jnp.ones_like(w))
    return scale * pts[..., :-1]


def transform3d(pts, tform):
    r"""Apply a (B, 4, 4) projective transform to homogeneous pts (..., 4)."""
    pts_tformed_homo = jnp.matmul(jnp.expand_dims(tform, -3), pts[..., None])
    return unhomogenize_points(pts_tformed_homo.squeeze(-1))[..., :3]


def compute_camera_params(azimuth: float, elevation: float, distance: float):
    """Return (cam_mat_3x3, cam_pos_3) as JAX arrays (float32)."""
    theta = np.deg2rad(azimuth)
    phi = np.deg2rad(elevation)

    camY = distance * np.sin(phi)
    temp = distance * np.cos(phi)
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)
    cam_pos = np.array([camX, camY, camZ])

    axisZ = cam_pos.copy()
    axisY = np.array([0.0, 1.0, 0.0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    cam_mat = np.array([axisX, axisY, axisZ])
    l2 = np.atleast_1d(np.linalg.norm(cam_mat, 2, 1))
    l2[l2 == 0] = 1.0
    cam_mat = cam_mat / np.expand_dims(l2, 1)

    return jnp.array(cam_mat, dtype=jnp.float32), jnp.array(cam_pos, dtype=jnp.float32)
