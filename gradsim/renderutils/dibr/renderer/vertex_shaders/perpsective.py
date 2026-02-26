# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Licensed under the MIT License.
#
# Pure JAX replacement — no torch dependency.

from __future__ import division, print_function

import jax.numpy as jnp


def perspective_projection(points_bxpx3, faces_fx3, cameras):
    """Project 3-D mesh vertices through a pinhole camera and gather per-face data.

    Args:
        points_bxpx3: (B, P, 3) world-space vertices.
        faces_fx3: (F, 3) face vertex indices (shared across batch).
        cameras: tuple (camera_rot_bx3x3, camera_pos_bx3, camera_proj_3x1)
            camera_rot_bx3x3 : (B, 3, 3) rotation matrix (world→cam axes as rows)
            camera_pos_bx3   : (B, 3)    camera position in world space
            camera_proj_3x1  : (B, 3, 1) or (1, 3, 1)  [[fx], [fy], [-1]]

    Returns:
        points3d_bxfx9: (B, F, 9)  per-face camera-space vertices [v0,v1,v2]
        points2d_bxfx6: (B, F, 6)  per-face projected 2-D coords [v0.xy, v1.xy, v2.xy]
        normal_bxfx3  : (B, F, 3)  unnormalised face normals in camera space
    """
    camera_rot_bx3x3, camera_pos_bx3, camera_proj_3x1 = cameras
    # camera_rot rows are [right, up, backward]; transpose gives columns = axes
    cameratrans_rot_bx3x3 = jnp.transpose(camera_rot_bx3x3, (0, 2, 1))  # (B,3,3)

    # Transform to camera space: p_cam = (p_world - cam_pos) @ cam_rot^T
    points_bxpx3 = points_bxpx3 - camera_pos_bx3[:, None, :]              # (B,P,3)
    points_bxpx3 = jnp.matmul(points_bxpx3, cameratrans_rot_bx3x3)       # (B,P,3)

    # Perspective projection: [x*fx, y*fy, z*(-1)] → divide x,y by third coord
    camera_proj_bx1x3 = camera_proj_3x1.reshape(-1, 1, 3)                 # (B,1,3)
    xy_bxpx3 = points_bxpx3 * camera_proj_bx1x3                           # (B,P,3)
    xy_bxpx2 = xy_bxpx3[:, :, :2] / xy_bxpx3[:, :, 2:3]                  # (B,P,2)

    # Gather per-face 3-D and 2-D vertex data
    pf0_bxfx3 = points_bxpx3[:, faces_fx3[:, 0], :]                       # (B,F,3)
    pf1_bxfx3 = points_bxpx3[:, faces_fx3[:, 1], :]
    pf2_bxfx3 = points_bxpx3[:, faces_fx3[:, 2], :]
    points3d_bxfx9 = jnp.concatenate([pf0_bxfx3, pf1_bxfx3, pf2_bxfx3], axis=2)

    xy_f0 = xy_bxpx2[:, faces_fx3[:, 0], :]
    xy_f1 = xy_bxpx2[:, faces_fx3[:, 1], :]
    xy_f2 = xy_bxpx2[:, faces_fx3[:, 2], :]
    points2d_bxfx6 = jnp.concatenate([xy_f0, xy_f1, xy_f2], axis=2)

    # Face normals (unnormalised)
    v01_bxfx3 = pf1_bxfx3 - pf0_bxfx3
    v02_bxfx3 = pf2_bxfx3 - pf0_bxfx3
    normal_bxfx3 = jnp.cross(v01_bxfx3, v02_bxfx3, axis=2)

    return points3d_bxfx9, points2d_bxfx6, normal_bxfx3
