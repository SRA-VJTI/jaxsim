# This code contains NVIDIA Confidential Information and is disclosed to you
# under a form of NVIDIA software license agreement provided separately to you.

# Notice
# NVIDIA Corporation and its licensors retain all intellectual property and
# proprietary rights in and to this software and related documentation and
# any modifications thereto. Any use, reproduction, disclosure, or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.

# ALL NVIDIA DESIGN SPECIFICATIONS, CODE ARE PROVIDED "AS IS.". NVIDIA MAKES
# NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
# THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT,
# MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.

# Information and code furnished is believed to be accurate and reliable.
# However, NVIDIA Corporation assumes no responsibility for the consequences of use of such
# information or for any infringement of patents or other rights of third parties that may
# result from its use. No license is granted by implication or otherwise under any patent
# or patent rights of NVIDIA Corporation. Details are subject to change without notice.
# This code supersedes and replaces all information previously supplied.
# NVIDIA Corporation products are not authorized for use as critical
# components in life support devices or systems without express written approval of
# NVIDIA Corporation.

# Copyright (c) 2020-2021 NVIDIA Corporation. All rights reserved.

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import types as _types

from . import util
from . import config
from .model import *

# ── Bare primitive aliases (previously from .adjoint import *) ────────────────
from .util import (
    length, normalize,
    quat_identity, quat_inverse, quat_rotate, quat_multiply,
    transform as _transform_fn,
    transform_identity, transform_multiply,
    transform_inertia, transform_twist,
    spatial_vector, spatial_cross, spatial_cross_dual,
    spatial_dot, spatial_matrix, spatial_matrix_from_inertia,
)

# Bare aliases for articulation functions
spatial_transform = _transform_fn
spatial_transform_identity = transform_identity
spatial_transform_multiply = transform_multiply
spatial_transform_twist = transform_twist
spatial_transform_inertia = transform_inertia

float3 = lambda *a: jnp.array(a, dtype=jnp.float32)
float4 = lambda *a: jnp.array(a, dtype=jnp.float32)
dot    = jnp.dot
cross  = jnp.cross
clamp  = jnp.clip
tid    = lambda: 0
load   = lambda arr, i: arr[i]
store  = lambda arr, i, v: arr   # immutable; real impl uses .at[i].set(v)


# ── df compatibility namespace ────────────────────────────────────────────────
class _NoopTape:
    adjoints = {}
    def launch(self, **kwargs): pass
    def replay(self): pass
    def reset(self): pass


df = _types.SimpleNamespace(
    # type stubs used in function annotations
    float3=float3, float4=float4,
    mat22=object, mat33=object, quat=object,
    spatial_transform=object,
    int=int, float=float,
    tensor=lambda t: t,           # df.tensor(df.float3) → just returns the arg
    func=lambda f: f,             # @df.func → identity decorator

    # scalar / vector math
    dot=jnp.dot,
    cross=jnp.cross,
    normalize=normalize,
    length=length,
    clamp=jnp.clip,
    min=jnp.minimum,
    max=jnp.maximum,
    mul=jnp.matmul,
    transpose=jnp.transpose,
    determinant=jnp.linalg.det,
    leaky_min=lambda a, b, s: jnp.where(a < b, a + s * (b - a), b),
    step=lambda x: (x > 0.0).astype(jnp.float32),

    # array primitives
    load=lambda arr, i: arr[i],
    store=lambda arr, i, v: arr,
    atomic_add=lambda arr, i, v: arr,
    atomic_sub=lambda arr, i, v: arr,
    tid=lambda: 0,

    # quaternion / transform
    quat_rotate=quat_rotate,
    quat_multiply=quat_multiply,
    transform_point=lambda T, p: p,        # placeholder
    transform_multiply=transform_multiply,
    transform_identity=transform_identity,

    # spatial
    spatial_vector=spatial_vector,
    spatial_cross=spatial_cross,
    spatial_cross_dual=spatial_cross_dual,
    spatial_dot=spatial_dot,
    spatial_matrix=spatial_matrix,
    spatial_matrix_from_inertia=spatial_matrix_from_inertia,
    transform_inertia=transform_inertia,
    transform_twist=transform_twist,
    spatial_transform_twist=transform_twist,
    spatial_transform_inertia=transform_inertia,
    nonzero=lambda x: (x != 0.0).astype(jnp.float32),
    rotate=lambda q, v: quat_rotate(q, v),
    rotate_inv=lambda q, v: quat_rotate(jnp.array([q[0], -q[1], -q[2], -q[3]]), v),
    sign=jnp.sign,
    abs=jnp.abs,
    acos=jnp.arccos,

    # tape / autodiff (replaced by jax.grad)
    Tape=_NoopTape,
    compile=lambda: None,

    # PyTorch-era helpers (stubs)
    to_weak_list=list,
    to_strong_list=list,
    make_contiguous=list,
    filter_grads=list,
)

# Todo
#-----
#
# [x] Spring model
# [x] 2D FEM model
# [x] 3D FEM model
# [x] Cloth
#     [x] Wind/Drag model
#     [x] Bending model
#     [x] Triangle collision
# [x] Rigid body model
# [x] Rigid shape contact
#     [x] Sphere
#     [x] Capsule
#     [x] Box
#     [ ] Convex
#     [ ] SDF
# [ ] Implicit solver
# [x] USD import
# [x] USD export
# -----

# externally compiled kernels module (C++/CUDA code with PyBind entry points)
kernels = None

def test(c: float):

    x = 1.0

    if (c < 3.0):
        x = 2.0

    return x*6.0



def kernel_init():
    pass  # no-op: dflex kernels replaced by JAX


def integrate_particles(x, v, f, w, gravity, dt):
    """Semi-implicit Euler integration for particles.

    Args:
        x:       (N, 3) positions
        v:       (N, 3) velocities
        f:       (N, 3) net forces
        w:       (N,)   inverse masses  (0 = kinematic/pinned)
        gravity: (3,) or (1, 3)  gravity vector
        dt:      float  timestep
    Returns:
        x_new:   (N, 3) updated positions
        v_new:   (N, 3) updated velocities
    """
    g = gravity[0] if gravity.ndim == 2 else gravity          # (3,)
    inv_mass = w[:, None]                                      # (N, 1)
    # df.step(0 - inv_mass): guard term; >0 only when inv_mass<0 (unused for valid particles)
    step_mask = (0.0 - inv_mass > 0.0).astype(jnp.float32)    # (N, 1)
    v_new = v + (f * inv_mass + g * step_mask) * dt
    x_new = x + v_new * dt
    return x_new, v_new


# semi-implicit Euler integration
def integrate_rigids(rigid_x, rigid_r, rigid_v, rigid_w,
                     rigid_f, rigid_t, inv_m, inv_I, gravity, dt):

    """Semi-implicit Euler for rigid bodies.

    Returns x_new (B,3), r_new (B,4), v_new (B,3), w_new (B,3).
    """
    g = gravity[0] if gravity.ndim == 2 else gravity          # (3,)
    nonzero_m = (inv_m != 0.0).astype(jnp.float32)            # (B,)

    # linear
    v_new = rigid_v + (rigid_f * inv_m[:, None] + g * nonzero_m[:, None]) * dt
    x_new = rigid_x + v_new * dt

    # angular: rotate angular velocity / torque to body frame, apply I^{-1}, rotate back
    def _rot_inv(r, vec):
        r_conj = jnp.array([r[0], -r[1], -r[2], -r[3]])
        return quat_rotate(r_conj, vec)

    wb = jax.vmap(_rot_inv)(rigid_r, rigid_w)                 # (B,3) body-frame ω
    tb = jax.vmap(_rot_inv)(rigid_r, rigid_t)                 # (B,3) body-frame τ

    wb_new = wb + jnp.einsum('bij,bj->bi', inv_I, tb) * dt   # (B,3)
    w_new  = jax.vmap(quat_rotate)(rigid_r, wb_new)           # (B,3) world-frame

    # integrate quaternion: r_dot = [0, w] * r * 0.5
    def _update_quat(r, w):
        w_q  = jnp.array([0.0, w[0], w[1], w[2]])
        r_dot = quat_multiply(w_q, r) * 0.5
        r_new = r + r_dot * dt
        return r_new / jnp.linalg.norm(r_new)

    r_new = jax.vmap(_update_quat)(rigid_r, w_new)            # (B,4)

    return x_new, r_new, v_new, w_new


def eval_springs(x, v, spring_indices, spring_rest_lengths, spring_stiffness, spring_damping):
    """Compute spring forces for all springs and scatter to particles.

    Args:
        x:                  (N, 3) positions
        v:                  (N, 3) velocities
        spring_indices:     (2*S,) flat int array, alternating [i, j] pairs
        spring_rest_lengths:(S,)   rest lengths
        spring_stiffness:   (S,)   stiffness ke
        spring_damping:     (S,)   damping kd
    Returns:
        f:  (N, 3) force contributions (scatter-accumulated over all springs)
    """
    idx = spring_indices.reshape(-1, 2)   # (S, 2)
    i_idx = idx[:, 0]                     # (S,)
    j_idx = idx[:, 1]                     # (S,)

    xi = x[i_idx]    # (S, 3)
    xj = x[j_idx]    # (S, 3)
    vi = v[i_idx]    # (S, 3)
    vj = v[j_idx]    # (S, 3)

    xij = xi - xj
    vij = vi - vj

    l = jnp.linalg.norm(xij, axis=-1)               # (S,)
    dir = xij / l[:, None]                           # (S, 3) normalized

    c    = l - spring_rest_lengths                   # (S,)   stretch
    dcdt = jnp.sum(dir * vij, axis=-1)               # (S,)   vel along spring

    # spring + damping force magnitude along the spring axis
    fs = dir * (spring_stiffness * c + spring_damping * dcdt)[:, None]   # (S, 3)

    # scatter: f[i] -= fs  (spring pulls i toward j)
    #          f[j] += fs  (spring pulls j toward i)
    f = jnp.zeros_like(x)
    f = f.at[i_idx].add(-fs)
    f = f.at[j_idx].add(fs)
    return f


def eval_triangles(x, v, indices, pose, activation,
                   k_mu, k_lambda, k_damp, k_drag, k_lift):
    """Neo-Hookean cloth FEM with lift/drag. Returns (N,3) force contributions."""
    idx = indices.reshape(-1, 3)
    ii, jj, kk = idx[:, 0], idx[:, 1], idx[:, 2]

    p, q, r   = x[ii], x[jj], x[kk]
    vp, vq, vr = v[ii], v[jj], v[kk]
    qp = q - p;  rp = r - p

    Dm = pose                                                   # (T,2,2)
    inv_ra = jnp.linalg.det(Dm) * 2.0                          # (T,)
    ra     = 1.0 / jnp.where(jnp.abs(inv_ra) > 1e-10, inv_ra, 1.0)

    kmu = k_mu * ra;  klam = k_lambda * ra;  kdamp = k_damp * ra

    # Deformation gradient
    f1 = qp * Dm[:, 0, 0:1] + rp * Dm[:, 1, 0:1]              # (T,3)
    f2 = qp * Dm[:, 0, 1:2] + rp * Dm[:, 1, 1:2]

    fq = (f1 * Dm[:, 0, 0:1] + f2 * Dm[:, 0, 1:2]) * kmu[:, None]
    fr = (f1 * Dm[:, 1, 0:1] + f2 * Dm[:, 1, 1:2]) * kmu[:, None]
    alpha = 1.0 + kmu / jnp.where(jnp.abs(klam) > 1e-10, klam, 1.0)

    # Area preservation
    n_vec = jnp.cross(qp, rp)
    n_len = jnp.linalg.norm(n_vec, axis=-1)
    area  = n_len * 0.5
    act   = activation                                          # (T,)
    c     = area * inv_ra - alpha + act

    n_hat = n_vec / jnp.where(n_len > 1e-10, n_len, 1.0)[:, None]
    dcdq  = jnp.cross(rp, n_hat) * inv_ra[:, None] * 0.5
    dcdr  = jnp.cross(n_hat, qp) * inv_ra[:, None] * 0.5

    f_area = klam * c
    dcdt   = (jnp.sum(dcdq * vq, axis=-1) + jnp.sum(dcdr * vr, axis=-1)
              - jnp.sum((dcdq + dcdr) * vp, axis=-1))
    f_damp = kdamp * dcdt

    fq = fq + dcdq * (f_area + f_damp)[:, None]
    fr = fr + dcdr * (f_area + f_damp)[:, None]
    fp = fq + fr

    # Lift + Drag
    vmid   = (vp + vq + vr) * (1.0 / 3.0)
    vmid_n = jnp.linalg.norm(vmid, axis=-1, keepdims=True)
    vdir   = vmid / jnp.where(vmid_n > 1e-10, vmid_n, 1.0)

    f_drag = vmid * (k_drag * area * jnp.abs(jnp.sum(n_hat * vmid, axis=-1)))[:, None]
    cos_nv = jnp.clip(jnp.sum(n_hat * vdir, axis=-1), -1.0, 1.0)
    f_lift = n_hat * (k_lift * area * (1.57079 - jnp.arccos(cos_nv))
                      * jnp.sum(vmid ** 2, axis=-1))[:, None]

    fp = fp - f_drag - f_lift
    fq = fq + f_drag + f_lift
    fr = fr + f_drag + f_lift

    f = jnp.zeros_like(x)
    f = f.at[ii].add(fp)
    f = f.at[jj].add(-fq)
    f = f.at[kk].add(-fr)
    return f

def triangle_closest_point_barycentric(a: df.float3, b: df.float3, c: df.float3, p: df.float3):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = df.dot(ab, ap)
    d2 = df.dot(ac, ap)

    if (d1 <= 0.0 and d2 <= 0.0):
        return float3(1.0, 0.0, 0.0)

    bp = p - b
    d3 = df.dot(ab, bp)
    d4 = df.dot(ac, bp)

    if (d3 >= 0.0 and d4 <= d3):
        return float3(0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    if (vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0):
        return float3(1.0 - v, v, 0.0)

    cp = p - c
    d5 = dot(ab, cp)
    d6 = dot(ac, cp)

    if (d6 >= 0.0 and d5 <= d6):
        return float3(0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    if (vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0):
        return float3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    if (va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0):
        return float3(0.0, w, 1.0 - w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom

    return float3(1.0 - v - w, v, w)


def _triangle_barycentric_batch(a, b, c, p):
    """Vectorized triangle closest-point barycentric coords.
    All inputs shape (M, 3). Returns (M, 3) barycentric coords."""
    ab = b - a; ac = c - a; ap = p - a
    d1 = jnp.sum(ab * ap, axis=-1); d2 = jnp.sum(ac * ap, axis=-1)

    bp = p - b
    d3 = jnp.sum(ab * bp, axis=-1); d4 = jnp.sum(ac * bp, axis=-1)

    vc = d1 * d4 - d3 * d2
    denom_ab = jnp.where(jnp.abs(d1 - d3) > 1e-10, d1 - d3, 1.0)
    v_ab = d1 / denom_ab

    cp = p - c
    d5 = jnp.sum(ab * cp, axis=-1); d6 = jnp.sum(ac * cp, axis=-1)

    vb = d5 * d2 - d1 * d6
    denom_ac = jnp.where(jnp.abs(d2 - d6) > 1e-10, d2 - d6, 1.0)
    w_ac = d2 / denom_ac

    va = d3 * d6 - d5 * d4
    bc_denom = (d4 - d3) + (d5 - d6)
    denom_bc = jnp.where(jnp.abs(bc_denom) > 1e-10, bc_denom, 1.0)
    w_bc = (d4 - d3) / denom_bc

    int_denom = jnp.where(jnp.abs(va + vb + vc) > 1e-10, va + vb + vc, 1.0)
    v_int = vb / int_denom
    w_int = vc / int_denom

    # Start from interior, then override with boundary regions
    # in reverse priority order (highest priority applied last)
    b0 = 1.0 - v_int - w_int
    b1 = v_int
    b2 = w_int

    def sel(cond, b0n, b1n, b2n, b0o, b1o, b2o):
        return (jnp.where(cond, b0n, b0o),
                jnp.where(cond, b1n, b1o),
                jnp.where(cond, b2n, b2o))

    # edge bc
    c_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    b0, b1, b2 = sel(c_bc, 0.0, w_bc, 1.0 - w_bc, b0, b1, b2)
    # edge ac
    c_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    b0, b1, b2 = sel(c_ac, 1.0 - w_ac, 0.0, w_ac, b0, b1, b2)
    # vertex c
    c_vc = (d6 >= 0.0) & (d5 <= d6)
    b0, b1, b2 = sel(c_vc, 0.0, 0.0, 1.0, b0, b1, b2)
    # edge ab
    c_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    b0, b1, b2 = sel(c_ab, 1.0 - v_ab, v_ab, 0.0, b0, b1, b2)
    # vertex b
    c_vb = (d3 >= 0.0) & (d4 <= d3)
    b0, b1, b2 = sel(c_vb, 0.0, 1.0, 0.0, b0, b1, b2)
    # vertex a (highest priority)
    c_va = (d1 <= 0.0) & (d2 <= 0.0)
    b0, b1, b2 = sel(c_va, 1.0, 0.0, 0.0, b0, b1, b2)

    return jnp.stack([b0, b1, b2], axis=-1)  # (M, 3)


# @df.func
# def triangle_closest_point(a: df.float3, b: df.float3, c: df.float3, p: df.float3):
#     ab = b - a
#     ac = c - a
#     ap = p - a

#     d1 = df.dot(ab, ap)
#     d2 = df.dot(ac, ap)

#     if (d1 <= 0.0 and d2 <= 0.0):
#         return a

#     bp = p - b
#     d3 = df.dot(ab, bp)
#     d4 = df.dot(ac, bp)

#     if (d3 >= 0.0 and d4 <= d3):
#         return b

#     vc = d1 * d4 - d3 * d2
#     v = d1 / (d1 - d3)
#     if (vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0):
#         return a + ab * v

#     cp = p - c
#     d5 = dot(ab, cp)
#     d6 = dot(ac, cp)

#     if (d6 >= 0.0 and d5 <= d6):
#         return c

#     vb = d5 * d2 - d1 * d6
#     w = d2 / (d2 - d6)
#     if (vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0):
#         return a + ac * w

#     va = d3 * d6 - d5 * d4
#     w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
#     if (va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0):
#         return b + (c - b) * w

#     denom = 1.0 / (va + vb + vc)
#     v = vb * denom
#     w = vc * denom

#     return a + ab * v + ac * w


def eval_triangles_contact(x, v, indices, k_contact=1e5, threshold=0.01):
    """Particle–triangle contact forces (cloth self-collision).

    Each particle is tested against every triangle it does not belong to.
    Returns (N,3) force array.

    x       : (N,3) particle positions
    v       : (N,3) particle velocities (unused but kept for API symmetry)
    indices : (T*3,) or (T,3) triangle vertex indices
    """
    N = x.shape[0]
    idx = indices.reshape(-1, 3)          # (T,3)
    T   = idx.shape[0]

    ii = idx[:, 0]; jj = idx[:, 1]; kk = idx[:, 2]   # (T,)

    # tile: for each (face, particle) pair → shape (T*N,)
    face_rep  = jnp.repeat(jnp.arange(T), N)          # face index
    part_rep  = jnp.tile(jnp.arange(N), T)            # particle index

    fi = ii[face_rep]; fj = jj[face_rep]; fk = kk[face_rep]

    # mask out pairs where the particle is a vertex of the face
    own = (part_rep == fi) | (part_rep == fj) | (part_rep == fk)

    # triangle vertices and the query particle (all shape (T*N, 3))
    p = x[fi]; q = x[fj]; r = x[fk]
    pos = x[part_rep]

    bary = _triangle_barycentric_batch(p, q, r, pos)   # (T*N, 3)
    closest = p * bary[:, 0:1] + q * bary[:, 1:2] + r * bary[:, 2:3]

    diff = pos - closest
    dist = jnp.sum(diff * diff, axis=-1)               # squared dist
    d_len = jnp.sqrt(jnp.maximum(dist, 1e-20))
    n = diff / d_len[:, None]

    c = jnp.minimum(dist - threshold, 0.0)             # penetration depth (<=0)
    fn_mag = c * k_contact                             # negative scalar force

    # zero out own-vertex pairs
    active = (~own).astype(jnp.float32)
    fn = n * (fn_mag * active)[:, None]                # (T*N, 3)

    f = jnp.zeros_like(x)
    f = f.at[part_rep].add(-fn)                        # repel particle
    f = f.at[fi].add(fn * bary[:, 0:1])
    f = f.at[fj].add(fn * bary[:, 1:2])
    f = f.at[fk].add(fn * bary[:, 2:3])
    return f


def eval_triangles_rigid_contacts(
        x, v, indices,
        rigid_x, rigid_r, rigid_v, rigid_w,
        contact_body, contact_point, contact_dist, contact_mat,
        materials):
    """Rigid-body contact forces distributed onto cloth triangles.

    Returns tri_f : (N,3) force array for cloth particles.

    x            : (N,3) cloth particle positions
    v            : (N,3) cloth particle velocities
    indices      : (T*3,) or (T,3) triangle vertex indices
    rigid_x      : (B,3)  rigid body positions
    rigid_r      : (B,4)  rigid body orientations (quaternions)
    rigid_v      : (B,3)  rigid body linear velocities
    rigid_w      : (B,3)  rigid body angular velocities
    contact_body : (C,)   which rigid body each contact point belongs to
    contact_point: (C,3)  contact point in body-local frame
    contact_dist : (C,)   contact shape thickness
    contact_mat  : (C,)   material index per contact point
    materials    : (M,4)  [ke, kd, kf, mu] per material
    """
    N = x.shape[0]
    idx = indices.reshape(-1, 3)   # (T,3)
    T   = idx.shape[0]
    C   = contact_body.shape[0]   # number of contact points

    ii = idx[:, 0]; jj = idx[:, 1]; kk = idx[:, 2]

    # ── per-contact: world-space position and velocity of the contact point ──
    cb  = contact_body                              # (C,)
    x0  = rigid_x[cb]                              # (C,3)
    r0  = rigid_r[cb]                              # (C,4)
    v0  = rigid_v[cb]                              # (C,3)
    w0  = rigid_w[cb]                              # (C,3)

    # rotate contact point into world frame
    cp_world = jax.vmap(quat_rotate)(r0, contact_point)   # (C,3)
    r_arm    = cp_world                             # moment arm from body centre
    r_hat    = r_arm / jnp.maximum(jnp.linalg.norm(r_arm, axis=-1, keepdims=True), 1e-10)
    pos      = x0 + r_arm + r_hat * contact_dist[:, None]  # world-space contact position (C,3)
    dpdt     = v0 + jnp.cross(w0, r_arm)           # contact point velocity (C,3)

    ke_all = materials[contact_mat, 0]
    kd_all = materials[contact_mat, 1]
    kf_all = materials[contact_mat, 2]
    mu_all = materials[contact_mat, 3]

    # ── tile: (face × contact) pairs ──
    face_rep = jnp.repeat(jnp.arange(T), C)        # (T*C,)
    cont_rep = jnp.tile(jnp.arange(C), T)          # (T*C,)

    fi = ii[face_rep]; fj = jj[face_rep]; fk = kk[face_rep]
    p  = x[fi]; q = x[fj]; r_v = x[fk]
    vp = v[fi]; vq = v[fj]; vr = v[fk]
    P  = pos[cont_rep]; Dpdt = dpdt[cont_rep]
    ke = ke_all[cont_rep]; kd = kd_all[cont_rep]
    kf = kf_all[cont_rep]; mu = mu_all[cont_rep]

    bary    = _triangle_barycentric_batch(p, q, r_v, P)    # (T*C, 3)
    closest = p * bary[:, 0:1] + q * bary[:, 1:2] + r_v * bary[:, 2:3]

    diff = P - closest
    dist = jnp.sum(diff * diff, axis=-1)
    d_len = jnp.sqrt(jnp.maximum(dist, 1e-20))
    n    = diff / d_len[:, None]
    c    = jnp.minimum(dist - 0.05, 0.0)           # penetration (<=0)

    fn = c * ke                                     # normal force magnitude

    vtri = vp * bary[:, 0:1] + vq * bary[:, 1:2] + vr * bary[:, 2:3]
    vrel = vtri - Dpdt
    vn   = jnp.sum(n * vrel, axis=-1)
    vt   = vrel - n * vn[:, None]

    # damping (only when penetrating)
    step_c = (c < 0.0).astype(jnp.float32)
    fd = -jnp.maximum(vn, 0.0) * kd * step_c

    # Coulomb friction (box)
    lower = mu * (fn + fd)
    upper = -lower
    z3    = jnp.zeros_like(n[:, 0])
    o3    = jnp.ones_like(n[:, 0])
    nx    = jnp.cross(n, jnp.stack([ z3,  z3,  o3], axis=-1))
    nz    = jnp.cross(n, jnp.stack([ o3,  z3,  z3], axis=-1))
    vx    = jnp.clip(jnp.sum(nx * kf[:, None] * vt, axis=-1), lower, upper)
    vz    = jnp.clip(jnp.sum(nz * kf[:, None] * vt, axis=-1), lower, upper)
    ft    = (nx * vx[:, None] + nz * vz[:, None]) * (-step_c[:, None])

    f_total = n * (fn + fd)[:, None] + ft          # (T*C, 3)

    tri_f = jnp.zeros_like(x)
    tri_f = tri_f.at[fi].add(f_total * bary[:, 0:1])
    tri_f = tri_f.at[fj].add(f_total * bary[:, 1:2])
    tri_f = tri_f.at[fk].add(f_total * bary[:, 2:3])
    return tri_f


def eval_bending(x, v, indices, rest, ke, kd):
    """Dihedral bending forces for cloth.

    x       : (N,3) positions
    v       : (N,3) velocities
    indices : (B*4,) or (B,4) quad indices [i,j,k,l]
                i,j  share the bending edge (x3–x4)
                k,l  are the opposite vertices of the two triangles
    rest    : (B,)  rest dihedral angles (radians)
    ke      : scalar elastic stiffness
    kd      : scalar damping coefficient
    Returns : (N,3) force array
    """
    idx = indices.reshape(-1, 4)   # (B,4)
    ii = idx[:, 0]; jj = idx[:, 1]; kk = idx[:, 2]; ll = idx[:, 3]

    x1 = x[ii]; x2 = x[jj]; x3 = x[kk]; x4 = x[ll]
    v1 = v[ii]; v2 = v[jj]; v3 = v[kk]; v4 = v[ll]

    n1 = jnp.cross(x3 - x1, x4 - x1)   # (B,3) normal to face 1
    n2 = jnp.cross(x4 - x2, x3 - x2)   # (B,3) normal to face 2

    n1_len = jnp.linalg.norm(n1, axis=-1)   # (B,)
    n2_len = jnp.linalg.norm(n2, axis=-1)

    rcp_n1 = 1.0 / jnp.where(n1_len > 1e-10, n1_len, 1.0)
    rcp_n2 = 1.0 / jnp.where(n2_len > 1e-10, n2_len, 1.0)

    cos_theta = jnp.sum(n1 * n2, axis=-1) * rcp_n1 * rcp_n2
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)

    # n1/|n1|², n2/|n2|² for gradient computation
    n1_sq = n1 * (rcp_n1 * rcp_n1)[:, None]
    n2_sq = n2 * (rcp_n2 * rcp_n2)[:, None]

    e     = x4 - x3                     # (B,3) shared edge
    e_len = jnp.linalg.norm(e, axis=-1)
    e_hat = e / jnp.where(e_len > 1e-10, e_len, 1.0)[:, None]

    s     = jnp.sign(jnp.sum(jnp.cross(n2_sq, n1_sq) * e_hat, axis=-1))
    angle = jnp.arccos(cos_theta) * s   # signed dihedral angle

    # dihedral angle gradient vectors
    d1 = n1_sq * e_len[:, None]
    d2 = n2_sq * e_len[:, None]
    d3 = (n1_sq * jnp.sum((x1 - x4) * e_hat, axis=-1, keepdims=True)
        + n2_sq * jnp.sum((x2 - x4) * e_hat, axis=-1, keepdims=True))
    d4 = (n1_sq * jnp.sum((x3 - x1) * e_hat, axis=-1, keepdims=True)
        + n2_sq * jnp.sum((x3 - x2) * e_hat, axis=-1, keepdims=True))

    f_elastic = ke * (angle - rest)     # (B,)
    f_damp    = kd * (jnp.sum(d1 * v1, axis=-1) + jnp.sum(d2 * v2, axis=-1)
                    + jnp.sum(d3 * v3, axis=-1) + jnp.sum(d4 * v4, axis=-1))
    f_total   = -(e_len * (f_elastic + f_damp))[:, None]   # (B,1)

    f = jnp.zeros_like(x)
    f = f.at[ii].add(d1 * f_total)
    f = f.at[jj].add(d2 * f_total)
    f = f.at[kk].add(d3 * f_total)
    f = f.at[ll].add(d4 * f_total)
    return f


def eval_tetrahedra(x, v, indices, pose, activation, materials):
    """Neo-Hookean volumetric FEM for tetrahedra.

    x         : (N,3) particle positions
    v         : (N,3) particle velocities
    indices   : (T*4,) or (T,4) tet vertex indices
    pose      : (T,3,3) inverse rest-pose matrices (Dm)
    activation: (T,)   muscle activation per tet
    materials : (T,3)  [k_mu, k_lambda, k_damp] per tet
    Returns   : (N,3) force array
    """
    idx = indices.reshape(-1, 4)          # (T,4)
    T   = idx.shape[0]
    ii  = idx[:, 0]; jj = idx[:, 1]; kk = idx[:, 2]; ll = idx[:, 3]

    x0 = x[ii]; x1 = x[jj]; x2 = x[kk]; x3 = x[ll]
    v0 = v[ii]; v1 = v[jj]; v2 = v[kk]; v3 = v[ll]

    x10 = x1 - x0; x20 = x2 - x0; x30 = x3 - x0
    v10 = v1 - v0; v20 = v2 - v0; v30 = v3 - v0

    # Ds: deformed shape matrix (T,3,3) — columns are edge vectors
    Ds = jnp.stack([x10, x20, x30], axis=-1)   # (T,3,3)
    Dm = pose                                   # (T,3,3) inv rest-pose

    inv_rest_volume = jnp.linalg.det(Dm) * 6.0  # (T,)
    rest_volume     = 1.0 / jnp.where(jnp.abs(inv_rest_volume) > 1e-10, inv_rest_volume, 1.0)

    k_mu     = materials[:, 0] * rest_volume
    k_lambda = materials[:, 1] * rest_volume
    k_damp   = materials[:, 2] * rest_volume
    alpha    = (1.0 + materials[:, 0] / jnp.where(jnp.abs(materials[:, 1]) > 1e-10, materials[:, 1], 1.0)
                - materials[:, 0] / (4.0 * jnp.where(jnp.abs(materials[:, 1]) > 1e-10, materials[:, 1], 1.0)))

    # F = Ds * Dm  (deformation gradient)
    F    = jnp.einsum('bij,bjk->bik', Ds, Dm)   # (T,3,3)
    dFdt_s = jnp.stack([v10, v20, v30], axis=-1)
    dFdt = jnp.einsum('bij,bjk->bik', dFdt_s, Dm)

    # Ic = trace(F^T F)  ≈ sum of squared singular values
    Ic = jnp.sum(F * F, axis=(-2, -1))          # (T,)

    # Deviatoric PK1 + damping
    P = (F * (k_mu * (1.0 - 1.0 / (Ic + 1.0)))[:, None, None]
       + dFdt * k_damp[:, None, None])
    H = jnp.einsum('bij,bkj->bik', P, Dm)       # P * Dm^T  (T,3,3)

    # Force contributions from deviatoric part (columns of H)
    g1 = H[:, :, 0];  g2 = H[:, :, 1];  g3 = H[:, :, 2]

    # Hydrostatic (volume) part
    J = jnp.linalg.det(F)                        # (T,)
    s = (inv_rest_volume / 6.0)[:, None]          # (T,1)
    dJdx1 = jnp.cross(x20, x30) * s
    dJdx2 = jnp.cross(x30, x10) * s
    dJdx3 = jnp.cross(x10, x20) * s

    f_volume = (J - alpha + activation) * k_lambda  # (T,)
    f_damp_h = ((jnp.sum(dJdx1 * v1, axis=-1)
               + jnp.sum(dJdx2 * v2, axis=-1)
               + jnp.sum(dJdx3 * v3, axis=-1)) * k_damp)
    f_total = (f_volume + f_damp_h)[:, None]        # (T,1)

    f1 = g1 + dJdx1 * f_total
    f2 = g2 + dJdx2 * f_total
    f3 = g3 + dJdx3 * f_total
    f0 = -(f1 + f2 + f3)

    f = jnp.zeros_like(x)
    f = f.at[ii].add(-f0)
    f = f.at[jj].add(-f1)
    f = f.at[kk].add(-f2)
    f = f.at[ll].add(-f3)
    return f


def eval_contacts(x, v, ke, kd, kf, mu):
    """Ground-plane contact forces for particles.

    Ground is the Y=0 plane with upward normal (0,1,0).

    x   : (N,3) particle positions
    v   : (N,3) particle velocities
    ke  : elastic (restitution) coefficient
    kd  : damping coefficient
    kf  : friction coefficient
    mu  : Coulomb friction scale
    Returns (N,3) force array (repulsive, upward when penetrating).
    """
    n  = jnp.array([0.0, 1.0, 0.0], dtype=x.dtype)   # ground normal

    # penetration depth (<=0 when inside)
    c  = jnp.minimum(x[:, 1] - 0.01, 0.0)             # (N,)

    vn = v[:, 1]                                       # normal velocity
    vt = v - n[None] * vn[:, None]                     # tangential velocity (N,3)

    fn = c * ke                                        # normal force magnitude (N,)

    # damping (only when penetrating)
    step_c = (c < 0.0).astype(x.dtype)
    fd = jnp.minimum(vn, 0.0) * kd                    # (N,)

    # Coulomb friction (box)
    lower = mu * c * ke                                # (N,) negative
    upper = -lower

    vx = jnp.clip(vt[:, 0] * kf, lower, upper)        # (N,)
    vz = jnp.clip(vt[:, 2] * kf, lower, upper)
    ft = jnp.stack([vx, jnp.zeros_like(vx), vz], axis=-1)  # (N,3)

    # total: normal + damping + friction (friction & damping only when penetrating)
    fn_vec  = n[None] * fn[:, None]                   # (N,3)
    fd_vec  = n[None] * fd[:, None]
    ftotal  = fn_vec + (fd_vec + ft) * step_c[:, None]

    return -ftotal   # atomic_sub(f, tid, ftotal) → f -= ftotal


def eval_rigid_contacts(rigid_x, rigid_r, rigid_v, rigid_w,
                        contact_body, contact_point, contact_dist, contact_mat,
                        materials):
    """Ground-plane contact forces / torques for rigid bodies.

    Returns (rigid_f, rigid_t): per-body force and torque contributions,
    both shape (B,3).

    rigid_x      : (B,3)  body positions
    rigid_r      : (B,4)  body orientations (quaternion [x,y,z,w])
    rigid_v      : (B,3)  body linear velocities
    rigid_w      : (B,3)  body angular velocities
    contact_body : (C,)   body index per contact point
    contact_point: (C,3)  contact point in body-local frame
    contact_dist : (C,)   contact shape thickness
    contact_mat  : (C,)   material index
    materials    : (M,4)  [ke, kd, kf, mu] per material
    """
    C = contact_body.shape[0]
    B = rigid_x.shape[0]

    cb  = contact_body                              # (C,)
    x0  = rigid_x[cb]                              # (C,3)
    r0  = rigid_r[cb]                              # (C,4)
    v0  = rigid_v[cb]                              # (C,3)
    w0  = rigid_w[cb]                              # (C,3)

    # World-space contact point
    cp_world = jax.vmap(quat_rotate)(r0, contact_point)   # (C,3)
    n_ground = jnp.array([0.0, 1.0, 0.0])
    p   = x0 + cp_world - n_ground[None] * contact_dist[:, None]
    r_arm = p - x0                                 # moment arm (C,3)

    # Contact point velocity
    dpdt = v0 + jnp.cross(w0, r_arm)              # (C,3)

    ke = materials[contact_mat, 0]                 # (C,)
    kd = materials[contact_mat, 1]
    kf = materials[contact_mat, 2]
    mu = materials[contact_mat, 3]

    # Penetration depth (<=0 inside ground)
    c  = jnp.minimum(p[:, 1], 0.0)               # Y component (C,)

    vn = dpdt[:, 1]                               # normal velocity (C,)
    vt = dpdt - n_ground[None] * vn[:, None]      # tangential velocity (C,3)

    fn = c * ke                                    # normal force magnitude (C,)

    # Damping
    step_c = (c < 0.0).astype(rigid_x.dtype)
    fd = jnp.minimum(vn, 0.0) * kd * step_c       # (C,)

    # Coulomb friction (box)
    lower = mu * (fn + fd)
    upper = -lower
    vx = jnp.clip(vt[:, 0] * kf, lower, upper)    # (C,)
    vz = jnp.clip(vt[:, 2] * kf, lower, upper)
    ft = jnp.stack([vx, jnp.zeros_like(vx), vz], axis=-1) * step_c[:, None]  # (C,3)

    f_total = n_ground[None] * (fn + fd)[:, None] + ft   # (C,3)
    t_total = jnp.cross(r_arm, f_total)                  # (C,3)

    # Scatter to per-body accumulators
    rigid_f = jnp.zeros((B, 3), dtype=rigid_x.dtype)
    rigid_t = jnp.zeros((B, 3), dtype=rigid_x.dtype)
    rigid_f = rigid_f.at[cb].add(-f_total)
    rigid_t = rigid_t.at[cb].add(-t_total)
    return rigid_f, rigid_t


# compute transform across a joint
def jcalc_transform(type: int, axis: df.float3, joint_q: df.tensor(float), start: int):

    # prismatic
    if (type == 0):

        q = df.load(joint_q, start)
        X_jc = spatial_transform(axis * q, quat_identity())
        return X_jc

    # revolute
    if (type == 1):

        q = df.load(joint_q, start)
        X_jc = spatial_transform(float3(0.0, 0.0, 0.0), quat_from_axis_angle(axis, q))
        return X_jc

    # fixed
    if (type == 2):

        X_jc = spatial_transform_identity()
        return X_jc

    # free
    if (type == 3):

        px = df.load(joint_q, start + 0)
        py = df.load(joint_q, start + 1)
        pz = df.load(joint_q, start + 2)

        qx = df.load(joint_q, start + 3)
        qy = df.load(joint_q, start + 4)
        qz = df.load(joint_q, start + 5)
        qw = df.load(joint_q, start + 6)

        X_jc = spatial_transform(float3(px, py, pz), quat(qx, qy, qz, qw))
        return X_jc


# compute motion subspace and velocity for a joint
def jcalc_motion(type: int, axis: df.float3, X_sc: df.spatial_transform, joint_S_s: df.tensor(df.spatial_vector), joint_qd: df.tensor(float), joint_start: int):

    # prismatic
    if (type == 0):

        S_s = df.spatial_transform_twist(X_sc, spatial_vector(float3(0.0, 0.0, 0.0), axis))
        v_j_s = S_s * df.load(joint_qd, joint_start)

        df.store(joint_S_s, joint_start, S_s)
        return v_j_s

    # revolute
    if (type == 1):

        S_s = df.spatial_transform_twist(X_sc, spatial_vector(axis, float3(0.0, 0.0, 0.0)))
        v_j_s = S_s * df.load(joint_qd, joint_start)

        df.store(joint_S_s, joint_start, S_s)
        return v_j_s

    # fixed
    if (type == 2):
        return spatial_vector()

    # free
    if (type == 3):

        v_j_c = spatial_vector(df.load(joint_qd, joint_start + 0),
                               df.load(joint_qd, joint_start + 1),
                               df.load(joint_qd, joint_start + 2),
                               df.load(joint_qd, joint_start + 3),
                               df.load(joint_qd, joint_start + 4),
                               df.load(joint_qd, joint_start + 5))

        v_j_s = spatial_transform_twist(X_sc, v_j_c)

        # write motion subspace
        df.store(joint_S_s, joint_start + 0, spatial_transform_twist(X_sc, spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        df.store(joint_S_s, joint_start + 1, spatial_transform_twist(X_sc, spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)))
        df.store(joint_S_s, joint_start + 2, spatial_transform_twist(X_sc, spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)))
        df.store(joint_S_s, joint_start + 3, spatial_transform_twist(X_sc, spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0)))
        df.store(joint_S_s, joint_start + 4, spatial_transform_twist(X_sc, spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)))
        df.store(joint_S_s, joint_start + 5, spatial_transform_twist(X_sc, spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)))

        return v_j_s


# # compute the velocity across a joint
# #@df.func
# def jcalc_velocity(self, type, S_s, joint_qd, start):

#     # prismatic
#     if (type == 0):
#         v_j_s = df.load(S_s, start)*df.load(joint_qd, start)
#         return v_j_s

#     # revolute
#     if (type == 1):
#         v_j_s = df.load(S_s, start)*df.load(joint_qd, start)
#         return v_j_s

#     # fixed
#     if (type == 2):
#         v_j_s = spatial_vector()
#         return v_j_s

#     # free
#     if (type == 3):
#         v_j_s =  S_s[start+0]*joint_qd[start+0]
#         v_j_s += S_s[start+1]*joint_qd[start+1]
#         v_j_s += S_s[start+2]*joint_qd[start+2]
#         v_j_s += S_s[start+3]*joint_qd[start+3]
#         v_j_s += S_s[start+4]*joint_qd[start+4]
#         v_j_s += S_s[start+5]*joint_qd[start+5]
#         return v_j_s


# computes joint space forces/torques in tau
def jcalc_tau(type: int, joint_S_s: df.tensor(spatial_vector), joint_start: int, body_f_s: spatial_vector, tau: df.tensor(float)):

    # prismatic / revolute
    if (type == 0 or type == 1):
        S_s = df.load(joint_S_s, joint_start)
        df.store(tau, joint_start, spatial_dot(S_s, body_f_s))

    # free
    if (type == 3):
            
        for i in range(0, 6):
            S_s = df.load(joint_S_s, joint_start+i)
            df.store(tau, joint_start+i, spatial_dot(S_s, body_f_s))

    return 0


def jcalc_integrate(type: int, joint_q: df.tensor(float), joint_qd: df.tensor(float), joint_qdd: df.tensor(float), coord_start: int, dof_start: int, dt: float):

    # prismatic / revolute
    if (type == 0 or type == 1):

        qdd = df.load(joint_qdd, dof_start)
        qd = df.load(joint_qd, dof_start)
        q = df.load(joint_q, coord_start)

        df.store(joint_qd, dof_start, qd + qdd*dt)
        df.store(joint_qd, coord_start, q + qd*dt)

    # free
    if (type == 3):

        # linear part
        for i in range(0, 3):
            
            qdd = df.load(joint_qdd, dof_start+3+i)
            qd = df.load(joint_qd, dof_start+3+i)
            q = df.load(joint_q, coord_start+i)

            df.store(joint_qd, dof_start+3+i, qd + qdd*dt)
            df.store(joint_q, coord_start+i, q + qd*dt)

        # angular part
        
        w = float3(df.load(joint_qd, dof_start + 0),
                   df.load(joint_qd, dof_start + 1),
                   df.load(joint_qd, dof_start + 2))

        # # quat and quat derivative
        # r = quat(
            
        #     q[q_start + 3], q[q_start + 4], q[q_start + 5], q[q_start + 6])
        # drdt = quat_multiply((*w, 0.0), r) * 0.5

        # # new orientation (normalized)
        # r_new = normalize(r + drdt * dt)

        # q[q_start + 3] = r_new[0]
        # q[q_start + 4] = r_new[1]
        # q[q_start + 5] = r_new[2]
        # q[q_start + 6] = r_new[3]

    return 0

def compute_link_transform(i: int,
                           joint_type: df.tensor(int),
                           joint_parent: df.tensor(int),
                           joint_q_start: df.tensor(int),
                           joint_qd_start: df.tensor(int),
                           joint_q: df.tensor(float),
                           joint_X_pj: df.tensor(df.spatial_transform),
                           joint_X_cm: df.tensor(df.spatial_transform),
                           joint_axis: df.tensor(df.float3),
                           joint_S_s: df.tensor(df.spatial_vector),
                           body_X_sc: df.tensor(df.spatial_transform),
                           body_X_sm: df.tensor(df.spatial_transform)):

    # parent transform
    parent = load(joint_parent, i)

    # parent transform in spatial coordinates
    X_sp = spatial_transform_identity()
    if (parent >= 0):
        X_sp = load(body_X_sc, parent)

    type = load(joint_type, i)
    axis = load(joint_axis, i)
    coord_start = load(joint_q_start, i)
    dof_start = load(joint_qd_start, i)

    # compute transform across joint
    #X_jc = spatial_jcalc(type, joint_q, axis, coord_start)
    X_jc = jcalc_transform(type, axis, joint_q, coord_start)

    X_pj = load(joint_X_pj, i)
    X_sc = spatial_transform_multiply(X_sp, spatial_transform_multiply(X_pj, X_jc))

    # compute transform of center of mass
    X_cm = load(joint_X_cm, i)
    X_sm = spatial_transform_multiply(X_sc, X_cm)

    # compute motion subspace in space frame (J)
    #self.joint_S_s[i] = transform_twist(X_sc, S_c)

    # store geometry transforms
    store(body_X_sc, i, X_sc)
    store(body_X_sm, i, X_sm)

    return 0


def eval_rigid_fk(articulation_start: df.tensor(int),
                  articulation_end: df.tensor(int),
                  joint_type: df.tensor(int),
                  joint_parent: df.tensor(int),
                  joint_q_start: df.tensor(int),
                  joint_qd_start: df.tensor(int),
                  joint_q: df.tensor(float),
                  joint_X_pj: df.tensor(df.spatial_transform),
                  joint_X_cm: df.tensor(df.spatial_transform),
                  joint_axis: df.tensor(df.float3),
                  joint_S_s: df.tensor(df.spatial_vector),
                  body_X_sc: df.tensor(df.spatial_transform),
                  body_X_sm: df.tensor(df.spatial_transform)):

    # one thread per-articulation
    index = tid()

    start = df.load(articulation_start, index)
    end = df.load(articulation_end, index)

    for i in range(start, end):
        compute_link_transform(i,
                               joint_type,
                               joint_parent,
                               joint_q_start,
                               joint_qd_start,
                               joint_q,
                               joint_X_pj,
                               joint_X_cm,
                               joint_axis,
                               joint_S_s,
                               body_X_sc,
                               body_X_sm)




#@df.func
def compute_link_velocity(i: int,
                          joint_type: df.tensor(int),
                          joint_parent: df.tensor(int),
                          joint_qd_start: df.tensor(int),
                          joint_qd: df.tensor(float),
                          joint_X_pj: df.tensor(df.spatial_transform),
                          joint_X_cm: df.tensor(df.spatial_transform),
                          joint_axis: df.tensor(df.float3),
                          body_I_m: df.tensor(df.spatial_matrix),
                          body_X_sc: df.tensor(df.spatial_transform),
                          body_X_sm: df.tensor(df.spatial_transform),
                          gravity: df.tensor(df.float3),
                          # outputs
                          joint_S_s: df.tensor(df.spatial_vector),
                          body_I_s: df.tensor(df.spatial_matrix),
                          body_v_s: df.tensor(df.spatial_vector),
                          body_f_s: df.tensor(df.spatial_vector),
                          body_a_s: df.tensor(df.spatial_vector)):

    type = df.load(joint_type, i)
    axis = df.load(joint_axis, i)
    dof_start = df.load(joint_qd_start, i)

    X_sc = df.load(body_X_sc, i)

    # compute motion subspace and velocity across the joint (stores S_s to global memory)
    v_j_s = jcalc_motion(type, axis, X_sc, joint_S_s, joint_qd, dof_start)

    # parent velocity
    parent = df.load(joint_parent, i)

    v_parent_s = spatial_vector()
    a_parent_s = spatial_vector()

    if (parent >= 0):
        v_parent_s = df.load(body_v_s, parent)
        a_parent_s = df.load(body_a_s, parent)

    # body velocity, acceleration
    v_s = v_parent_s + v_j_s
    a_s = a_parent_s + spatial_cross(v_s, v_j_s) # + self.joint_S_s[i]*self.joint_qdd[i]

    # compute body forces
    X_sm = df.load(body_X_sm, i)
    I_m = df.load(body_I_m, i)

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    g = df.load(gravity, 0)

    m = I_m[3, 3]
    f_ext_m = spatial_vector(float3(), g) * m
    f_ext_s = f_ext_m # todo: spatial_transform_wrench(X_sm, f_ext_m)

    # body forces
    I_s = spatial_transform_inertia(X_sm, I_m)

    f_b_s = df.mul(I_s, a_s) + spatial_cross_dual(v_s, df.mul(I_s, v_s))

    df.store(body_v_s, i, v_s)
    df.store(body_a_s, i, a_s)
    df.store(body_f_s, i, f_b_s - f_ext_s)
    df.store(body_I_s, i, I_s)

    return 0


#@df.func
def compute_link_tau(i: int,
                     joint_type: df.tensor(int),
                     joint_parent: df.tensor(int),
                     joint_q_start: df.tensor(int),
                     joint_qd_start: df.tensor(int),
                     joint_S_s: df.tensor(df.spatial_vector),
                     body_f_s: df.tensor(df.spatial_vector),
                     body_f_subtree_s: df.tensor(df.spatial_vector),
                     tau: df.tensor(float)):

    type = df.load(joint_type, i)
    parent = df.load(joint_parent, i)
    dof_start = df.load(joint_qd_start, i)

    f_s = df.load(body_f_s, i)                   # external forces on body
    f_child = df.load(body_f_subtree_s, i)       # forces acting on subtree of body

    f_tot = f_s + f_child

    # compute joint-space forces
    jcalc_tau(type, joint_S_s, dof_start, f_s, tau)

    # update parent forces
    df.atomic_add(body_f_subtree_s, parent, f_tot)

    return 0


#@df.kernel
def eval_rigid_id(joint_type: df.tensor(int),
                  joint_parent: df.tensor(int),
                  joint_q_start: df.tensor(int),
                  joint_qd_start: df.tensor(int),
                  joint_q: df.tensor(float),
                  joint_X_pj: df.tensor(df.spatial_transform),
                  joint_X_cm: df.tensor(df.spatial_transform),
                  joint_axis: df.tensor(df.float3),
                  joint_S_s: df.tensor(df.spatial_vector),
                  body_X_sc: df.tensor(df.spatial_transform),
                  body_X_sm: df.tensor(df.spatial_transform)):

    # one thread per-articulation
    index = tid()

    start = df.load(articulation_start, index)
    end = df.load(articulation_end, index)

    for i in range(0, count):
        compute_link_velocity(i,
                              joint_type,
                              joint_parent,
                              joint_q_start,
                              joint_qd_start,
                              joint_q,
                              joint_X_pj,
                              joint_X_cm,
                              joint_axis,
                              joint_S_s,
                              body_X_sc,
                              body_X_sm)

    for i in range(count - 1, -1):
        compute_link_tau(
            i,
            joint_type,
            joint_parent,
            joint_q_start,
            joint_qd_start,
        )




class SemiImplicitIntegrator:

    def __init__(self):
        pass

    def simulate(self, tape, model, state_in, state_out, dt):
        # if config.use_taichi:
        #     from dflex.taichi_sim import TaichiSemiImplicitIntegrator
        #     return TaichiSemiImplicitIntegrator.forward(model, state_in, state_out, dt)

        with util.ScopedTimer("simulate", False):

            # alloc particle force buffer
            if (model.particle_count):
                f_particle = jnp.zeros_like(state_in.u)

            # alloc rigid force buffer
            if (model.rigid_count):
                f_rigid = jnp.zeros_like(state_in.rigid_v)
                t_rigid = jnp.zeros_like(state_in.rigid_w)

            # damped springs
            if (model.spring_count):
                f_particle = f_particle + eval_springs(
                    state_in.q, state_in.u,
                    model.spring_indices, model.spring_rest_length,
                    model.spring_stiffness, model.spring_damping,
                )

            # triangle elastic and lift/drag forces
            if (model.tri_count and model.tri_ke > 0.0):
                f_particle = f_particle + eval_triangles(
                    state_in.q, state_in.u,
                    model.tri_indices, model.tri_poses, model.tri_activations,
                    model.tri_ke, model.tri_ka, model.tri_kd,
                    model.tri_drag, model.tri_lift,
                )

            # triangle/triangle self-contacts
            if (model.tri_collisions and model.tri_count and model.tri_ke > 0.0):
                f_particle = f_particle + eval_triangles_contact(
                    state_in.q, state_in.u,
                    model.tri_indices,
                )

            # triangle bending
            if (model.edge_count):
                f_particle = f_particle + eval_bending(
                    state_in.q, state_in.u,
                    model.edge_indices, model.edge_rest_angle,
                    model.edge_ke, model.edge_kd,
                )

            # ground contact (particles)
            if (model.ground):
                f_particle = f_particle + eval_contacts(
                    state_in.q, state_in.u,
                    model.contact_ke, model.contact_kd,
                    model.contact_kf, model.contact_mu,
                )

            # tetrahedral FEM
            if (model.tet_count):
                f_particle = f_particle + eval_tetrahedra(
                    state_in.q, state_in.u,
                    model.tet_indices, model.tet_poses,
                    model.tet_activations, model.tet_materials,
                )

            #----------------------------
            # rigid forces

            if (model.contact_count):
                df_rigid, dt_rigid = eval_rigid_contacts(
                    state_in.rigid_x, state_in.rigid_r,
                    state_in.rigid_v, state_in.rigid_w,
                    model.contact_body0, model.contact_point0,
                    model.contact_dist, model.contact_material,
                    model.shape_materials,
                )
                f_rigid = f_rigid + df_rigid
                t_rigid = t_rigid + dt_rigid

                if model.tri_collisions:
                    f_particle = f_particle + eval_triangles_rigid_contacts(
                        state_in.q, state_in.u,
                        model.tri_indices,
                        state_in.rigid_x, state_in.rigid_r,
                        state_in.rigid_v, state_in.rigid_w,
                        model.contact_body0, model.contact_point0,
                        model.contact_dist, model.contact_material,
                        model.shape_materials,
                    )

            #----------------------------
            # integrate

            if (model.particle_count):
                state_out.q, state_out.u = integrate_particles(
                    state_in.q, state_in.u, f_particle,
                    model.particle_inv_mass, model.gravity, dt,
                )

            if (model.rigid_count):
                (state_out.rigid_x, state_out.rigid_r,
                 state_out.rigid_v, state_out.rigid_w) = integrate_rigids(
                    state_in.rigid_x, state_in.rigid_r,
                    state_in.rigid_v, state_in.rigid_w,
                    f_rigid, t_rigid,
                    model.rigid_inv_mass, model.rigid_inv_inertia,
                    model.gravity, dt,
                )

            return state_out


    def forward(self, model, state_in, dt):
        # Allocate output state and run simulation step
        state_out = state_in.clone()
        self.simulate(_NoopTape(), model, state_in, state_out, dt)
        return state_out
            
