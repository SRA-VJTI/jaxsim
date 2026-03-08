import jax.numpy as jnp


def normalize(quaternion):
    r"""Normalizes a quaternion to unit norm.

    Args:
        quaternion (jnp.ndarray): Quaternion to normalize (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).

    Returns:
        (jnp.ndarray): Normalized quaternion (shape: :math:`(4)`).
    """
    norm = jnp.linalg.norm(quaternion) + 1e-5
    return quaternion / norm


def quaternion_to_rotmat(quaternion):
    r"""Converts a quaternion to a :math:`3 \times 3` rotation matrix.

    Args:
        quaternion (jnp.ndarray): Quaternion to convert (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).

    Returns:
        (jnp.ndarray): rotation matrix (shape: :math:`(3, 3)`).
    """
    r = quaternion[0]
    i = quaternion[1]
    j = quaternion[2]
    k = quaternion[3]
    twoisq = 2 * i * i
    twojsq = 2 * j * j
    twoksq = 2 * k * k
    twoij = 2 * i * j
    twoik = 2 * i * k
    twojk = 2 * j * k
    twori = 2 * r * i
    tworj = 2 * r * j
    twork = 2 * r * k
    rotmat = jnp.array([
        [1 - twojsq - twoksq, twoij - twork,        twoik + tworj],
        [twoij + twork,        1 - twoisq - twoksq,  twojk - twori],
        [twoik - tworj,        twojk + twori,         1 - twoisq - twojsq],
    ])
    return rotmat


def multiply(q1, q2):
    r"""Multiply two quaternions `q1`, `q2`.

    Args:
        q1 (jnp.ndarray): First quaternion (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).
        q2 (jnp.ndarray): Second quaternion (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).

    Returns:
        (jnp.ndarray): Quaternion product (shape: :math:`(4)`)
            (Assumes (r, i, j, k) convention, with :math:`r` being the scalar).
    """
    r1 = q1[0]
    v1 = q1[1:]
    r2 = q2[0]
    v2 = q2[1:]
    return jnp.concatenate(
        [
            (r1 * r2 - jnp.dot(v1, v2)).reshape(1),
            r1 * v2 + r2 * v1 + jnp.cross(v1, v2),
        ],
        axis=0,
    )
