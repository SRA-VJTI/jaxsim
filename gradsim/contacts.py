import jax.numpy as jnp

from .utils.defaults import Defaults


def detect_ground_plane_contacts(vertices_world, eps=Defaults.EPSILON):
    """Detect contact points (vertices) with the ground-plane, given a set of
    vertices (usually --- belonging to a single mesh).

    Args:
        vertices_world (jnp.ndarray): Set of vertices whose collisions with
            the ground plane (assumed to be the XZ-plane) are to be detected.
        eps (float): Contact detection threshold (i.e., distance below which
            two bodies will be considered penetrating).

    Returns:
        contact_inds (jnp.ndarray): Indices of contact vertices.
        contact_points (jnp.ndarray): Positions of contact vertices.
        contact_normals (jnp.ndarray): Normals of contact (i.e., ground-plane
            normals here).
    """
    if eps < 0:
        raise ValueError(f"eps cannot be negative! Got: {eps}")
    mask = (vertices_world[:, 1] < eps)
    contact_inds = jnp.where(mask)[0]
    contact_points, contact_normals = None, None
    if contact_inds.size > 0:
        contact_points = vertices_world[contact_inds]
        contact_normals = jnp.tile(
            jnp.array([0.0, 1.0, 0.0]), (contact_inds.shape[0], 1)
        )
    else:
        contact_inds = None

    return contact_inds, contact_points, contact_normals
