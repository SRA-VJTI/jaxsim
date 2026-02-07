import jax.numpy as jnp


def normalize_vertices(vertices, mean_subtraction=True, scale_factor=1.0):
    # vertices: N x 3
    if mean_subtraction:
        vertices = vertices - jnp.mean(vertices, axis=-2, keepdims=True)
    dists = jnp.linalg.norm(vertices, axis=-1)
    dist_max = jnp.max(dists, axis=-1)
    vertices = scale_factor * (vertices / dist_max[..., None, None])
    return vertices
