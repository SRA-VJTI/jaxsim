import pytest
import jax.numpy as jnp

from gradsim.bodies import RigidBody


def test_assertions():
    pytest.raises(TypeError, RigidBody)


def test_create_body():
    cube_verts = jnp.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ],
        dtype=jnp.float32,
    )
    cube = RigidBody(cube_verts)
