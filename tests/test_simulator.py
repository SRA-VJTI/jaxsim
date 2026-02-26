import jax.numpy as jnp

from gradsim.bodies import RigidBody
from gradsim.forces import Gravity
from gradsim.simulator import Simulator


CUBE_VERTS = jnp.array(
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


def test_smoke():
    sim = Simulator([])


def test_newtons_first_law_rest():
    # When no external force acts, bodies at rest remain at rest.
    cube = RigidBody(CUBE_VERTS)
    sim = Simulator([cube])
    sim.step()
    assert jnp.allclose(cube.position, jnp.zeros(3))
    sim.step()
    assert jnp.allclose(cube.position, jnp.zeros(3))


def test_newtons_first_law_motion():
    # When no external force acts, a body with constant velocity
    # continues to move with that velocity.
    cube = RigidBody(CUBE_VERTS)
    # Give the cube a linear momentum of (8, 8, 8) (its mass is 8),
    # so the velocity becomes 1 m/s. Frame rate = 30 Hz → 0.0333 m/frame.
    cube.linear_momentum = jnp.ones(3, dtype=jnp.float32) * 8
    sim = Simulator([cube])
    sim.step()
    assert jnp.allclose(cube.position, 0.0333 * jnp.ones(3), atol=1e-4)
    sim.step()
    assert jnp.allclose(cube.position, 0.0667 * jnp.ones(3), atol=1e-4)
    sim.step()
    assert jnp.allclose(cube.position, 0.1 * jnp.ones(3), atol=1e-4)


def test_cube_with_gravity():
    # Add gravity to the cube.
    cube = RigidBody(CUBE_VERTS)
    gravity = Gravity()
    direction = jnp.array([0.0, 0.0, -1.0])
    cube.add_external_force(gravity)
    sim = Simulator([cube])
    sim.step()
    assert jnp.allclose(cube.linear_velocity, 0.3333333 * direction, atol=1e-5)
    assert jnp.allclose(cube.position, 0.0111111 * direction, atol=1e-5)
    sim.step()
    assert jnp.allclose(cube.linear_velocity, 0.66666667 * direction, atol=1e-5)
    assert jnp.allclose(cube.position, 0.0333333 * direction, atol=1e-5)
    sim.step()
    assert jnp.allclose(cube.linear_velocity, 1.0 * direction, atol=1e-5)
    assert jnp.allclose(cube.position, 0.0666667 * direction, atol=1e-5)
