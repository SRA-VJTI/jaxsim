import math
import numpy as np
import jax.numpy as jnp

from gradsim.utils import quaternion


def test_normalize():
    rng = np.random.default_rng(0)
    quat = jnp.array(rng.standard_normal(4), dtype=jnp.float32)
    norm = float(jnp.linalg.norm(quaternion.normalize(quat)))
    assert abs(norm - 1.0) < 1e-4


def test_quaternion_to_rotmat():
    # Rotation of pi radians about Y-axis.
    axis = jnp.array([0.0, 1.0, 0.0])
    halfangle = jnp.array(math.pi / 2)
    cos = jnp.cos(halfangle).reshape(1)
    sin = jnp.sin(halfangle)
    quat = jnp.concatenate([cos, sin * axis])
    rotmat = jnp.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    assert jnp.allclose(quaternion.quaternion_to_rotmat(quat), rotmat, atol=1e-5)


def test_quaternion_multiply():
    # pi about Y + (-pi) about Y → identity.
    axis = jnp.array([0.0, 1.0, 0.0])
    halfangle = jnp.array(math.pi / 2)
    q1 = jnp.concatenate([jnp.cos(halfangle).reshape(1), jnp.sin(halfangle) * axis])
    q2 = jnp.concatenate([jnp.cos(-halfangle).reshape(1), jnp.sin(-halfangle) * axis])
    assert jnp.allclose(
        quaternion.multiply(q1, q2), jnp.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6
    )
