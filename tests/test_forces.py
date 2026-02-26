import numpy as np
import jax.numpy as jnp

from gradsim.forces import ConstantForce, Gravity, XForce, YForce


def test_constantforce():
    rng = np.random.default_rng(42)
    direction = jnp.array(rng.standard_normal(3), dtype=jnp.float32)
    magnitude = 10.0
    force = ConstantForce(direction, magnitude, starttime=0.5, endtime=1.0)
    assert jnp.allclose(force.apply(0.1), jnp.zeros(3))
    assert jnp.allclose(force.apply(1.1), jnp.zeros(3))
    assert jnp.allclose(force.apply(0.5), direction * magnitude)
    assert jnp.allclose(force.apply(0.9), direction * magnitude)


def test_gravity():
    direction = jnp.array([0.0, 0.0, -1.0])
    force = Gravity()
    assert jnp.allclose(force.apply(0.1), direction * 10.0)
    assert jnp.allclose(force.apply(1.1), direction * 10.0)
    magnitude = 1.0
    force = Gravity(magnitude=magnitude)
    assert jnp.allclose(force.apply(0.5), direction * magnitude)
    assert jnp.allclose(force.apply(0.9), direction * magnitude)


def test_xforce():
    direction = jnp.array([1.0, 0.0, 0.0])
    force = XForce()
    assert jnp.allclose(force.apply(0.1), direction * 10.0)
    assert jnp.allclose(force.apply(1.1), direction * 10.0)
    magnitude = 1.0
    force = XForce(magnitude=magnitude)
    assert jnp.allclose(force.apply(0.5), direction * magnitude)
    assert jnp.allclose(force.apply(0.9), direction * magnitude)


def test_yforce():
    direction = jnp.array([0.0, 1.0, 0.0])
    force = YForce()
    assert jnp.allclose(force.apply(0.1), direction * 10.0)
    assert jnp.allclose(force.apply(1.1), direction * 10.0)
    magnitude = 1.0
    force = YForce(magnitude=magnitude)
    assert jnp.allclose(force.apply(0.5), direction * magnitude)
    assert jnp.allclose(force.apply(0.9), direction * magnitude)
