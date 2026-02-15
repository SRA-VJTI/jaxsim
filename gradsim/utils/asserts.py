import jax.numpy as jnp


def assert_array(var, varname):
    r"""Assert that the variable is a JAX array (jnp.ndarray)."""
    if not isinstance(var, jnp.ndarray):
        raise TypeError(
            f"Expected {varname} of type jnp.ndarray. Got {type(var)} instead."
        )
