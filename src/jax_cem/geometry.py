import jax
import jax.numpy as jnp


def vector_length(v: jax.Array, keepdims: bool = True) -> jax.Array:
    """
    Calculate the length of a vector over its last dimension.
    """
    return jnp.linalg.norm(v, axis=-1, keepdims=keepdims)


def vector_normalized(u: jax.Array) -> jax.Array:
    """
    Scale a vector such that it has a unit length.
    """
    is_zero = jnp.allclose(u, 0.0)
    d = jnp.where(is_zero, jnp.ones_like(u), u)  # replace d with ones if is_zero

    return jnp.where(is_zero, u, u / vector_length(d))  # replace normalized vector with vector if is_zero
