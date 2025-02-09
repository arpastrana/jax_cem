import jax
import jax.numpy as jnp


def vector_length(v: jax.Array, keepdims: bool = True) -> jax.Array:
    """
    Calculate the length of a vector over its last dimension.
    """
    v = jnp.nan_to_num(v)
    is_zero_vector = jnp.allclose(v, 0.0)
    d = jnp.where(is_zero_vector, jnp.ones_like(v), v)  # replace d with ones if is_zero

    length = jnp.where(is_zero_vector, 0.0, jnp.linalg.norm(d, axis=-1, keepdims=keepdims))

    return length


def vector_normalized(u: jax.Array) -> jax.Array:
    """
    Scale a vector such that it has a unit length.
    """
    is_zero = jnp.allclose(u, 0.0)
    d = jnp.where(is_zero, jnp.ones_like(u), u)  # replace d with ones if is_zero

    return jnp.where(is_zero, u, u / vector_length(d))  # replace normalized vector with vector if is_zero
