import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float


def vector_length(v: Float[Array, "3"]) -> Float[Array, "1"]:
    """
    Calculate the length of a vector over its last dimension.

    Parameters
    ----------
    v :
        A vector.

    Returns
    -------
    length :
        The length of the vector, zero if the vector is zero.

    Notes
    -----
    A zero vector is replaced by a vector of ones before the norm is taken and
    the result discarded, which keeps the gradient at zero finite.
    """
    v = jnp.nan_to_num(v)
    is_zero_vector = jnp.allclose(v, 0.0)
    d = jnp.where(is_zero_vector, jnp.ones_like(v), v)

    length = jnp.where(
        is_zero_vector,
        0.0,
        jnp.linalg.norm(d, axis=-1, keepdims=True),
    )

    return length


def vector_normalized(u: Float[Array, "3"]) -> Float[Array, "3"]:
    """
    Scale a vector such that it has a unit length.

    Parameters
    ----------
    u :
        A vector.

    Returns
    -------
    vector :
        The unit vector, or the vector itself if it is zero.

    Notes
    -----
    A zero vector is divided by one instead of by its own length and the result
    discarded, which keeps the gradient at zero finite.
    """
    is_zero = jnp.allclose(u, 0.0)
    d = jnp.where(is_zero, jnp.ones_like(u), u)

    return jnp.where(is_zero, u, u / vector_length(d))
