import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Bool
from jaxtyping import Float

TOLERANCE_NORMAL = 1e-8


def is_plane_absent(plane: Float[Array, "*planes 6"]) -> Bool[Array, "*planes"]:
    """
    Whether a plane has a normal too short to point anywhere.

    Parameters
    ----------
    plane :
        The origin and the normal of a plane.

    Returns
    -------
    is_absent :
        Whether the plane is one in name only.

    Notes
    -----
    The normal arrives unnormalized, so it is compared against zero within an
    absolute tolerance rather than exactly, and a plane meant to be read has to
    carry a normal clear of it. Every use of a plane asks this one question, so
    the branch that reads a plane and the guard inside the arithmetic that reads
    it cannot come to different answers.
    """
    normal = plane[..., 3:]
    is_short = jnp.abs(normal) <= TOLERANCE_NORMAL

    return jnp.all(is_short, axis=-1)


def vector_length(v: Float[Array, "3"]) -> Float[Array, ""]:
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

    length = jnp.where(is_zero_vector, 0.0, jnp.linalg.norm(d, axis=-1))

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
