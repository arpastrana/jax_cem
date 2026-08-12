"""
Indexing an array whose indices carry a padded slot.

An index past the end of an array is dropped by a scatter and filled by a gather.
A negative index is not past the end: it wraps onto the last entry before either
applies, so padding marked that way has to be sent beyond the end to be seen as
absent rather than read as the last row.
"""

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

__all__ = [
    "gather_padded",
    "indices_beyond",
]


def indices_beyond(
    indices: Int[Array, "*slots"],
    size: int,
) -> Int[Array, "*slots"]:
    """
    Send the padded entries of an index array past the end of what it addresses.

    Parameters
    ----------
    indices :
        Indices into an array of ``size`` entries, where ``-1`` marks padding.
    size :
        The number of entries the indices address.

    Returns
    -------
    indices :
        The same indices, with every padded entry replaced by ``size``.

    Notes
    -----
    A negative index is valid and wraps onto the last entry, so padding marked
    that way reads a real row, and whatever it reads has to be masked out again
    afterwards. An index one past the end is out of bounds instead, which a
    scatter drops and a gather fills, so a padded slot reads and writes nothing.
    """
    return jnp.where(indices < 0, size, indices)


def gather_padded(
    values: Float[Array, "entries *columns"],
    indices: Int[Array, "slots"],
) -> Float[Array, "slots *columns"]:
    """
    Read one entry of an array per slot, taking zero where a slot is padded.

    Parameters
    ----------
    values :
        The array to read from.
    indices :
        One index into it per slot, where ``-1`` marks a padded slot.

    Returns
    -------
    gathered :
        The entry each slot addresses, and zero for the slots that address none.

    Notes
    -----
    The padded slots are sent out of bounds first, so the fill applies to them
    and to nothing else. Reading zero is what lets a padded slot fall out of the
    arithmetic it feeds rather than having to be masked out of the result.
    """
    slots = indices_beyond(indices, values.shape[0])
    gathered = values.at[slots].get(mode="fill", fill_value=0.0)

    return gathered
