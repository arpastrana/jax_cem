from typing import NamedTuple

import jax


class EquilibriumState(NamedTuple):
    """
    The equilibrium state of a structure.
    """
    xyz: jax.Array  # N x 3
    loads: jax.Array  # N x 3
    reactions: jax.Array  # N x 3
    lengths: jax.Array  # M x 1
    forces: jax.Array  # M x 1


class EquilibriumSequenceState(NamedTuple):
    """
    The equilibrium state of a sequence in a structure.
    """
    xyz: jax.Array  # S x 3
    residuals: jax.Array  # S x 3
    lengths: jax.Array  # S x 1
