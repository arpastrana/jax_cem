import jax

from typing import NamedTuple


__all__ = ["EquilibriumState"]


class EquilibriumState(NamedTuple):
    xyz: jax.Array  # N x 3
    reactions: jax.Array  # N x 3
    lengths: jax.Array  # M x 1
    forces: jax.Array  # M x 1
    loads: jax.Array  # N x 3
