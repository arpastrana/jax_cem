from typing import NamedTuple

import jax

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------


class ParameterState(NamedTuple):
    """
    The parameters of an equilibrium model.
    """

    xyz: jax.Array  # N x 3
    loads: jax.Array  # N x 3
    forces: jax.Array  # M x 1
    # TODO: find a way to treat edge lengths and planes edgewise, not nodewise
    lengths: jax.Array  # N x 1
    planes: jax.Array  # N x 6
