from typing import NamedTuple

from jaxtyping import Array
from jaxtyping import Float

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------


class Parameters(NamedTuple):
    """
    The parameters of an equilibrium model.

    Attributes
    ----------
    xyz_origin :
        The position of the origin node of every trail, column-aligned with the
        sequences.
    loads :
        The load vector applied at every node.
    forces :
        The force in every deviation edge. A trail edge force is an output,
        recovered from the trail residual that passes through it.
    lengths :
        The signed length of every trail edge, where zero hands the length over
        to the plane of the edge.
    planes :
        The origin and the normal of the plane that positions the node at the far
        end of every trail edge, where a zero normal marks an edge without one.

    Notes
    -----
    The lengths and the planes are keyed by the trail edge they drive rather than
    by the node that edge leaves. Every trail ends at a support, where no trail
    edge leaves and neither of them is read, so a nodewise array would spend a row
    on each one.
    """

    xyz_origin: Float[Array, "trails 3"]
    loads: Float[Array, "nodes 3"]
    forces: Float[Array, "edges_deviation"]
    lengths: Float[Array, "edges_trail"]
    planes: Float[Array, "edges_trail 6"]
