from typing import NamedTuple

from jaxtyping import Array
from jaxtyping import Float

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------


class ParameterState(NamedTuple):
    """
    The parameters of an equilibrium model.

    Attributes
    ----------
    xyz :
        The start position of every node, of which only the origin nodes are read.
    loads :
        The load vector applied at every node.
    forces :
        The force in every edge, of which only the deviation entries are read.
    lengths :
        The signed length of the trail edge outgoing from every node, where zero
        hands the length over to the plane of the node.
    planes :
        The origin and the normal of the plane that positions the next node on a
        trail, where a zero normal marks a node without one.
    """

    xyz: Float[Array, "nodes 3"]
    loads: Float[Array, "nodes 3"]
    forces: Float[Array, "edges 1"]
    # TODO: find a way to treat edge lengths and planes edgewise, not nodewise
    lengths: Float[Array, "nodes 1"]
    planes: Float[Array, "nodes 6"]
