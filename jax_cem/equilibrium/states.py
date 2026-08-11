from typing import NamedTuple

from jaxtyping import Array
from jaxtyping import Float


class EquilibriumState(NamedTuple):
    """
    The equilibrium state of a structure.

    Attributes
    ----------
    xyz :
        The position of every node.
    loads :
        The load vector applied at every node.
    reactions :
        The reaction force at every node, zero away from the supports.
    lengths :
        The length of every edge.
    forces :
        The force in every edge, positive in tension.
    """

    xyz: Float[Array, "nodes 3"]
    loads: Float[Array, "nodes 3"]
    reactions: Float[Array, "nodes 3"]
    lengths: Float[Array, "edges 1"]
    forces: Float[Array, "edges 1"]


class EquilibriumSequenceState(NamedTuple):
    """
    The equilibrium state of a sequence in a structure.

    Attributes
    ----------
    xyz :
        The position of the node of every trail in the sequence.
    residuals :
        The residual force at the node of every trail in the sequence.
    lengths :
        The signed length of the trail edge outgoing from every one of those
        nodes.
    """

    xyz: Float[Array, "trails 3"]
    residuals: Float[Array, "trails 3"]
    lengths: Float[Array, "trails"]
