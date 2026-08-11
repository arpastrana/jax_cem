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
    forces :
        The force in every edge, positive in tension.
    lengths :
        The length of every edge.
    vectors :
        The vector of every edge, from its tail node to its head node.
    loads :
        The load vector applied at every node.
    residuals :
        The residual force at every node, zero at free nodes in equilibrium.

    Notes
    -----
    The reaction at a supported node is the negation of its residual. It is
    deliberately not stored: an interface is the only place that reads a
    reaction, so keeping one quantity in one place removes the opportunity for
    the two to disagree.

    A residual is assembled from the edge forces and the loads, so a free node
    that is out of equilibrium reports it. The trail residual the sweep carries
    is a different quantity and is not stored, since the force in a trail edge
    and the vector of that edge already hold it.

    The field set follows the Formax contract. The container does not: Formax
    uses an equinox module where this is a named tuple.
    """

    xyz: Float[Array, "nodes 3"]
    forces: Float[Array, "edges"]
    lengths: Float[Array, "edges"]
    vectors: Float[Array, "edges 3"]
    loads: Float[Array, "nodes 3"]
    residuals: Float[Array, "nodes 3"]


class EquilibriumSequenceState(NamedTuple):
    """
    The equilibrium state of a sequence in a structure.

    Attributes
    ----------
    xyz :
        The position of the node of every trail in the sequence.
    residuals_trail :
        The force the outgoing trail edge carries at the node of every trail in
        the sequence.
    lengths :
        The signed length of the trail edge outgoing from every one of those
        nodes.

    Notes
    -----
    A trail residual is what the load and the deviation forces at a node leave
    for its outgoing trail edge to carry, so it does not vanish in equilibrium
    the way a nodal residual does.
    """

    xyz: Float[Array, "trails 3"]
    residuals_trail: Float[Array, "trails 3"]
    lengths: Float[Array, "trails"]
