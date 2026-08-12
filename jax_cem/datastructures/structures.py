import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Int

from jax_cem.datastructures.trails import Trails
from jax_cem.datastructures.trails import build_trails
from jax_cem.datastructures.trails import search_trails

__all__ = [
    "Structure",
]

# ------------------------------------------------------------------------------
# Structure
# ------------------------------------------------------------------------------


class Structure(eqx.Module):
    """
    The attributed, undirected graph describing a pin-jointed bar structure.

    Notes
    -----
    Trail and deviation edges are held apart rather than as one edge array and a
    mask, since an edge is one or the other and never both. `edges` concatenates
    them, trail edges first, and that order is the one every derived index and
    every per-edge parameter follows.

    The trail search runs in the constructor, so a structure is either laid out
    into trails and sequences or it does not exist.
    """

    nodes: Int[Array, "nodes"]
    supports: Int[Array, "nodes_fixed"]
    edges_trail: Int[Array, "edges_trail 2"]
    edges_deviation: Int[Array, "edges_deviation 2"]

    trails: Trails

    def __init__(
        self,
        nodes: Int[np.ndarray, "nodes"],
        supports: Int[np.ndarray, "nodes_fixed"],
        edges_trail: Int[np.ndarray, "edges_trail 2"],
        edges_deviation: Int[np.ndarray, "edges_deviation 2"],
    ):
        if edges_trail.ndim != 2 or edges_deviation.ndim != 2:
            raise ValueError("Edges must be given as an array of node key pairs")

        trail_nodes = search_trails(nodes, supports, edges_trail)

        self.nodes = jnp.asarray(nodes)
        self.supports = jnp.asarray(supports)
        self.edges_trail = jnp.asarray(edges_trail)
        self.edges_deviation = jnp.asarray(edges_deviation)
        self.trails = build_trails(trail_nodes, edges_trail)

    def __check_init__(self):
        """
        Reject a structure that the equilibrium computation cannot step through.
        """
        edges = np.asarray(self.edges)
        loops = np.flatnonzero(edges[:, 0] == edges[:, 1])
        if loops.size > 0:
            raise ValueError(f"Edges {loops.tolist()} are self-loops")

        # a negative key would otherwise wrap, and the scatter would drop the edge
        if np.any((edges < 0) | (edges >= self.num_nodes)):
            raise ValueError("Edges must index existing nodes")

        # a repeated pair collapses in the edge index, which is keyed by it
        pairs, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        repeated = pairs[counts > 1]
        if repeated.size > 0:
            raise ValueError(
                f"Node pairs {repeated.tolist()} carry more than one edge; two "
                f"nodes are joined once, by a trail edge or by a deviation edge",
            )

        if self.supports.shape[-1] != self.num_trails:
            raise ValueError(
                f"Got {self.supports.shape[-1]} supports for {self.num_trails} "
                f"trails; every trail ends at exactly one support",
            )

        if np.any((self.supports < 0) | (self.supports >= len(self.nodes))):
            raise ValueError("Supports must index existing nodes")

    # --------------------------------------------------------------------------
    # Derived views
    # --------------------------------------------------------------------------

    @property
    def edges(self) -> Int[Array, "edges 2"]:
        """
        The node key pair of each edge, trail edges first.
        """
        return jnp.concatenate((self.edges_trail, self.edges_deviation), axis=-2)

    @property
    def origin_nodes(self) -> Int[Array, "trails"]:
        """
        The first node of each trail, as the layout states it.
        """
        return self.trails.origin_nodes

    @property
    def node_index(self) -> dict[int, int]:
        """
        A dictionary between node keys and their enumeration indices.
        """
        nodes = np.asarray(self.nodes)
        node_index = {int(node): index for index, node in enumerate(nodes)}

        return node_index

    @property
    def edge_index(self) -> dict[tuple[int, int], int]:
        """
        A dictionary between edge keys and their enumeration indices.
        """
        edges = np.asarray(self.edges)
        edge_index = {(int(u), int(v)): index for index, (u, v) in enumerate(edges)}

        return edge_index

    # --------------------------------------------------------------------------
    # Counts
    # --------------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """
        The number of nodes.
        """
        return self.nodes.shape[-1]

    @property
    def num_edges(self) -> int:
        """
        The number of edges.
        """
        return self.num_edges_trail + self.num_edges_deviation

    @property
    def num_edges_trail(self) -> int:
        """
        The number of trail edges.
        """
        return self.edges_trail.shape[-2]

    @property
    def num_edges_deviation(self) -> int:
        """
        The number of deviation edges.
        """
        return self.edges_deviation.shape[-2]

    @property
    def num_trails(self) -> int:
        """
        The number of trails.
        """
        return self.trails.sequences.nodes.shape[-1]

    @property
    def num_sequences(self) -> int:
        """
        The number of sequences.
        """
        return self.trails.sequences.nodes.shape[-2]
