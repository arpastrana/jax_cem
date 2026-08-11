import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix

from jax_cem.datastructures.sequences import build_sequences
from jax_cem.datastructures.trails import build_trails

# ------------------------------------------------------------------------------
# Connectivity
# ------------------------------------------------------------------------------


def connectivity_matrix(
    edges: Int[np.ndarray, "edges 2"],
    num_nodes: int | None = None,
) -> csc_matrix:
    """
    Build the signed edge-node incidence matrix from indexed edges.

    Parameters
    ----------
    edges :
        The node index pair of each edge.
    num_nodes :
        The total number of nodes, fixing the column count. When omitted, it is
        inferred as the largest node index plus one, which undercounts columns
        if the highest-indexed node touches no edge.

    Returns
    -------
    connectivity :
        The incidence matrix in sparse format, one row per edge, with ``-1``
        in the start node's column and ``+1`` in the end node's column.
    """
    # Iterating a JAX array element-wise costs one device sync per element;
    # convert to NumPy once and slice columns vectorized.
    edges_np = np.asarray(edges)
    m = len(edges_np)
    data = np.concatenate((-np.ones(m), np.ones(m)))
    rows = np.concatenate((np.arange(m), np.arange(m)))
    cols = np.concatenate((edges_np[:, 0], edges_np[:, 1]))

    n = num_nodes if num_nodes is not None else int(np.max(edges_np)) + 1
    shape = (m, n)

    # coo_matrix.tocsc() yields a csc_matrix at runtime; scipy's bundled stubs
    # widen the return to csc_array
    return coo_matrix((data, (rows, cols)), shape=shape).tocsc()  # pyright: ignore[reportReturnType]


# ------------------------------------------------------------------------------
# Structure
# ------------------------------------------------------------------------------


class Structure(eqx.Module):
    pass


class EquilibriumStructure(Structure):
    """
    The attributed, undirected graph describing a pin-jointed bar structure.

    Notes
    -----
    Trail and deviation edges are held apart rather than as one edge array and a
    mask, since an edge is one or the other and never both. `edges` concatenates
    them, trail edges first, and that order is the one every derived index and
    every per-edge parameter follows.

    The trail search runs in the constructor, so a structure is either laid out
    into sequences or it does not exist.
    """

    nodes: Int[Array, "nodes"]
    supports: Int[Array, "nodes_fixed"]
    edges_trail: Int[Array, "edges_trail 2"]
    edges_deviation: Int[Array, "edges_deviation 2"]

    sequences: Int[Array, "sequences trails"]
    origin_nodes: Int[Array, "trails"]
    sequences_edges: Int[Array, "sequences_edges trails"]
    sequences_edges_indices: Int[Array, "edges_trail"]
    edges_deviation_direct: Float[Array, "edges"]
    connectivity: Float[Array, "edges nodes"]

    def __init__(
        self,
        nodes: Int[np.ndarray, "nodes"],
        supports: Int[np.ndarray, "nodes_fixed"],
        edges_trail: Int[np.ndarray, "edges_trail 2"],
        edges_deviation: Int[np.ndarray, "edges_deviation 2"],
    ):
        nodes = np.asarray(nodes)
        supports = np.asarray(supports)
        edges_trail = np.asarray(edges_trail)
        edges_deviation = np.asarray(edges_deviation)

        if edges_trail.ndim != 2 or edges_deviation.ndim != 2:
            raise ValueError("Edges must be given as an array of node key pairs")

        edges = np.concatenate((edges_trail, edges_deviation))

        trails = build_trails(nodes, supports, edges_trail)
        data = build_sequences(trails, edges, len(nodes), len(edges_trail))

        self.nodes = jnp.asarray(nodes)
        self.supports = jnp.asarray(supports)
        self.edges_trail = jnp.asarray(edges_trail)
        self.edges_deviation = jnp.asarray(edges_deviation)

        self.sequences = data.sequences
        self.origin_nodes = data.origin_nodes
        self.sequences_edges = data.sequences_edges
        self.sequences_edges_indices = data.sequences_edges_indices
        self.edges_deviation_direct = data.edges_deviation_direct

        self.connectivity = jnp.asarray(
            connectivity_matrix(edges, len(nodes)).toarray(),
        )

    def __check_init__(self):
        """
        Reject a structure that the equilibrium computation cannot step through.
        """
        edges = np.asarray(self.edges)
        loops = np.flatnonzero(edges[:, 0] == edges[:, 1])
        if loops.size > 0:
            raise ValueError(f"Edges {loops.tolist()} are self-loops")

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
    def node_index(self) -> dict[int, int]:
        """
        A dictionary between node keys and their enumeration indices.
        """
        nodes = np.asarray(self.nodes)

        return {int(node): index for index, node in enumerate(nodes)}

    @property
    def edge_index(self) -> dict[tuple[int, int], int]:
        """
        A dictionary between edge keys and their enumeration indices.
        """
        edges = np.asarray(self.edges)

        return {(int(u), int(v)): index for index, (u, v) in enumerate(edges)}

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
        return self.sequences.shape[-1]

    @property
    def num_sequences(self) -> int:
        """
        The number of sequences.
        """
        return self.sequences.shape[-2]
