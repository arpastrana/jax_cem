from itertools import pairwise

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Int
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix

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
    """

    nodes: jax.Array  # nodes
    edges: jax.Array  # pairs of nodes
    origin_nodes: jax.Array  # nodes
    support_nodes: jax.Array  # nodes
    trail_edges: jax.Array  # indices in edges, or mask?
    deviation_edges: jax.Array  # indices in edges or mask?
    indirect_edges: jax.Array  # indices in edges or mask?
    sequences: jax.Array  # nodes verbatim
    node_index: dict[int, int]
    edge_index: dict[tuple[int, int], int]
    connectivity: jax.Array
    incidence: jax.Array
    sequences_edges: np.ndarray
    sequences_edges_indices: np.ndarray

    def __init__(
        self,
        nodes,
        edges,
        origin_nodes,
        support_nodes,
        trail_edges,
        deviation_edges,
        indirect_edges,
        sequences,
    ):
        self.nodes = nodes
        self.edges = edges
        self.origin_nodes = origin_nodes
        self.support_nodes = support_nodes
        self.trail_edges = trail_edges  # a boolean mask
        self.deviation_edges = deviation_edges  # a boolean mask
        self.indirect_edges = indirect_edges
        self.sequences = sequences

        self.node_index = {node: index for index, node in enumerate(self.nodes)}
        self.edge_index = {tuple(edge): index for index, edge in enumerate(self.edges)}
        self.connectivity = jnp.asarray(
            connectivity_matrix(self.edges, len(self.nodes)).toarray(),
        )
        self.incidence = self._incidence()
        self.sequences_edges = self._sequences_edges()
        self.sequences_edges_indices = self._sequences_edges_indices()

    def _incidence(self):
        incidence = np.zeros_like(self.connectivity)

        for node in self.nodes:
            edge_indices = np.nonzero(self.connectivity[:, node])
            connected_edges = self.edges[edge_indices]
            for i, edge in zip(np.reshape(edge_indices, (-1, 1)), connected_edges):
                val = 1.0
                if edge[0] != node:
                    val = -1.0
                incidence[i, node] = val

        return jnp.asarray(incidence)

    def _sequences_edges_indices(self):
        counts = []
        count = 0
        for sequence in self.sequences_edges:
            for idx in sequence:
                if idx >= 0:
                    counts.append(count)
                count += 1

        return np.asarray(counts).astype(int)

    def _sequences_edges(self):
        sequences = []
        for sequences_pair in pairwise(self.sequences):
            sequence = []
            for edge in zip(*sequences_pair):
                edge = tuple(edge)
                index = self.edge_index.get(
                    edge,
                    self.edge_index.get((edge[1], edge[0]), -1),
                )
                sequence.append(index)
            sequences.append(sequence)

        return np.asarray(sequences).astype(int)

    def number_of_nodes(self):
        """
        The number of nodes in the graph.
        """
        return len(self.nodes)

    def number_of_edges(self):
        """
        The number of edges in the graph.
        """
        return len(self.edges)

    def number_of_trail_edges(self):
        """
        The number of trail edges in the graph.
        """
        # return self.num_trail_edges
        return int(np.sum(self.trail_edges))

    def number_of_deviation_edges(self):
        """
        The number of deviation edges in the graph.
        """
        # return self.num_deviation_edges
        return int(np.sum(self.deviation_edges))

    def number_of_trails(self):
        """
        The number of trails in the graph.
        """
        return self.sequences.shape[1]

    def number_of_sequences(self):
        """
        The number of sequences in the graph.
        """
        return self.sequences.shape[0]
