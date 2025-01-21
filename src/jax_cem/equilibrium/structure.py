import jax

import equinox as eqx

import numpy as np
import jax.numpy as jnp

from compas.numerical import connectivity_matrix
from compas.utilities import pairwise


class EquilibriumStructure(eqx.Module):
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
    node_index: jax.Array
    edge_index: jax.Array
    connectivity: jax.Array
    incidence: jax.Array
    sequences_edges: jax.Array
    sequences_edges_indices: jax.Array

    def __init__(
        self, nodes, edges, origin_nodes, support_nodes, trail_edges, deviation_edges, indirect_edges, sequences
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
        self.connectivity = jnp.asarray(connectivity_matrix(self.edges))
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
                index = self.edge_index.get(edge, self.edge_index.get((edge[1], edge[0]), -1))
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

    @classmethod
    def from_topology_diagram(cls, topology):
        """
        Create a equilibrium graph from a COMPAS CEM topology diagram.

        Parameters
        ----------
        topology : `compas_cem.diagrams.TopologyDiagram`
            A valid topology diagram.
        """
        return structure_from_topology(cls, topology)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def structure_from_topology(cls, topology):
    """
    Create an equilibrium model from a COMPAS CEM topology diagram.

    Parameters
    ----------
    topology : `compas_cem.diagrams.TopologyDiagram`
        A valid topology diagram.

    Returns
    -------
    structure : `jax_cem.equilibrium.EquilibriumStructure`
        A structure.
    """
    # there must be at least one trail
    assert topology.number_of_trails() > 0, "No trails in the diagram!"

    # nodes
    # TODO: Is sorting not introducing bugs here?
    nodes = np.asarray(sorted(list(topology.nodes())))

    # edges
    edges = np.asarray(list(topology.edges()))

    # trail edges
    trail_edges = []
    for edge in edges:
        val = 0.0
        if topology.is_trail_edge(edge):
            val = 1.0
        trail_edges.append(val)
    trail_edges = np.asarray(trail_edges).astype(float)

    # deviation edges
    deviation_edges = np.logical_not(trail_edges).astype(float)

    # indirect deviation edges
    indirect_edges = deviation_edges.copy()  # np.zeros_like(deviation_edges)
    for i, edge in enumerate(edges):
        if topology.is_indirect_deviation_edge(edge):
            indirect_edges[i] = 0.0

    # sequences
    sequences = np.ones((topology.number_of_sequences(), topology.number_of_trails())).astype(int)

    # negate to deal with shifted trail
    sequences *= -1

    origin_nodes = []
    support_nodes = []
    for tidx, (onode, trail) in enumerate(topology.trails(True)):
        origin_nodes.append(onode)
        for sidx, node in enumerate(trail):
            seq = topology.node_sequence(node)
            sequences[seq][tidx] = node
            if sidx == (len(trail) - 1):
                support_nodes.append(node)

    origin_nodes = np.asarray(origin_nodes).astype(int)
    support_nodes = np.asarray(support_nodes).astype(int)

    return cls(
        nodes=nodes,
        edges=edges,
        origin_nodes=origin_nodes,
        support_nodes=support_nodes,
        trail_edges=trail_edges,
        deviation_edges=deviation_edges,
        indirect_edges=indirect_edges,
        sequences=sequences,
    )
