from dataclasses import dataclass

from typing import Dict
from typing import Tuple
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from compas.datastructures import network_connectivity_matrix


__all__ = ["EquilibriumStructure"]


DTYPE_NP_I = np.int64
DTYPE_NP_F = np.float64


# NOTE: Treat this object as a struct or as a standard object?
# class EquilibriumGraph(NamedTuple):
@dataclass
class EquilibriumStructure:
    """
    The topological graph of a discrete, pin-jointed structure.
    """
    nodes: Tuple # nodes
    edges: Tuple  # pairs of nodes
    origin_nodes: Tuple  # indices in nodes, or mask?
    trail_edges: Tuple  # indices in edges, or mask?
    deviation_edges: Tuple  # indices in edges or mask?
    sequences: np.array  # nodes verbatim

    # TODO: either incidence or connectivity should go
    _node_index: Dict[int, int] = None
    _edge_index: Dict[Tuple, int] = None
    _incidence: jnp.array = None
    _incidence_signed: jnp.array = None
    _connectivity: jnp.array = None

    @property
    def node_index(self):
        if self._node_index is None:
            self._node_index = {node: index for index, node in enumerate(self.nodes)}
        return self._node_index

    @property
    def edge_index(self):
        if self._edge_index is None:
            self._edge_index = {edge: index for index, edge in enumerate(self.edges)}
        return self._edge_index

    @property
    def connectivity(self):
        if self._connectivity is None:
            edges = [(self.node_index[u], self.node_index[v]) for u, v in network.edges()]
            self._connectivity = connectivity_matrix(edges)
        return self._connectivity

    @property
    def incidence(self):
        if self._incidence is None:
            self._incidence = np.abs(self.connectivity)
        return self._incidence

    @property
    def incidence_signed(self):
        if self._incidence_signed is None:
            self._incidence_signed = self.incidence
        return self._incidence_signed

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
        # there must be at least one trail
        assert topology.number_of_trails() > 0, "No trails in the diagram!"

        nodes = tuple(topology.nodes())
        edges = tuple(topology.edges())

        # trail edges
        trail_edges = []
        for edge in edges:
            val = 0.
            if topology.is_trail_edge(edge):
                val = 1.
            trail_edges.append(val)
        trail_edges = np.asarray(trail_edges, dtype=DTYPE_NP_F)

        # deviation edges
        deviation_edges = np.logical_not(trail_edges)

        # sequences
        node_index = topology.key_index()
        sequences = np.ones((topology.number_of_sequences(),
                             topology.number_of_trails()),
                             dtype=DTYPE_NP_I)

        sequences *= -1  # negate to deal with shifted trails

        for tidx, trail in enumerate(topology.trails()):
            for node in trail:
                seq = topology.node_sequence(node)
                sequences[seq][tidx] = node_index[node]

        # incidence matrix
        # NOTE: converted to jax numpy array. Is a tuple a better choice?
        incidence = jnp.asarray(network_signed_incidence_matrix(topology))

        # connectivity matrix
        # connectivity = network_connectivity_matrix(topology)

        return cls(nodes=nodes,
                   edges=edges,
                   origin_nodes=origin_nodes,
                   trail_edges=trail_edges,
                   deviation_edges=deviation_edges,
                   sequences=sequences)

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

# def network_incidence_matrix(network):
#     """
#     Calculate the incidence matrix of a network.
#     """
#     return np.abs(network_connectivity_matrix(network))


def network_signed_incidence_matrix(network):
    """
    Compute the signed incidence matrix of a network.
    """
    node_index = network.key_index()
    edge_index = network.uv_index()
    incidence = network_incidence_matrix(network)

    for node in network.nodes():
        j = node_index[node]

        for edge in network.connected_edges(node):
            i = edge_index[edge]
            val = 1.
            if edge[0] != node:
                val = -1.
            incidence[i, j] = val

    return incidence
