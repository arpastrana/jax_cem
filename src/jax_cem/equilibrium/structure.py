from dataclasses import dataclass

import jax

import equinox as eqx

from typing import Dict
from typing import Tuple

import numpy as np
import jax.numpy as jnp

from compas.numerical import connectivity_matrix
from compas.utilities import pairwise


__all__ = ["EquilibriumStructure"]


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


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

    def __init__(
        self, nodes, edges, origin_nodes, support_nodes, trail_edges, deviation_edges, indirect_edges, sequences
    ):
        self.nodes = nodes
        self.edges = edges
        self.origin_nodes = origin_nodes
        self.support_nodes = support_nodes
        self.trail_edges = trail_edges
        self.deviation_edges = deviation_edges
        self.indirect_edges = indirect_edges
        self.sequences = sequences

        self.node_index = {node: index for index, node in enumerate(self.nodes)}
        self.edge_index = {tuple(edge): index for index, edge in enumerate(self.edges)}
        self.connectivity = jnp.asarray(connectivity_matrix(self.edges))
        self.incidence = self._incidence()
        self.sequences_edges = self._sequences_edges()

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

    def _sequences_edges(self):
        sequences = []
        for sequences_pair in pairwise(self.sequences):
            sequence = []
            for edge in zip(*sequences_pair):
                edge = tuple(edge)
                index = self.edge_index.get(edge, self.edge_index.get((edge[1], edge[0]), -1))
                sequence.append(index)
            sequences.append(sequence)
        return np.asarray(sequences)

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
        return structure_from_topology(cls, topology)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


class EquilibriumStructureFrozen(eqx.Module):
    """
    An immutable version of an equilibrium structure.
    """

    nodes: jnp.array
    edges: jnp.array
    origin_nodes: jnp.array
    trail_edges: jnp.array
    deviation_edges: jnp.array
    sequences: jnp.array
    sequences_edges: jnp.array
    connectivity: jnp.array
    incidence: jnp.array

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


@dataclass
class EquilibriumStructure2:
    """
    The attributed, undirected graph describing a pin-jointed bar structure.
    """

    nodes: np.array  # nodes
    edges: np.array  # pairs of nodes
    origin_nodes: np.array  # nodes
    support_nodes: np.array  # nodes
    trail_edges: np.array  # indices in edges, or mask?
    deviation_edges: np.array  # indices in edges or mask?
    sequences: np.array  # nodes verbatim

    _node_index: Dict[int, int] = None
    _edge_index: Dict[Tuple, int] = None
    _connectivity: jnp.array = None
    _incidence: jnp.array = None
    _sequences_edges: np.array = None

    @property
    def node_index(self):
        if self._node_index is None:
            self._node_index = {node: index for index, node in enumerate(self.nodes)}
        return self._node_index

    @property
    def edge_index(self):
        if self._edge_index is None:
            self._edge_index = {tuple(edge): index for index, edge in enumerate(self.edges)}
        return self._edge_index

    @property
    def connectivity(self):
        if self._connectivity is None:
            # edges = [(self.node_index[u], self.node_index[v]) for u, v in self.edges]
            self._connectivity = jnp.asarray(connectivity_matrix(self.edges))
        return self._connectivity

    @property
    def incidence(self):
        if self._incidence is None:
            incidence = np.zeros_like(self.connectivity)

            for node in self.nodes:
                edge_indices = np.nonzero(self.connectivity[:, node])
                connected_edges = self.edges[edge_indices]
                for i, edge in zip(np.reshape(edge_indices, (-1, 1)), connected_edges):
                    val = 1.0
                    if edge[0] != node:
                        val = -1.0
                    incidence[i, node] = val

            self._incidence = jnp.asarray(incidence)

        return self._incidence

    @property
    def sequences_edges(self):
        if self._sequences_edges is None:
            sequences = []
            for sequences_pair in pairwise(self.sequences):
                sequence = []
                for edge in zip(*sequences_pair):
                    edge = tuple(edge)
                    index = self.edge_index.get(edge, self.edge_index.get((edge[1], edge[0]), -1))
                    sequence.append(index)
                sequences.append(sequence)
            self._sequences_edges = jnp.asarray(sequences)
        return self._sequences_edges

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
        return frozen_structure(structure_from_topology(cls, topology))


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------


def frozen_structure(structure):
    """
    Return an immutable version of a structure.

    Parameters
    ----------
    structure : `jax_cem.equilibrium.EquilibriumStructure`
        A structure.

    Returns
    -------
    structure : `jax_cem.equilibrium.EquilibriumStructureFrozen`
        A frozen structure.
    """
    return EquilibriumStructureFrozen(
        nodes=structure.nodes,
        edges=structure.edges,
        origin_nodes=structure.origin_nodes,
        trail_edges=structure.trail_edges,
        deviation_edges=structure.deviation_edges,
        sequences=structure.sequences,
        sequences_edges=structure.sequences_edges,
        connectivity=structure.connectivity,
        incidence=structure.incidence,
    )


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
    trail_edges = np.asarray(trail_edges).astype(int)

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
