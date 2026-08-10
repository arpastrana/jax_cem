"""
Scaffolding that builds jax_cem objects from COMPAS CEM topology diagrams.

The fixtures were authored as topology diagrams. Phase 2 replaces these with the
native array constructor and deletes the module, along with the compas_cem
dependency it carries (docs/roadmap.md).
"""

from itertools import pairwise

import jax.numpy as jnp
import numpy as np

from jax_cem.datastructures import EquilibriumStructure
from jax_cem.parameters import ParameterState


def structure_from_topology(topology):
    """
    Create an equilibrium structure from a COMPAS CEM topology diagram.
    """
    assert topology.number_of_trails() > 0, "No trails in the diagram!"

    nodes = np.asarray(sorted(topology.nodes()))
    edges = np.asarray(list(topology.edges()))

    trail_edges = np.asarray(
        [float(topology.is_trail_edge(edge)) for edge in edges],
    ).astype(float)
    deviation_edges = np.logical_not(trail_edges).astype(float)

    indirect_edges = deviation_edges.copy()
    for i, edge in enumerate(edges):
        if topology.is_indirect_deviation_edge(edge):
            indirect_edges[i] = 0.0

    # Negated to mark the padding of a shifted trail.
    shape = (topology.number_of_sequences(), topology.number_of_trails())
    sequences = np.ones(shape).astype(int) * -1

    origin_nodes = []
    support_nodes = []
    for tidx, (onode, trail) in enumerate(topology.trails(True)):
        origin_nodes.append(onode)
        for sidx, node in enumerate(trail):
            sequences[topology.node_sequence(node)][tidx] = node
            if sidx == (len(trail) - 1):
                support_nodes.append(node)

    return EquilibriumStructure(
        nodes=nodes,
        edges=edges,
        origin_nodes=np.asarray(origin_nodes).astype(int),
        support_nodes=np.asarray(support_nodes).astype(int),
        trail_edges=trail_edges,
        deviation_edges=deviation_edges,
        indirect_edges=indirect_edges,
        sequences=sequences,
    )


def parameters_from_topology(topology):
    """
    Create a parameter state from a COMPAS CEM topology diagram.
    """
    nodes = sorted(topology.nodes())
    edges = list(topology.edges())

    loads = jnp.asarray([topology.node_load(node) for node in nodes])
    xyz = jnp.asarray([topology.node_coordinates(node) for node in nodes])

    forces = jnp.asarray([topology.edge_force(edge) for edge in edges])
    forces = jnp.reshape(forces, (-1, 1))

    lengths = np.zeros((topology.number_of_nodes(), 1))
    planes = np.zeros((topology.number_of_nodes(), 6))

    for trail in topology.trails():
        for u, v in pairwise(trail):
            edge = (u, v) if (u, v) in edges else (v, u)
            plane = topology.edge_plane(edge)
            if plane is not None:
                origin, normal = plane
                planes[u, :] = [*origin, *normal]
            else:
                length = topology.edge_length_2(edge)
                if not length:
                    raise ValueError(f"No length defined on trail edge {edge}")
                lengths[u, :] = length

    return ParameterState(
        xyz=xyz,
        loads=loads,
        lengths=jnp.asarray(lengths),
        planes=jnp.asarray(planes),
        forces=forces,
    )
