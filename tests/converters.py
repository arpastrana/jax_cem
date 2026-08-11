"""
Scaffolding that builds jax_cem objects from COMPAS CEM topology diagrams.

The fixtures were authored as topology diagrams. Phase 2 replaces these with the
native array constructor and deletes the module, along with the compas_cem
dependency it carries (docs/roadmap.md).
"""

from itertools import pairwise

import jax.numpy as jnp
import numpy as np
from compas_cem.diagrams import FormDiagram
from compas_cem.elements import Edge
from compas_cem.elements import Node

from jax_cem.datastructures import EquilibriumStructure
from jax_cem.parameters import ParameterState


def structure_from_topology(topology):
    """
    Create an equilibrium structure from a COMPAS CEM topology diagram.
    """
    assert topology.number_of_trails() > 0, "No trails in the diagram!"

    nodes = np.asarray(sorted(topology.nodes()))

    edges_trail = []
    edges_deviation = []
    for edge in topology.edges():
        if topology.is_trail_edge(edge):
            edges_trail.append(edge)
        else:
            edges_deviation.append(edge)

    supports = sorted(trail[-1] for _, trail in topology.trails(True))

    return EquilibriumStructure(
        nodes=nodes,
        supports=np.asarray(supports, dtype=int),
        edges_trail=np.asarray(edges_trail, dtype=int).reshape(-1, 2),
        edges_deviation=np.asarray(edges_deviation, dtype=int).reshape(-1, 2),
    )


def parameters_from_topology(topology, structure):
    """
    Create a parameter state from a COMPAS CEM topology diagram.

    Notes
    -----
    Per-edge parameters follow the edge order of the structure, which puts trail
    edges before deviation edges, rather than the order of the diagram.
    """
    nodes = sorted(topology.nodes())

    loads = jnp.asarray([topology.node_load(node) for node in nodes])
    xyz = jnp.asarray([topology.node_coordinates(node) for node in nodes])

    forces = np.zeros((structure.num_edges, 1))
    edge_index = structure.edge_index
    for edge in topology.edges():
        u, v = edge
        index = edge_index.get((u, v), edge_index.get((v, u)))
        forces[index, :] = topology.edge_force(edge)

    lengths = np.zeros((topology.number_of_nodes(), 1))
    planes = np.zeros((topology.number_of_nodes(), 6))

    edges = list(topology.edges())
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
        forces=jnp.asarray(forces),
    )


def form_from_eqstate(eqstate, structure):
    """
    Build a COMPAS CEM form diagram from an equilibrium state.

    Notes
    -----
    `FormDiagram.from_equilibrium_state` reads a `support_nodes` attribute, which
    this structure calls `supports`, so the diagram is assembled here instead.
    """
    form = FormDiagram()

    for node in structure.nodes:
        form.add_node(Node(int(node)))

    for node in structure.supports:
        form.node_attribute(int(node), "type", "support")

    for u, v in structure.edges:
        form.add_edge(Edge(int(u), int(v), {}))

    xyz = eqstate.xyz.tolist()
    loads = eqstate.loads.tolist()
    reactions = eqstate.reactions.tolist()
    lengths = eqstate.lengths.tolist()
    forces = eqstate.forces.tolist()

    edge_index = structure.edge_index
    for edge in structure.edges:
        u, v = int(edge[0]), int(edge[1])
        index = edge_index[(u, v)]
        form.edge_attribute((u, v), name="force", value=forces[index].pop())
        form.edge_attribute((u, v), name="lengths", value=lengths[index].pop())

    for node in structure.nodes:
        key = int(node)
        form.node_attributes(key, "xyz", xyz[key])
        form.node_attributes(key, ["rx", "ry", "rz"], reactions[key])
        form.node_attributes(key, ["qx", "qy", "qz"], loads[key])

    return form
