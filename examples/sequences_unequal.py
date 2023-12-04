import jax
import jaxopt

import numpy as np
import jax.numpy as jnp

import equinox as eqx

from jax import jit

from compas.colors import Color
from compas.geometry import Translation
from compas.geometry import Point

from compas_cem.diagrams import TopologyDiagram

from compas_cem.elements import Node
from compas_cem.elements import TrailEdge
from compas_cem.elements import DeviationEdge

from compas_cem.loads import NodeLoad
from compas_cem.supports import NodeSupport

from compas_cem.equilibrium import static_equilibrium

from compas_cem.plotters import Plotter

from jax_cem.equilibrium import EquilibriumModel
from jax_cem.equilibrium import EquilibriumStructure
from jax_cem.equilibrium import form_from_eqstate

from jax.tree_util import tree_map

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------

plot = True

# -------------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------------

points = [
    (0, [1.0, 0.0, 0.0]),
    (1, [1.0, -1.0, 0.0]),
    (2, [1.0, -2.0, 0.0]),
    (3, [1.0, -3.0, 0.0]),
    (4, [2.0, 0.0, 0.0]),
    (5, [2.0, -1.0, 0.0]),
    (6, [2.0, -2.0, 0.0]),
    (7, [2.0, -3.0, 0.0]),
]

# key: plane
trail_edges = {
    (0, 1): ([0.0, -1.0, 0.0], [0.0, -1.0, 0.0]),
    (1, 2): ([0.0, -2.0, 0.0], [0.0, -1.0, 0.0]),
    (2, 3): ([0.0, -3.0, 0.0], [0.0, -1.0, 0.0]),
}

deviation_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]

length = -1.0
force = -2.0
load = [0.0, -1.0, 0.0]

# ------------------------------------------------------------------------------
# Instantiate a topology diagram
# ------------------------------------------------------------------------------

topology = TopologyDiagram()

# ------------------------------------------------------------------------------
# Add Nodes
# ------------------------------------------------------------------------------

for key, point in points:
    topology.add_node(Node(key, point))

# ------------------------------------------------------------------------------
# Add Trail Edges
# ------------------------------------------------------------------------------

for (u, v), plane in trail_edges.items():
    topology.add_edge(TrailEdge(u, v, length=length, plane=plane))

# ------------------------------------------------------------------------------
# Add Deviation Edges
# ------------------------------------------------------------------------------

for u, v in deviation_edges:
    topology.add_edge(DeviationEdge(u, v, force=force))

# ------------------------------------------------------------------------------
# Set Supports Nodes
# ------------------------------------------------------------------------------

topology.add_support(NodeSupport(3))

# ------------------------------------------------------------------------------
# Add Loads
# ------------------------------------------------------------------------------

for node in range(4):
    topology.add_load(NodeLoad(0, [0.0, -1.0, 0.0]))

# ------------------------------------------------------------------------------
# Build trails automatically
# ------------------------------------------------------------------------------

topology.build_trails(auxiliary_trails=True)

# ------------------------------------------------------------------------------
# Shift trails to remove indirect deviation edges
# ------------------------------------------------------------------------------

for i in range(100):
    if topology.number_of_indirect_deviation_edges() == 0:
        print(f"No indirect deviation edges after {i} iterations!")
        break
    for node in topology.origin_nodes():
        edges = topology.connected_edges(node)
        for edge in edges:
            if not topology.is_indirect_deviation_edge(edge):
                continue
            u, v = edge
            node_other = u if node != u else v
            sequence = topology.node_sequence(node)
            sequence_other = topology.node_sequence(node_other)

            if sequence_other != sequence:
                topology.shift_trail(node, sequence_other)

# ------------------------------------------------------------------------------
# Compute a state of static equilibrium with COMPAS CEM
# ------------------------------------------------------------------------------

form = static_equilibrium(topology, tmax=1)

# for node in form.nodes():
#     print(f"{node}: {form.node_coordinates(node)}")

# for node in form.nodes():
#     print(f'{node}: {form.node_attributes(node, names=["rx", "ry", "rz"])}')

# for edge in form.edges():
#     print(f'{edge}: {form.edge_attribute(edge, "length")}')

# for edge in form.edges():
#     print(f'{edge}: {form.edge_attribute(edge, "force")}')

# ------------------------------------------------------------------------------
# Compute a state of static equilibrium with JAX CEM
# ------------------------------------------------------------------------------

structure = EquilibriumStructure.from_topology_diagram(topology)
model = EquilibriumModel.from_topology_diagram(topology)

from jax import jit
from equinox import filter_jit as jit

# model = jit(model)  # , static_argnums=(1, ))
eqstate = model(structure, tmax=1)
# print(eqstate.xyz)

# form = EquilibriumForm(structure, eqstate)  # an idea for the future
form_jax = form_from_eqstate(structure, eqstate)

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

xyz_compas = np.asarray([form.node_coordinates(node) for node in structure.nodes])
reactions_compas = np.asarray([form.reaction_force(node) for node in structure.nodes])
lengths_compas = np.asarray([form.edge_length(*edge) for edge in structure.edges])
forces_compas = np.asarray([form.edge_force(edge) for edge in structure.edges])

print(f"distance jax to compas: {np.linalg.norm(eqstate.xyz - xyz_compas):.4f}")

assert np.allclose(xyz_compas, eqstate.xyz), f"\n{xyz_compas}\n{eqstate.xyz}"
assert np.allclose(reactions_compas, eqstate.reactions)
assert np.allclose(lengths_compas, eqstate.lengths.ravel()), f"{lengths_compas}\n{eqstate.lengths}"
assert np.allclose(forces_compas, eqstate.forces.ravel()), f"\n{forces_compas}\n{eqstate.forces}"

print("happy ever after")

# ------------------------------------------------------------------------------
# Plot results
# ------------------------------------------------------------------------------

if plot:
    plotter = Plotter(figsize=(12, 8))

    plotter.add(Point(2, -3, 0))

    # add topology diagram to scene
    # artist = plotter.add(topology, nodesize=0.2, nodetext="sequence", nodecolor="sequence", show_nodetext=False)

    # add shifted form diagram to the scene
    form = form.transformed(Translation.from_vector([0.0, -5.0, 0.0]))
    plotter.add(form, nodesize=0.2, show_edgetext=False, edgetext="key", show_nodetext=True)

    # add shifted form diagram to the scene
    form_jax = form_jax.transformed(Translation.from_vector([0.0, 0.0, 0.0]))
    plotter.add(form_jax, nodesize=0.2, show_edgetext=False, edgetext="key", show_nodetext=True)

    # show plotter contents
    plotter.zoom_extents()
    plotter.show()
