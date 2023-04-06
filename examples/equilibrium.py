import jax
import jaxopt

import numpy as np
import jax.numpy as jnp

import equinox as eqx

from jax import jit

from compas.colors import Color
from compas.geometry import Translation

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
    (1, [1.0, 1.0, 0.0]),
    (2, [1.0, 2.0, 0.0]),
    (3, [2.0, 0.0, 0.0]),
    (4, [2.0, 1.0, 0.0]),
    (5, [2.0, 2.0, 0.0]),
]

# key: plane
trail_edges = {
    (0, 1): ([0.0, -1.5, 0.0], [0.0, -1.0, 0.0]),
    (1, 2): ([0.0, -3.0, 0.0], [0.0, -1.0, 0.0]),
    (3, 4): ([0.0, -1.5, 0.0], [0.0, -1.0, 0.0]),
    (4, 5): ([0.0, -3.0, 0.0], [0.0, -1.0, 0.0]),
}

deviation_edges = [(0, 3), (1, 4), (2, 5)]

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
# Add Indirect Deviation Edges
# ------------------------------------------------------------------------------

# topology.add_edge(DeviationEdge(1, 5, force=1.0))
# topology.add_edge(DeviationEdge(1, 3, force=1.0))
# topology.add_edge(DeviationEdge(2, 4, force=1.0))

# ------------------------------------------------------------------------------
# Set Supports Nodes
# ------------------------------------------------------------------------------

topology.add_support(NodeSupport(2))
topology.add_support(NodeSupport(5))

# ------------------------------------------------------------------------------
# Add Loads
# ------------------------------------------------------------------------------

topology.add_load(NodeLoad(0, load))
topology.add_load(NodeLoad(3, load))
topology.add_load(NodeLoad(1, load))
topology.add_load(NodeLoad(4, load))

# ------------------------------------------------------------------------------
# Build trails automatically
# ------------------------------------------------------------------------------

topology.build_trails()

# ------------------------------------------------------------------------------
# Compute a state of static equilibrium
# ------------------------------------------------------------------------------

form = static_equilibrium(topology)

structure = EquilibriumStructure.from_topology_diagram(topology)
model = EquilibriumModel.from_topology_diagram(topology)
print(model.planes)
eqstate = model(structure)
tree_map(lambda x: print(x), eqstate)

# form = EquilibriumForm(structure, eqstate)
form_jax = form_from_eqstate(structure, eqstate)

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

xyz_compas = [form.node_coordinates(node) for node in structure.nodes]
reactions_compas = [form.reaction_force(node) for node in structure.nodes]
lengths_compas = [form.edge_length(*edge) for edge in structure.edges]
forces_compas = [form.edge_force(edge) for edge in structure.edges]

assert np.allclose(np.asarray(xyz_compas), eqstate.xyz)
assert np.allclose(np.asarray(reactions_compas), eqstate.reactions)
assert np.allclose(np.asarray(lengths_compas), eqstate.lengths.ravel()), f"{lengths_compas}\n{eqstate.lengths}"
assert np.allclose(np.asarray(forces_compas), eqstate.forces.ravel()), f"\n{forces_compas}\n{eqstate.forces}"

print("happy ever after")

# ------------------------------------------------------------------------------
# Plot results
# ------------------------------------------------------------------------------

if plot:
    plotter = Plotter(figsize=(12, 8))

    # add topology diagram to scene
    artist = plotter.add(topology, nodesize=0.2, nodetext="key", nodecolor="sequence", show_nodetext=True)

    # add shifted form diagram to the scene
    form = form.transformed(Translation.from_vector([0.0, -1.0, 0.0]))
    plotter.add(form, nodesize=0.2, show_edgetext=False, edgetext="key")

    # add shifted form diagram to the scene
    form_jax = form_jax.transformed(Translation.from_vector([0.0, -1.0, 0.0]))
    plotter.add(form_jax, nodesize=0.2, edgewidth=2.0, show_edgetext=True, edgetext="key")

    # show plotter contents
    plotter.zoom_extents()
    plotter.show()
