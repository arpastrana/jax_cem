import numpy as np

from compas.geometry import Translation

from compas_cem.diagrams import TopologyDiagram

from compas_cem.elements import Node
from compas_cem.elements import TrailEdge
from compas_cem.elements import DeviationEdge

from compas_cem.loads import NodeLoad
from compas_cem.supports import NodeSupport

from compas_cem.equilibrium import static_equilibrium_numpy

from compas_cem.plotters import Plotter

from jax_cem.equilibrium import EquilibriumModel
from jax_cem.datastructures import EquilibriumStructure
from jax_cem.parameters import ParameterState
from jax_cem.datastructures import form_from_eqstate

from jax.tree_util import tree_map

# -------------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------------

points = [(0, [0.0, 0.0, 0.0]),
          (1, [0.0, 1.0, 0.0]),
          (2, [0.0, 2.0, 0.0]),
          (3, [1.0, 0.0, 0.0]),
          (4, [1.0, 1.0, 0.0]),
          (5, [1.0, 2.0, 0.0])]

trail_edges = [(0, 1),
               (1, 2),
               (3, 4),
               (4, 5)]

deviation_edges = [(1, 4),
                   (2, 5)]


# points = [(0, [0.0, 0.0, 0.0]),
#           (1, [0.0, 1.0, 0.0]),
#           (2, [1.0, 1.0, 0.0]),
#           (3, [1.0, 0.0, 0.0]),
#           ]

# trail_edges = [(0, 1),
#                (2, 3),
#                ]

# deviation_edges = [(1, 2)
#                    ]

# ------------------------------------------------------------------------------
# Topology Diagram
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

for u, v in trail_edges:
    topology.add_edge(TrailEdge(u, v, length=-1.0))

# ------------------------------------------------------------------------------
# Add Deviation Edges
# ------------------------------------------------------------------------------

for u, v in deviation_edges:
    topology.add_edge(DeviationEdge(u, v, force=-1.0))

# ------------------------------------------------------------------------------
# Add Indirect Deviation Edges
# ------------------------------------------------------------------------------

topology.add_edge(DeviationEdge(1, 5, force=1.0))
topology.add_edge(DeviationEdge(1, 3, force=1.0))
topology.add_edge(DeviationEdge(2, 4, force=1.0))
topology.add_edge(DeviationEdge(0, 4, force=1.0))

# force = 1.0
# topology.add_edge(DeviationEdge(0, 2, force=force))
# topology.add_edge(DeviationEdge(1, 3, force=force))

# ------------------------------------------------------------------------------
# Set Supports Nodes
# ------------------------------------------------------------------------------

topology.add_support(NodeSupport(0))
topology.add_support(NodeSupport(3))

# ------------------------------------------------------------------------------
# Add Loads
# ------------------------------------------------------------------------------

load = [0.0, -1.0, 0.0]
topology.add_load(NodeLoad(2, load))
topology.add_load(NodeLoad(5, load))
# topology.add_load(NodeLoad(1, load))
# topology.add_load(NodeLoad(2, load))

# ------------------------------------------------------------------------------
# Equilibrium of forces
# ------------------------------------------------------------------------------

tmax = 1000
topology.build_trails()
form = static_equilibrium_numpy(topology, eta=1e-9, tmax=tmax, verbose=False)

# ------------------------------------------------------------------------------
# Compute a state of static equilibrium with JAX CEM
# ------------------------------------------------------------------------------

structure = EquilibriumStructure.from_topology_diagram(topology)
parameters = ParameterState.from_topology_diagram(topology)
# model = EquilibriumModel.from_topology_diagram(topology)
model = EquilibriumModel(tmax=tmax, verbose=True)
eqstate = model(parameters, structure)
# tree_map(lambda x: print(x), eqstate)
form_jax = form_from_eqstate(structure, eqstate)

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------

xyz_compas = [form.node_coordinates(node) for node in structure.nodes]
reactions_compas = [form.reaction_force(node) for node in structure.nodes]
lengths_compas = [form.edge_length(*edge) for edge in structure.edges]
forces_compas = [form.edge_force(edge) for edge in structure.edges]

assert np.allclose(np.asarray(xyz_compas), eqstate.xyz, atol=1e-6), f"{xyz_compas}\n{eqstate.xyz}"
assert np.allclose(np.asarray(reactions_compas), eqstate.reactions), f"{reactions_compas}\n{eqstate.reactions}"
assert np.allclose(np.asarray(lengths_compas), eqstate.lengths.ravel()), f"{lengths_compas}\n{eqstate.lengths}"
assert np.allclose(np.asarray(forces_compas), eqstate.forces.ravel()), f"\n{forces_compas}\n{eqstate.forces}"

print("happy ever after")

# ------------------------------------------------------------------------------
# Plotter
# ------------------------------------------------------------------------------

plotter = Plotter()

# ------------------------------------------------------------------------------
# Plot topology diagram
# ------------------------------------------------------------------------------

plotter.add(topology, nodesize=0.2, show_nodetext=True)

# ------------------------------------------------------------------------------
# Plot translated form diagram
# ------------------------------------------------------------------------------

plotter.add(form.transformed(Translation.from_vector([2.0, 0.0, 0.0])),
            nodesize=0.2,
            loadscale=0.5,
            reactionscale=0.5,
            edgetext="force",
            show_edgetext=False)

plotter.add(form_jax.transformed(Translation.from_vector([4.0, 0.0, 0.0])),
            nodesize=0.2,
            loadscale=0.5,
            reactionscale=0.5,
            edgetext="force",
            show_edgetext=False)

# ------------------------------------------------------------------------------
# Plot scene
# -------------------------------------------------------------------------------

plotter.zoom_extents()
plotter.show()
