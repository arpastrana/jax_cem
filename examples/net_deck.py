import os

from functools import partial
from time import time

from compas.geometry import Translation
from compas.geometry import scale_vector
from compas.utilities import geometric_key

from compas_cem.diagrams import TopologyDiagram

from compas_cem.equilibrium import static_equilibrium

from compas_cem.optimization import Optimizer

from compas_cem.optimization import PointConstraint

from compas_cem.optimization import DeviationEdgeParameter

from compas_cem.plotters import Plotter
from compas_cem.viewers import Viewer

import jaxopt

from jax import jit

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from jax_cem.equilibrium import EquilibriumModel
from jax_cem.equilibrium import EquilibriumStructure
from jax_cem.equilibrium import form_from_eqstate

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import EquilibriumModel as FDModel
from jax_fdm.equilibrium import EquilibriumStructure as FDStructure
from jax_fdm.equilibrium import network_updated
from jax_fdm.visualization import Plotter as PlotterFD


VIEW = True
OPTIMIZE_CEM = True
OPTIMIZE_FDM = True

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------

HERE = os.path.dirname(__file__)
IN_DECK = os.path.abspath(os.path.join(HERE, "data/deck.json"))
IN_NET = os.path.abspath(os.path.join(HERE, "data/net.json"))

# ------------------------------------------------------------------------------
# Load topology diagram from JSON
# ------------------------------------------------------------------------------

topology = TopologyDiagram.from_json(IN_DECK)

# ------------------------------------------------------------------------------
# Copy topology
# ------------------------------------------------------------------------------

topology.build_trails(True)
topology.auxiliary_trail_length = 0.1
for edge in topology.auxiliary_trail_edges():
    topology.edge_attribute(edge, "length", 0.1)

print(f"{topology.number_of_indirect_deviation_edges()=}")

for i in range(10):
    if topology.number_of_indirect_deviation_edges() == 0:
        print(f"No indirect deviation edges at iteration: {i}")
        break

    for node_origin in topology.origin_nodes():

        for edge in topology.connected_edges(node_origin):

            if topology.is_indirect_deviation_edge(edge):
                u, v = edge
                node_other = u if node_origin != u else v
                sequence = topology.node_sequence(node_origin)
                sequence_other = topology.node_sequence(node_other)

                if sequence_other > sequence:
                    topology.shift_trail(node_origin, sequence_other)

print(f"{topology.number_of_indirect_deviation_edges()=}")

# ------------------------------------------------------------------------------
# JAX CEM - form finding
# ------------------------------------------------------------------------------

structure = EquilibriumStructure.from_topology_diagram(topology)
model = EquilibriumModel.from_topology_diagram(topology)
eqstate = model(structure, tmax=1)
form_jax = form_from_eqstate(structure, eqstate)

# ------------------------------------------------------------------------------
# JAX CEM - optimization
# ------------------------------------------------------------------------------

if OPTIMIZE_CEM:

    nodes_opt = []
    for node in topology.nodes():
        if topology.is_node_origin(node):
            continue
        if topology.is_node_support(node):
            neighbor = topology.neighbors(node).pop()
            if topology.is_node_origin(neighbor):
                continue
        nodes_opt.append(node)

    xyz_target = []
    indices_opt = []
    for node in nodes_opt:
        index = structure.node_index[node]
        indices_opt.append(index)
        xyz_target.append(topology.node_coordinates(node))

    # define loss function
    @jit
    def loss_fn(diff_model, static_model, structure, y):
        model = eqx.combine(diff_model, static_model)
        eqstate = model(structure, tmax=1)
        pred_y = eqstate.xyz[indices_opt, :]
        return jnp.sum((pred_y - y) ** 2)

    # define targets
    y = jnp.asarray(xyz_target)

    # set tree filtering specification
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(lambda tree: (tree.forces), filter_spec, replace=(True))

    # split model into differentiable and static submodels
    diff_model, static_model = eqx.partition(model, filter_spec)

    bounds_low_compas = jnp.zeros_like(model.forces)
    bounds_up_compas = jnp.ones_like(model.forces) * jnp.inf

    bound_low = eqx.tree_at(lambda tree: (tree.forces),
                            diff_model,
                            replace=(bounds_low_compas))
    bound_up = eqx.tree_at(lambda tree: (tree.forces),
                           diff_model,
                           replace=(bounds_up_compas))

    bounds = (bound_low, bound_up)

    # evaluate loss function at the start
    loss = loss_fn(diff_model, static_model, structure, y)
    print(f"{loss=}")

    # # solve optimization problem with scipy
    print("\n***Optimizing CEM with scipy***")
    optimizer = jaxopt.ScipyMinimize
    # optimizer = jaxopt.ScipyBoundedMinimize

    opt = optimizer(fun=loss_fn, method="L-BFGS-B", jit=True, tol=1e-6, maxiter=300)

    start = time()
    opt_result = opt.run(diff_model, static_model, structure, y)
    # opt_result = opt.run(diff_model, bounds, static_model, structure, y)
    print(f"Opt time: {time() - start:.4f} sec")
    diff_model_star, opt_state_star = opt_result
    print(opt_state_star)

    # evaluate loss function at optimum point
    loss = loss_fn(diff_model_star, static_model, structure, y)
    print(f"{loss=}")

    # generate optimized compas cem form diagram
    model_star = eqx.combine(diff_model_star, static_model)
    eqstate_star = model_star(structure)
    form_jax_opt = form_from_eqstate(structure, eqstate_star)

# ------------------------------------------------------------------------------
# Load network from JSON
# ------------------------------------------------------------------------------

network = FDNetwork.from_json(IN_NET)
network.edges_forcedensities(q=2.0)
gkey_key = network.gkey_key()

# add loads
nodes_opt_fdm = []
for node in topology.nodes():
    if topology.is_node_origin(node):
        neighbor = topology.neighbors(node).pop()
        if topology.is_node_support(neighbor):
            reaction = form_jax_opt.reaction_force(neighbor)
            key = gkey_key[geometric_key(topology.node_coordinates(node))]
            network.node_load(key, scale_vector(reaction, -1.))
            nodes_opt_fdm.append(key)

# network = fdm(network)

# ------------------------------------------------------------------------------
# JAX FDM - optimization
# ------------------------------------------------------------------------------

model = FDModel.from_network(network)
structure = FDStructure.from_network(network)
# eqstate = model(structure)
# network = network_updated(structure.network, eqstate)
# form_jax = form_from_eqstate(structure, eqstate)

if OPTIMIZE_FDM:

    indices_res_opt = []
    for node in nodes_opt_fdm:
        index = structure.node_index[node]
        indices_res_opt.append(index)

    indices_xyz_opt = []
    xyz_target = []
    for node in network.nodes_where({"is_target": True}):
        index = structure.node_index[node]
        indices_xyz_opt.append(index)
        xyz = network.node_coordinates(node)
        xyz_target.append(xyz)

    xyz_target = jnp.asarray(xyz_target)

    # define loss function
    def loss_fn(diff_model, static_model, y):
        model = eqx.combine(diff_model, static_model)
        eqstate = model(structure)
        pred_y = eqstate.residuals[indices_res_opt, :]
        xyz_pred = eqstate.xyz[indices_xyz_opt, :]
        return jnp.sum((pred_y - y) ** 2) + jnp.sum((xyz_pred - xyz_target) ** 2)

    # define targets
    y = 0.0

    # set tree filtering specification
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(lambda tree: (tree.q), filter_spec, replace=(True))

    # split model into differentiable and static submodels
    diff_model, static_model = eqx.partition(model, filter_spec)

    bounds_low_compas = jnp.zeros_like(model.q)
    bounds_up_compas = jnp.ones_like(model.q) * jnp.inf

    bound_low = eqx.tree_at(lambda tree: (tree.q),
                            diff_model,
                            replace=(bounds_low_compas))
    bound_up = eqx.tree_at(lambda tree: (tree.q),
                           diff_model,
                           replace=(bounds_up_compas))

    bounds = (bound_low, bound_up)

    print("\n***Optimizing FDM with scipy***")
    # evaluate loss function at the start
    loss = loss_fn(diff_model, static_model, y)
    print(f"{loss=}")

    # # solve optimization problem with scipy
    # optimizer = jaxopt.ScipyMinimize
    optimizer = jaxopt.ScipyBoundedMinimize

    opt = optimizer(fun=loss_fn, method="L-BFGS-B", jit=True, tol=1e-9, maxiter=5000, implicit_diff_solve=False)

    start = time()
    # opt_result = opt.run(diff_model, static_model, y)
    opt_result = opt.run(diff_model, bounds, static_model, y)
    print(f"Opt time: {time() - start:.4f} sec")
    diff_model_star, opt_state_star = opt_result
    print(opt_state_star)

    # evaluate loss function at optimum point
    # loss = loss_fn(diff_model_star, static_model, structure, y)
    loss = loss_fn(diff_model_star, static_model, y)
    print(f"{loss=}")

    # generate optimized compas cem form diagram
    model_star = eqx.combine(diff_model_star, static_model)
    eqstate_star = model_star(structure)
    network_jax_opt = network_updated(structure.network, eqstate_star)


plotter = PlotterFD(figsize=(8, 5), dpi=200)
# plotter.add(network, nodesize=2, edgecolor="fd", show_reactions=True, show_lodes=False, show_nodes=True, reactionscale=1.0)
plotter.add(network_jax_opt,
            nodesize=2,
            edgecolor="fd",
            show_reactions=True,
            show_loads=False,
            edgewidth=(1., 3.),
            edgetext="fd",
            show_edgetext=True,
            show_nodes=True,
            reactionscale=1.0)

plotter.add(form_jax_opt, nodesize=2, show_reactions=False, show_loads=False)
plotter.zoom_extents()
plotter.show()

# print(network)
# print(network.number_of_supports())
# print(list(network.nodes_where({"is_support": True})))
# raise

# ------------------------------------------------------------------------------
# Launch viewer
# ------------------------------------------------------------------------------

# if VIEW:

#     shift_vector = [0.0, -10.0, 0.0]
#     # form = form.transformed(Translation.from_vector(scale_vector(shift_vector, 0.)))
#     form_jax = form_jax.transformed(Translation.from_vector(scale_vector(shift_vector, 0.)))
#     # forms = [form, form_jax]
#     forms = [form_jax]

#     # i = 2
#     # if OPTIMIZE:
#         # form_opt = form_opt.transformed(Translation.from_vector(scale_vector(shift_vector, 0 * 2.)))
#         # forms.append(form_opt)
#         # i += 1.

#     if OPTIMIZE_CEM:
#         form_jax_opt = form_jax_opt.transformed(Translation.from_vector(scale_vector(shift_vector, 0 * 2.)))
#         forms.append(form_jax_opt)
#         i += 1.

#     # viewer = Viewer(width=1600, height=900, show_grid=False)
#     # viewer.view.color = (0.5, 0.5, 0.5, 1)  # change background to black
#     viewer = Plotter(figsize=(8, 5), dpi=200)

# # ------------------------------------------------------------------------------
# # Visualize topology diagram
# # ------------------------------------------------------------------------------

#     viewer.add(topology,
#                edgewidth=0.3,
#                show_loads=False)

# # ------------------------------------------------------------------------------
# # Visualize translated form diagram
# # ------------------------------------------------------------------------------

#     for form in forms:
#         viewer.add(form,
#                    edgewidth=(0.5, 2.),
#                    show_nodes=True,
#                    nodesize=2,
#                    show_loads=False,
#                    show_edgetext=False,
#                    show_reactions=False,
#                    reactionscale=4.0,
#                    edgetext="key",
#                    )

# # ------------------------------------------------------------------------------
# # Show scene
# # -------------------------------------------------------------------------------

#     viewer.zoom_extents()
#     viewer.show()
