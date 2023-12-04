import os

import matplotlib.pyplot as plt

from math import copysign

from functools import partial
from time import time

from compas.geometry import Line, Polyline, Point
from compas.geometry import Translation
from compas.geometry import scale_vector
from compas.utilities import geometric_key

from compas_cem.diagrams import TopologyDiagram

from jax import jit

import jaxopt

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium import EquilibriumModel as FDModel
from jax_fdm.equilibrium import EquilibriumStructure as FDStructure
from jax_fdm.equilibrium import network_updated
from jax_fdm.visualization import Plotter as PlotterFD


VIEW = True
OPTIMIZE = True
OPT_METHOD = "L-BFGS-B"
q0 = 2.0

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------

HERE = os.path.dirname(__file__)
IN_DECK = os.path.abspath(os.path.join(HERE, "data/deck.json"))
IN_NET = os.path.abspath(os.path.join(HERE, "data/net.json"))

# ------------------------------------------------------------------------------
# Load from JSON
# ------------------------------------------------------------------------------

network = FDNetwork.from_json(IN_NET)
network.edges_forcedensities(q0)

# ------------------------------------------------------------------------------
# Copy items from topology into FD network
# ------------------------------------------------------------------------------

topology = TopologyDiagram.from_json(IN_DECK)

nodes_cem_target = []

gkey_key = network.gkey_key()

for node in topology.nodes():
    xyz = topology.node_coordinates(node)
    key = gkey_key.get(geometric_key(xyz))

    if key:
        nodes_cem_target.append(key)
        if network.is_node_support(key):
            network.node_attribute(key, "is_support", False)
        continue

    node_new = network.add_node(attr_dict={k: v for k, v in zip("xyz", xyz)})

    network.node_load(node_new, topology.node_load(node))

    if topology.is_node_support(node):
        network.node_support(node_new)
        continue

    nodes_cem_target.append(node_new)


gkey_key = network.gkey_key()

for edge in topology.edges():
    u, v = (gkey_key.get(geometric_key(topology.node_coordinates(node))) for node in edge)
    network.add_edge(u, v)

    # q0_signed = copysign(q0, topology.edge_force(edge) or topology.edge_length_2(edge))
    q0_signed = copysign(q0 * 2.0, topology.edge_force(edge) or topology.edge_length_2(edge))

    network.edge_forcedensity((u, v), q0_signed)


# ------------------------------------------------------------------------------
# Manipulate topology
# ------------------------------------------------------------------------------

if not OPTIMIZE:
    # network_opt = network
    network_opt = fdm(network)

# ------------------------------------------------------------------------------
# Indices pre calculation
# ------------------------------------------------------------------------------

gkey_key = network.gkey_key()

# add loads
nodes_cem = []  # support nodes in cem world where to get reaction force from
nodes_fdm = []  # nodes in fdm where to apply cem reaction as a load

for node in topology.nodes():
    if topology.is_node_origin(node):
        neighbor = topology.neighbors(node).pop()
        if topology.is_node_support(neighbor):
            # reaction = form_jax_opt.reaction_force(neighbor)
            nodes_cem.append(neighbor)
            key = gkey_key[geometric_key(topology.node_coordinates(node))]
            # network.node_load(key, scale_vector(reaction, -1.))
            nodes_fdm.append(key)

# ------------------------------------------------------------------------------
# Equilibrium models
# ------------------------------------------------------------------------------

model = FDModel.from_network(network)
structure = FDStructure.from_network(network)
fdq = model(structure)

if not OPTIMIZE:
    network_opt = network_updated(network, fdq)

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

if OPTIMIZE:

    indices_xyz_fd = []
    fd_xyz_target = []

    nodes_target = list(network.nodes_where({"is_target": True})) + nodes_cem_target

    for node in nodes_target:
        index = structure.node_index[node]
        indices_xyz_fd.append(index)

        xyz = network.node_coordinates(node)
        fd_xyz_target.append(xyz)

    xyz_fd_target = jnp.asarray(fd_xyz_target)

    # define loss function
    @jit
    def loss_fn(diff_model, static_model):
        """
        A loss function.
        """
        model = eqx.combine(diff_model, static_model)
        eq_state = model(structure)
        xyz_pred = eq_state.xyz[indices_xyz_fd, :]

        return jnp.sum((xyz_pred - xyz_fd_target) ** 2)

    # set tree filtering specification
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(lambda tree: (tree.q), filter_spec, replace=(True))

    # split model into differentiable and static submodels
    diff_model, static_model = eqx.partition(model, filter_spec)

    # bounds_low_compas = jnp.zeros_like(model.forces)
    # bounds_up_compas = jnp.ones_like(model.forces) * jnp.inf

    # bound_low = eqx.tree_at(lambda tree: (tree.forces),
    #                         diff_model,
    #                         replace=(bounds_low_compas))
    # bound_up = eqx.tree_at(lambda tree: (tree.forces),
    #                        diff_model,
    #                        replace=(bounds_up_compas))

    # bounds = (bound_low, bound_up)

    # evaluate loss function at the start
    loss = loss_fn(diff_model, static_model)
    print(f"{loss=}")

    # solve optimization problem with scipy
    print("\n***Optimizing FDM with scipy***")
    optimizer = jaxopt.ScipyMinimize
    # optimizer = jaxopt.ScipyBoundedMinimize

    history = []
    def recorder(xk):
        history.append(xk)

    opt = optimizer(fun=loss_fn,
                    method=OPT_METHOD,
                    jit=True,
                    tol=1e-6,  # 1e-12,
                    maxiter=5000,
                    callback=recorder)

    start = time()
    opt_result = opt.run(diff_model, static_model)

    # opt_result = opt.run(diff_model, bounds, static_model, structure, y)
    print(f"Opt time: {time() - start:.4f} sec")
    diff_model_star, opt_state_star = opt_result
    # print(opt_state_star)

    # evaluate loss function at optimum point
    loss = loss_fn(diff_model_star, static_model)
    print(f"{loss=}")
    print(f"{opt_state_star.iter_num=}")

    # generate optimized compas datastructures
    model_star = eqx.combine(diff_model_star, static_model)
    eqstate_star = model_star(structure)

    network_opt = network_updated(network, eqstate_star)

# ------------------------------------------------------------------------------
# Plott loss function
# ------------------------------------------------------------------------------
#
    print("\nPlotting loss function...")
    plt.figure(figsize=(8, 5))
    start_time = time()

    losses = [loss_fn(h_model, static_model) for h_model in history]

    plt.plot(losses, label="Loss FDM only")
    plt.xlabel("Optimization iterations")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    print(f"Plotting time: {(time() - start_time):.4} seconds")

# ------------------------------------------------------------------------------
# Plotter
# ------------------------------------------------------------------------------

plotter = PlotterFD(figsize=(8, 5), dpi=200)

from compas.datastructures import Network
from compas.colors import Color


nodes, edges = network.to_nodes_and_edges()
_network = Network.from_nodes_and_edges(nodes, edges)
plotter.add(_network,
            show_nodes=False,
            edgewidth=0.5,
            edgecolor={edge: Color.grey() for edge in network.edges()})

for node in nodes_target:
    point = Point(*network.node_coordinates(node))
    plotter.add(point, size=3)


plotter.add(network_opt,
            nodesize=2,
            edgecolor="force",
            show_reactions=False,
            show_loads=False,
            edgewidth=(1., 2.),
            show_edgetext=False,
            show_nodes=True,
            reactionscale=1.0)

# plotter.add(network_opt,
#             nodesize=2,
#             edgecolor="fd",
#             show_reactions=True,
#             show_loads=True,
#             edgewidth=(1., 3.),
#             show_edgetext=True,
#             show_nodes=True,
#             reactionscale=1.0)

# for node in network.nodes():
#     line = Polyline((network.node_coordinates(node), network_opt.node_coordinates(node)))
#     plotter.add(line)

plotter.zoom_extents()
# plotter.save("net_deck_fdm.pdf")
plotter.show()

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
