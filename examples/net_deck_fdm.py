import os

import matplotlib.pyplot as plt

from math import copysign
from math import fabs

from time import time

from compas.datastructures import Network
from compas.colors import Color
from compas.geometry import Line, Point
from compas.utilities import geometric_key
from compas.utilities import remap_values

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


OPTIMIZE = True
PLOT_LOSS = True
PLOT = True
PLOT_SAVE = True
EXPORT_LOSS = True

q0 = 1.5
target_length_ratio_fd = 1.0
target_force_fd = 5

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
        # NOTE: If we don't add the nodes at the interface between cablenet and deck
        # as targets, and don't add a target force on the cable,
        # then the optimization problem DOES converge
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

    q0_signed = copysign(q0, topology.edge_force(edge) or topology.edge_length_2(edge))

    # NOTE: problem does converge if we double the starting FD for CEM components!
    # This suggests fdm only is pretty sensitive to initialization
    # q0_signed = copysign(q0 * 2.0, topology.edge_force(edge) or topology.edge_length_2(edge))

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

    # nodes_target = list(network.nodes_where({"is_target": True})) + nodes_cem_target
    nodes_target = nodes_cem_target

    for node in nodes_target:
        index = structure.node_index[node]
        indices_xyz_fd.append(index)

        xyz = network.node_coordinates(node)
        fd_xyz_target.append(xyz)

    xyz_fd_target = jnp.asarray(fd_xyz_target)

    indices_fd_length_opt = []
    fd_lengths_target = []

    for edge in network.edges_where({"group": "hangers"}):
        index = structure.edge_index[edge]
        indices_fd_length_opt.append(index)
        length = network.edge_length(*edge)
        fd_lengths_target.append(length)

    fd_lengths_target = jnp.asarray(fd_lengths_target)

    indices_fd_force_opt = []
    for edge in network.edges_where({"group": "cable"}):
        index = structure.edge_index[edge]
        indices_fd_force_opt.append(index)

    # define loss function
    @jit
    def loss_fn(diff_model, static_model):
        """
        A loss function.
        """
        model = eqx.combine(diff_model, static_model)
        eq_state = model(structure)
        xyz_pred = eq_state.xyz[indices_xyz_fd, :]

        goal_xyz = jnp.sum((xyz_pred - xyz_fd_target) ** 2)

        lengths_pred_fd = eq_state.lengths[indices_fd_length_opt, :].ravel()
        lengths_diff = lengths_pred_fd - fd_lengths_target * target_length_ratio_fd
        goal_length_fd = jnp.sum(lengths_diff ** 2)

        forces_pred_fd = eq_state.forces[indices_fd_force_opt, :].ravel()
        goal_force_fd = jnp.sum((forces_pred_fd - target_force_fd) ** 2)

        return goal_xyz + goal_force_fd + goal_length_fd

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
                    method="L-BFGS-B",
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

    more_stats = {}
    more_stats["CabForce"] = [network_opt.edge_force(edge) for edge in network.edges_where({"group": "cable"})]
    network_opt.print_stats(more_stats)

# ------------------------------------------------------------------------------
# Export loss function
# ------------------------------------------------------------------------------

if EXPORT_LOSS:
    losses = [loss_fn(h_model, static_model) for h_model in history]
    with open("netdeck_fdm_loss.txt", "w") as file:
        for loss in losses:
            file.write(f"{loss}\n")

# ------------------------------------------------------------------------------
# Plot loss function
# ------------------------------------------------------------------------------

if PLOT_LOSS:

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

if PLOT:

    plotter = PlotterFD(figsize=(8, 5), dpi=200)
    for edge in network.edges():
        plotter.add(Line(*network.edge_coordinates(*edge)),
                    draw_as_segment=True,
                    linestyle="dashed",
                    color=Color.grey(),
                    linewidth=0.5)

    # manually calculate edge widths
    width = (1.0, 3.0)
    width_min, width_max = width
    forces_network = [fabs(network_opt.edge_force(edge)) for edge in network_opt.edges()]
    forces = forces_network
    force_max = 5.0  # max(forces)
    force_min = 0.0  # min(forces)
    widths_network = remap_values(forces_network, width_min, width_max, force_min, force_max)

    plotter.add(network_opt,
                nodesize=20,
                edgecolor="force",
                show_reactions=False,
                show_loads=False,
                edgewidth={edge: width for edge, width in zip(network_opt.edges(), widths_network)},
                show_edgetext=False,
                show_nodes=True,
                reactioncolor=Color.from_rgb255(0, 150, 10),
                reactionscale=-0.5,
                sizepolicy="absolute"
                )

    plotter.zoom_extents()

    if PLOT_SAVE:
        filename = "netdeck_fdm.pdf"
        plotter.save(filename, transparent=True)
        print(f"Saved pdf to {filename}")

    plotter.show()
