import os

import matplotlib.pyplot as plt

from time import time

from compas.datastructures import Network
from compas.colors import Color
from compas.geometry import Point
from compas.geometry import scale_vector
from compas.utilities import geometric_key

from compas_cem.diagrams import TopologyDiagram

import jaxopt

from jax import jit
from jax import vmap

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from jax_cem.equilibrium import EquilibriumModel as CEModel
from jax_cem.equilibrium import EquilibriumStructure as CEStructure
from jax_cem.equilibrium import form_from_eqstate

from jax_fdm.datastructures import FDNetwork

from jax_fdm.geometry import closest_point_on_line

from jax_fdm.equilibrium import EquilibriumModel as FDModel
from jax_fdm.equilibrium import EquilibriumStructure as FDStructure
from jax_fdm.equilibrium import network_updated

from jax_fdm.visualization import Viewer as ViewerFD


VIEW = True

RECORD = True
PLOT_LOSS = True
EXPORT_LOSS = True
EXPORT_JSON = True

OPTIMIZE_CEM = True
OPTIMIZE_FDM = True

FIX_INTERFACE = False

q0 = 1.0
qmin, qmax = 1e-3, 30.0
fmin, fmax = -50.0, 50.0

target_length_ratio_fd = 1.0  # 0.9
target_force_fd = 10.0
weight_xyz = 1.0

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------

HERE = os.path.dirname(__file__)
IN_DECK = os.path.abspath(os.path.join(HERE, "data/deck_3d.json"))
IN_NET = os.path.abspath(os.path.join(HERE, "data/net_hexagon_3d.json"))

# ------------------------------------------------------------------------------
# Load from JSON
# ------------------------------------------------------------------------------

network = FDNetwork.from_json(IN_NET)
topology = TopologyDiagram.from_json(IN_DECK)
assert topology.number_of_indirect_deviation_edges() == 0

# ------------------------------------------------------------------------------
# Manipulate topology
# ------------------------------------------------------------------------------

network.edges_forcedensities(q=q0)

# ------------------------------------------------------------------------------
# Equilibrium structs
# ------------------------------------------------------------------------------

ce_structure = CEStructure.from_topology_diagram(topology)
fd_structure = FDStructure.from_network(network)

# ------------------------------------------------------------------------------
# Indices pre calculation
# ------------------------------------------------------------------------------

gkey_key = network.gkey_key()

# add loads
nodes_cem = []  # support nodes in cem world where to get reaction force from
nodes_fdm = []  # nodes in fdm where to apply cem reaction as a load

for node in topology.nodes():
    if topology.is_node_origin(node):
        for neighbor in topology.neighbors(node):

            if not topology.is_node_support(neighbor):
                continue

            key = gkey_key.get(geometric_key(topology.node_coordinates(node)))

            if key is None:
                continue

            # reaction = form_jax_opt.reaction_force(neighbor)
            nodes_cem.append(neighbor)

            # key = gkey_key[geometric_key(topology.node_coordinates(node))]
            # network.node_load(key, scale_vector(reaction, -1.))
            nodes_fdm.append(key)

assert len(nodes_cem) == len(nodes_fdm)

# assumes interface nodes in fdm network are supports
if not FIX_INTERFACE:
    for node in nodes_fdm:
        if network.is_node_support(node):
            network.node_attribute(node, "is_support", False)

indices_cem = []
for node in nodes_cem:
    indices_cem.append(ce_structure.node_index[node])

indices_fdm = []
for node in nodes_fdm:
    indices_fdm.append(fd_structure.node_index[node])

# ------------------------------------------------------------------------------
# Equilibrium models
# ------------------------------------------------------------------------------

ce_model = CEModel.from_topology_diagram(topology)
fd_model = FDModel.from_network(network)

ceq = ce_model(ce_structure)
form_opt = form_from_eqstate(ce_structure, ceq)

fdq = fd_model(fd_structure)
network_opt = network_updated(fd_structure.network, fdq)

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

# ce goals
nodes_ce_xyz_opt = []
for node in topology.nodes():
    if topology.is_node_origin(node):
        continue
    if topology.is_node_support(node):
        neighbor = topology.neighbors(node).pop()
        if topology.is_node_origin(neighbor):
            continue
    nodes_ce_xyz_opt.append(node)

xyz_ce_target = []
indices_ce_xyz_opt = []
for node in nodes_ce_xyz_opt:
    index = ce_structure.node_index[node]
    indices_ce_xyz_opt.append(index)
    xyz_ce_target.append(topology.node_coordinates(node))

xyz_ce_target_copy = xyz_ce_target[:]
xyz_ce_target = jnp.asarray(xyz_ce_target)
lines_ce_target = (xyz_ce_target, xyz_ce_target + jnp.array([0.0, 0.0, 1.0]))

nodes_ce_res_opt = []
indices_ce_res_opt = []
for node in topology.nodes():
    if node in nodes_cem or node in nodes_ce_xyz_opt:
        continue
    if not topology.is_node_support(node):
        continue
    nodes_ce_res_opt.append(node)
    index = ce_structure.node_index[node]
    indices_ce_res_opt.append(index)

# fd goals
indices_fd_res_opt = indices_fdm

indices_fd_xyz_opt = []
xyz_fd_target = []
# for node in network.nodes_where({"is_target": True}):
for node in nodes_fdm:
    index = fd_structure.node_index[node]
    indices_fd_xyz_opt.append(index)
    xyz = network.node_coordinates(node)
    xyz_fd_target.append(xyz)

xyz_fd_target_copy = xyz_fd_target[:]
xyz_fd_target = jnp.asarray(xyz_fd_target)

indices_fd_length_opt = []
fd_lengths_target = []

for edge in network.edges_where({"group": "hangers"}):
    index = fd_structure.edge_index[edge]
    indices_fd_length_opt.append(index)
    length = network.edge_length(*edge)
    fd_lengths_target.append(length)

fd_lengths_target = jnp.asarray(fd_lengths_target)

indices_fd_force_opt = []
for edge in network.edges_where({"group": "cable"}):
    index = fd_structure.edge_index[edge]
    indices_fd_force_opt.append(index)


if OPTIMIZE_CEM:

    # define loss function
    @jit
    def ce_loss_fn(diff_model, static_model):
        """
        A loss function.
        """
        model = eqx.combine(diff_model, static_model)
        ce_eqstate = model(ce_structure)

        # cem loss
        xyz_pred = ce_eqstate.xyz[indices_ce_xyz_opt, :]
        xyz_ce_target = vmap(closest_point_on_line)(xyz_pred, lines_ce_target)
        goal_xyz_ce = jnp.sum((xyz_pred - xyz_ce_target) ** 2)

        residuals_pred_ce = ce_eqstate.reactions[indices_ce_res_opt, :]
        goal_res_ce = jnp.sum((residuals_pred_ce - 0.0) ** 2)

        loss_ce = goal_xyz_ce + goal_res_ce

        return loss_ce

    # set tree filtering specification
    filter_spec = jtu.tree_map(lambda _: False, ce_model)
    filter_spec = eqx.tree_at(lambda tree: (tree.forces), filter_spec, replace=(True))

    # split model into differentiable and static submodels
    ce_diff_model, ce_static_model = eqx.partition(ce_model, filter_spec)

    # define parameter bounds
    bound_low = eqx.tree_at(lambda tree: (tree.forces), ce_diff_model,
                            replace=(jnp.ones_like(ce_model.forces) * fmin))

    bound_up = eqx.tree_at(lambda tree: (tree.forces), ce_diff_model,
                           replace=(jnp.ones_like(ce_model.forces) * fmax)
                           )

    ce_bounds = (bound_low, bound_up)

    # evaluate loss function at the start
    ce_loss = ce_loss_fn(ce_diff_model, ce_static_model)
    print(f"{ce_loss=}")

    # solve optimization problem with scipy
    print("\n***Optimizing CEM alone with scipy***")
    optimizer = jaxopt.ScipyBoundedMinimize
    # optimizer = jaxopt.ScipyMinimize

    history_cem = []

    def recorder_cem(xk):
        history_cem.append(xk)

    opt = optimizer(fun=ce_loss_fn,
                    method="L-BFGS-B",
                    jit=True,
                    tol=1e-6,  # 1e-12,
                    maxiter=500,
                    callback=recorder_cem)

    start = time()
    opt_result = opt.run(ce_diff_model, ce_bounds, ce_static_model)
    # opt_result = opt.run(ce_diff_model, ce_static_model)
    print(f"Opt time: {time() - start:.4f} sec")
    ce_diff_model_star, ce_opt_state_star = opt_result

    # evaluate loss function at optimum point
    ce_loss = ce_loss_fn(ce_diff_model_star, ce_static_model)
    print(f"{ce_loss=}")
    print(f"{ce_opt_state_star.iter_num=}")

    # generate optimized compas datastructures
    ce_model_star = eqx.combine(ce_diff_model_star, ce_static_model)
    ce_eqstate_star = ce_model_star(ce_structure)
    form_opt = form_from_eqstate(ce_structure, ce_eqstate_star)

# ------------------------------------------------------------------------------
# Plott loss function
# ------------------------------------------------------------------------------

if OPTIMIZE_FDM:

    # define loss function
    @jit
    def fd_loss_fn(diff_model, static_model):
        """
        A loss function.
        """
        model = eqx.combine(diff_model, static_model)
        fd_eqstate = model(fd_structure)

        # fd loss
        if FIX_INTERFACE:
            # residuals
            residuals_pred_fd = fd_eqstate.residuals[indices_fd_res_opt, :]
            goal_res_fd = jnp.sum((residuals_pred_fd - 0.0) ** 2)
            goal_interface_fd = goal_res_fd
        else:
            # xyz at interface
            xyz_pred_fd = fd_eqstate.xyz[indices_fd_xyz_opt, :]
            goal_xyz_fd = jnp.sum((xyz_pred_fd - xyz_fd_target) ** 2) * weight_xyz
            goal_interface_fd = goal_xyz_fd

        lengths_pred_fd = fd_eqstate.lengths[indices_fd_length_opt, :].ravel()
        lengths_diff = lengths_pred_fd - fd_lengths_target * target_length_ratio_fd
        goal_length_fd = jnp.sum(lengths_diff ** 2)

        forces_pred_fd = fd_eqstate.forces[indices_fd_force_opt, :].ravel()
        goal_force_fd = jnp.sum((forces_pred_fd - target_force_fd) ** 2)

        loss_fd = goal_interface_fd + goal_length_fd + goal_force_fd

        return loss_fd

    # update applied loads to fd model based on reaction from optimized ce model
    if OPTIMIZE_CEM:
        ce_reactions = ce_eqstate_star.reactions[indices_cem, :]
    else:
        ce_reactions = ceq.reactions[indices_cem, :]

    loads = fd_model.loads.at[indices_fdm, :].set(-ce_reactions)
    fd_model = eqx.tree_at(lambda tree: (tree.loads), fd_model, replace=(loads))

    # set tree filtering specification
    filter_spec = jtu.tree_map(lambda _: False, fd_model)
    filter_spec = eqx.tree_at(lambda tree: (tree.q), filter_spec, replace=(True))

    # split model into differentiable and static submodels
    fd_diff_model, fd_static_model = eqx.partition(fd_model, filter_spec)

    # define parameter bounds
    bound_low = eqx.tree_at(lambda tree: (tree.q), fd_diff_model,
                            replace=(jnp.ones_like(fd_model.q) * qmin))

    bound_up = eqx.tree_at(lambda tree: (tree.q), fd_diff_model,
                           replace=(jnp.ones_like(fd_model.q) * qmax))

    fd_bounds = (bound_low, bound_up)

    # evaluate loss function at the start
    fd_loss = fd_loss_fn(fd_diff_model, fd_static_model)
    print(f"{fd_loss=}")

    # solve optimization problem with scipy
    print("\n***Optimizing FDM alone with scipy***")
    optimizer = jaxopt.ScipyBoundedMinimize
    # optimizer = jaxopt.ScipyMinimize

    history_fdm = []

    def recorder_fdm(xk):
        history_fdm.append(xk)

    opt = optimizer(fun=fd_loss_fn,
                    method="L-BFGS-B",
                    jit=True,
                    tol=1e-6,  # 1e-12,
                    maxiter=500,
                    callback=recorder_fdm)

    start = time()
    opt_result = opt.run(fd_diff_model, fd_bounds, fd_static_model)
    # opt_result = opt.run(fd_diff_model, fd_static_model)
    print(f"Opt time: {time() - start:.4f} sec")
    fd_diff_model_star, fd_opt_state_star = opt_result

    # evaluate loss function at optimum point
    loss_fd = fd_loss_fn(fd_diff_model_star, fd_static_model)
    print(f"{loss_fd=}")
    print(f"{fd_opt_state_star.iter_num=}")

    # generate optimized compas datastructures
    fd_model_star = eqx.combine(fd_diff_model_star, fd_static_model)
    fd_eqstate_star = fd_model_star(fd_structure)
    network_opt = network_updated(fd_structure.network, fd_eqstate_star)

# ------------------------------------------------------------------------------
# Export loss function
# ------------------------------------------------------------------------------

if EXPORT_JSON:

    filepath_deck = os.path.abspath(os.path.join(HERE, "data/deck_3d_separated_opt.json"))
    form_opt.to_json(filepath_deck)
    filepath_net = os.path.abspath(os.path.join(HERE, "data/net_hexagon_3d_separated_opt.json"))
    network_opt.to_json(filepath_net)
    print(f"\nExported optimized deck JSON file to {filepath_deck}")
    print(f"\nExported optimized cablenet JSON file to {filepath_net}")

# ------------------------------------------------------------------------------
# Plot loss function
# ------------------------------------------------------------------------------

if PLOT_LOSS:
    print("\nPlotting loss functions...")
    plt.figure(figsize=(8, 5))
    start_time = time()

    losses_cem = [ce_loss_fn(h_model, ce_static_model) for h_model in history_cem]
    plt.plot(losses_cem, label="Loss CEM")

    losses_fdm = [fd_loss_fn(h_model, fd_static_model) for h_model in history_fdm]
    plt.plot(losses_fdm, label="Loss FDM")

    losses = losses_cem + losses_fdm

    if EXPORT_LOSS:
        filepath = "netdeck_3d_separated_loss.txt"
        with open(filepath, "w") as file:
            for loss in losses:
                file.write(f"{loss}\n")
        print(f"Saved loss history to {filepath}")

    plt.plot(losses, label="Loss CEM + FDM")

    plt.xlabel("Optimization iterations")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid()
    plt.legend()

    print(f"Plotting time: {(time() - start_time):.4} seconds")
    plt.show()

# ------------------------------------------------------------------------------
# Viewer
# ------------------------------------------------------------------------------

if VIEW:
    viewer = ViewerFD(width=1000, height=1000, show_grid=False, viewmode="lighted")

    viewer.view.camera.distance = 28.0
    viewer.view.camera.position = (-13.859, 20.460, 14.682)
    viewer.view.camera.target = (1.008, 4.698, -3.034)
    viewer.view.camera.rotation = (0.885, 0.000, -2.385)

    form_opt_view = form_opt.copy(FDNetwork)

    for node in topology.nodes():
        load = form_opt.node_attributes(node, ["qx", "qy", "qz"])
        form_opt_view.node_load(node, load)

    for node in topology.nodes():
        if topology.is_node_support(node):
            form_opt_view.node_support(node)
            reaction = form_opt.node_attributes(node, ["rx", "ry", "rz"])
            form_opt_view.node_attributes(node, ["rx", "ry", "rz"], scale_vector(reaction, -1.0))

    for edge in topology.edges():
        length = form_opt.edge_length(*edge)
        form_opt_view.edge_attribute(edge, "length", length)
        force = form_opt.edge_force(edge)
        _q = force / length
        form_opt_view.edge_attribute(edge, "q", _q)

    print()
    print("\nCablenet")
    more_stats = {}
    more_stats["CableForce"] = [network_opt.edge_force(edge) for edge in network.edges_where({"group": "cable"})]
    network_opt.print_stats(more_stats)

    print("\nDeck")
    form_opt_view.print_stats()

    _edges = []
    for edge in form_opt_view.edges():
        if topology.is_auxiliary_trail_edge(edge):
            continue
        u, v = edge
        if not form_opt_view.has_edge(u, v):
            u, v = v, u
        _edges.append((u, v))

    _nodes = []
    for node in topology.nodes():
        if node in nodes_ce_res_opt:
            continue
        if topology.is_node_support(node):
            nbr = topology.neighbors(node).pop()
            if topology.is_node_origin(nbr):
                continue

        _nodes.append(node)

    viewer.add(form_opt_view,
               show_nodes=False,
               nodesize=0.1,
               nodecolor=Color.black(),
               nodes=_nodes, # nodes_ce_res_opt,
               edges=_edges,
               edgecolor="force",
               edgewidth=(0.05, 0.1),
               show_reactions=True,
               show_loads=True,
               loadscale=1.5,
               reactionscale=0.4,
               reactioncolor=Color.from_rgb255(0, 150, 10)
               )

    viewer.add(network_opt,
               nodesize=0.05,
               edgecolor="force",
               show_reactions=True,
               show_loads=False,
               edgewidth=(0.05, 0.1),
               show_nodes=True,
               reactionscale=0.40,
               reactioncolor=Color.from_rgb255(0, 150, 10)
               )

    from compas.geometry import Point, Line

    for node, xyz in zip(nodes_ce_xyz_opt, xyz_ce_target_copy):

        pt = Point(*xyz)
        viewer.add(pt, color=Color.orange())
        xyz_pred = form_opt_view.node_coordinates(node)
        viewer.add(Line(xyz_pred, xyz), color=Color.orange())

    for node, xyz in zip(nodes_fdm, xyz_fd_target_copy):

        pt = Point(*xyz)
        viewer.add(pt, color=Color.purple())
        xyz_pred = network_opt.node_coordinates(node)
        viewer.add(Line(xyz_pred, xyz), color=Color.purple())

    viewer.show()
