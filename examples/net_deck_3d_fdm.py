import os

import matplotlib.pyplot as plt

from time import time

from compas.datastructures import Network
from compas.colors import Color
from compas.utilities import geometric_key

from compas_cem.diagrams import TopologyDiagram
from compas_cem.equilibrium import static_equilibrium

import jaxopt

from jax import jit
from jax import vmap

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from jax_fdm.datastructures import FDNetwork

from jax_fdm.geometry import closest_point_on_line

from jax_fdm.equilibrium import EquilibriumModel as FDModel
from jax_fdm.equilibrium import EquilibriumStructure as FDStructure
from jax_fdm.equilibrium import network_updated

from jax_fdm.visualization import Viewer as ViewerFD


VIEW = True
OPTIMIZE = True  #True

q0 = 1.0
qmin, qmax = 1e-3, jnp.inf  # 0, 30 cablenet

target_length_ratio_fd = 1.0
target_force_fd = 8.0

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
network.edges_forcedensities(q=q0)

# ------------------------------------------------------------------------------
# Manipulate topology
# ------------------------------------------------------------------------------

topology = TopologyDiagram.from_json(IN_DECK)
assert topology.number_of_indirect_deviation_edges() == 0

form = static_equilibrium(topology)

# delete auxiliary trails edges
deletable_nodes = []
for edge in topology.edges():
    if topology.is_auxiliary_trail_edge(edge):
        for node in edge:
            if topology.is_node_support(node):
                deletable_nodes.append(node)

for node in deletable_nodes:
    topology.delete_node(node)

# ------------------------------------------------------------------------------
# Copy items from topology into FD network
# ------------------------------------------------------------------------------

# copy nodes
nodes_cem_target = []
gkey_key = network.gkey_key()

for node in topology.nodes():
    xyz = topology.node_coordinates(node)
    key = gkey_key.get(geometric_key(xyz))

    # node exists in fdm cablenet
    if key:
        nodes_cem_target.append(key)

        _load = topology.node_load(node)
        network.node_load(key, _load)
        # remove supports from cablenet at interface with deck
        if network.is_node_support(key):
            network.node_attribute(key, "is_support", False)
        continue

    # node does not exist, then add
    node_new = network.add_node(attr_dict={k: v for k, v in zip("xyz", xyz)})
    _load = topology.node_load(node)
    network.node_load(node_new, _load)

    if topology.is_node_support(node):
        network.node_support(node_new)
        continue

    nodes_cem_target.append(node_new)

# copy edges
gkey_key = network.gkey_key()
for edge in topology.edges():
    u, v = (gkey_key.get(geometric_key(topology.node_coordinates(node))) for node in edge)

    network.add_edge(u, v)

    q0_ce = form.edge_force(edge) / form.edge_length(*edge)

    network.edge_forcedensity((u, v), q0_ce)

# ------------------------------------------------------------------------------
# Equilibrium model
# ------------------------------------------------------------------------------

model = FDModel.from_network(network)
fd_structure = FDStructure.from_network(network)
fdq = model(fd_structure)
network_opt = network_updated(fd_structure.network, fdq)
# network_opt = network

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

# xyz goals
nodes_xyz_opt = []
indices_xyz_opt = []
xyz_target = []

nodes_xyz_line_opt = []
indices_xyz_line_opt = []
xyz_line_target = []

gkey_key = topology.gkey_key()
for node in nodes_cem_target:

    xyz = network.node_coordinates(node)
    key = gkey_key.get(geometric_key(xyz))

    if topology.is_node_origin(key):
        xyz_target.append(topology.node_coordinates(key))

        nodes_xyz_opt.append(node)
        index = fd_structure.node_index[node]
        indices_xyz_opt.append(index)
    else:
        xyz_line_target.append(topology.node_coordinates(key))

        nodes_xyz_line_opt.append(node)
        index = fd_structure.node_index[node]
        indices_xyz_line_opt.append(index)

xyz_target_copy = xyz_target[:]
xyz_target = jnp.asarray(xyz_target)
xyz_line_target = jnp.asarray(xyz_line_target)
lines_target = (xyz_line_target, xyz_line_target + jnp.array([0.0, 0.0, 1.0]))

# length goals
indices_fd_length_opt = []
fd_lengths_target = []

for edge in network.edges_where({"group": "hangers"}):
    index = fd_structure.edge_index[edge]
    indices_fd_length_opt.append(index)
    length = network.edge_length(*edge)
    fd_lengths_target.append(length)

fd_lengths_target = jnp.asarray(fd_lengths_target)

# force goals
indices_fd_force_opt = []
for edge in network.edges_where({"group": "cable"}):
    index = fd_structure.edge_index[edge]
    indices_fd_force_opt.append(index)

if OPTIMIZE:

    # define loss function
    @jit
    def loss_fn(diff_model, static_model):
        """
        A loss function.
        """
        model = eqx.combine(diff_model, static_model)
        fd_eqstate = model(fd_structure)

        # goal xyz
        xyz_pred = fd_eqstate.xyz[indices_xyz_opt, :]
        goal_xyz_fd = jnp.sum((xyz_pred - xyz_target) ** 2)

        # goal xyz line
        xyz_line_pred = fd_eqstate.xyz[indices_xyz_line_opt, :]
        xyz_line_target = vmap(closest_point_on_line)(xyz_line_pred, lines_target)
        goal_xyz_line_fd = jnp.sum((xyz_line_pred - xyz_line_target) ** 2)

        # goal length
        lengths_pred_fd = fd_eqstate.lengths[indices_fd_length_opt, :].ravel()
        lengths_diff = lengths_pred_fd - fd_lengths_target * target_length_ratio_fd
        goal_length_fd = jnp.sum(lengths_diff ** 2)

        # goal force
        forces_pred_fd = fd_eqstate.forces[indices_fd_force_opt, :].ravel()
        goal_force_fd = jnp.sum((forces_pred_fd - target_force_fd) ** 2)

        return goal_xyz_fd + goal_xyz_line_fd + goal_length_fd + goal_force_fd

    # set tree filtering specification
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(lambda tree: (tree.q), filter_spec, replace=(True))

    # split model into differentiable and static submodels
    diff_model, static_model = eqx.partition(model, filter_spec)

    # define parameter bounds
    blow = []
    blup = []
    gkey_key = topology.gkey_key()
    for edge in network.edges():
        edge_topo = [gkey_key.get(geometric_key(network.node_coordinates(node))) for node in edge]
        # if edge is in topology, can take tension and compression:
        if all(edge_topo):
            blow.append(-qmax)
            blup.append(qmax)
        else:
            blow.append(qmin)
            blup.append(qmax)

    blow = jnp.asarray(blow)
    blup = jnp.asarray(blup)
    bound_low = eqx.tree_at(lambda tree: (tree.q), diff_model, replace=(blow))
    bound_up = eqx.tree_at(lambda tree: (tree.q), diff_model, replace=(blup))
    bounds = (bound_low, bound_up)

    # evaluate loss function at the start
    loss = loss_fn(diff_model, static_model)
    print(f"{loss=}")

    # solve optimization problem with scipy
    print("\n***Optimizing CEM and FDM jointly with scipy***")
    optimizer = jaxopt.ScipyBoundedMinimize

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
    opt_result = opt.run(diff_model, bounds, static_model)
    print(f"Opt time: {time() - start:.4f} sec")
    diff_model_star, opt_state_star = opt_result

    # evaluate loss function at optimum point
    loss = loss_fn(diff_model_star, static_model)
    print(f"{loss=}")
    print(f"{opt_state_star.iter_num=}")

    # generate optimized compas datastructures
    model_star = eqx.combine(diff_model_star, static_model)
    fd_eqstate_star = model_star(fd_structure)
    network_opt = network_updated(fd_structure.network, fd_eqstate_star)

# ------------------------------------------------------------------------------
# Plott loss function
# ------------------------------------------------------------------------------

    print("\nPlotting loss function...")
    plt.figure(figsize=(8, 5))
    start_time = time()

    losses = [loss_fn(h_model, static_model) for h_model in history]

    plt.plot(losses, label="Loss FDM")
    plt.xlabel("Optimization iterations")
    plt.ylabel("Loss")
    plt.yscale("log")
    # plt.xscale("log")
    plt.grid()
    plt.legend()
    print(f"Plotting time: {(time() - start_time):.4} seconds")
    plt.show()

# ------------------------------------------------------------------------------
# Viewer
# ------------------------------------------------------------------------------

if VIEW:
    viewer = ViewerFD(width=900, height=900, show_grid=False)

    viewer.view.camera.distance = 28.0
    viewer.view.camera.position = (-13.859, 20.460, 14.682)
    viewer.view.camera.target = (1.008, 4.698, -3.034)
    viewer.view.camera.rotation = (0.885, 0.000, -2.385)

    for _network in [network]:
        nodes, edges = _network.to_nodes_and_edges()
        _network = Network.from_nodes_and_edges(nodes, edges)
        viewer.add(_network,
                   show_points=False,
                   linewidth=0.5,
                   linecolor=Color.grey(),
                   )

    print("\nCablenet with deck")
    network_opt.print_stats()

    viewer.add(network_opt,
               edgecolor="force",
               show_reactions=True,
               show_loads=True,
               edgewidth=(0.01, 0.05),
               reactionscale=1.0,
               loadscale=1.0,
               show_nodes=True,
               nodecolor=Color.black(),
               nodesize=0.25,
               nodes=nodes_xyz_opt,
               )

    from compas.geometry import Point

    for xyz in xyz_target_copy:
        pt = Point(*xyz)
        viewer.add(pt, color=Color.lime())

    viewer.show()
