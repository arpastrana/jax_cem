import os

import matplotlib.pyplot as plt

from functools import partial
from time import time

from compas.datastructures import Network
from compas.colors import Color
from compas.geometry import Point
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

from jax_cem.equilibrium import EquilibriumModel as CEModel
from jax_cem.equilibrium import EquilibriumStructure as CEStructure
from jax_cem.equilibrium import form_from_eqstate

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import EquilibriumModel as FDModel
from jax_fdm.equilibrium import EquilibriumStructure as FDStructure
from jax_fdm.equilibrium import network_updated
from jax_fdm.visualization import Plotter as PlotterFD


VIEW = True
OPTIMIZE = True

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
topology = TopologyDiagram.from_json(IN_DECK)

# ------------------------------------------------------------------------------
# Manipulate topology
# ------------------------------------------------------------------------------

network.edges_forcedensities(q=q0)

# ------------------------------------------------------------------------------
# Manipulate topology
# ------------------------------------------------------------------------------

topology.build_trails(True)

topology.auxiliary_trail_length = 0.1
for edge in topology.auxiliary_trail_edges():
    topology.edge_attribute(edge, "length", 0.1)

print(f"{topology.number_of_indirect_deviation_edges()=}")

# Shift trails sequences
print("\nShifting trail sequences")
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
        neighbor = topology.neighbors(node).pop()
        if topology.is_node_support(neighbor):
            # reaction = form_jax_opt.reaction_force(neighbor)
            nodes_cem.append(neighbor)
            key = gkey_key[geometric_key(topology.node_coordinates(node))]
            # network.node_load(key, scale_vector(reaction, -1.))
            nodes_fdm.append(key)

indices_cem = []
for node in nodes_cem:
    indices_cem.append(ce_structure.node_index[node])

indices_fdm = []
for node in nodes_fdm:
    indices_fdm.append(fd_structure.node_index[node])


# ------------------------------------------------------------------------------
# Custom equilibrium model
# ------------------------------------------------------------------------------

class MixedEquilibriumModel(eqx.Module):
    """
    A custom equilibrium model
    """
    cem: CEModel
    fdm: FDModel

    def __call__(self, ce_structure, fd_structure):
        """
        Compute a state of static equilibrium.
        """
        ce_equilibrium = self.cem(ce_structure, tmax=1)
        ce_reactions = ce_equilibrium.reactions[indices_cem, :]

        loads = self.fdm.loads.at[indices_fdm, :].set(-ce_reactions)
        fdm = eqx.tree_at(lambda tree: (tree.loads), self.fdm, replace=(loads))
        fd_equilibrium = fdm(fd_structure)

        return ce_equilibrium, fd_equilibrium


# ------------------------------------------------------------------------------
# Equilibrium models
# ------------------------------------------------------------------------------

ce_model = CEModel.from_topology_diagram(topology)
fd_model = FDModel.from_network(network)

model = MixedEquilibriumModel(cem=ce_model, fdm=fd_model)
ceq, fdq = model(ce_structure, fd_structure)

form_opt = form_from_eqstate(ce_structure, ceq)
network_opt = network_updated(fd_structure.network, fdq)

# if not OPTIMIZE:
#     form_opt = form_from_eqstate(ce_structure, ceq)
#     # network_opt = network_updated(fd_structure.network, fdq)

#     # print(ce_eqstate_star.forces.shape, ce_structure.deviation_edges.shape)
#     print("initial state")
#     for edge, dev_force in zip(ce_structure.edges, ce_model.forces):
#         if topology.is_deviation_edge(edge):
#             print(edge, dev_force, topology.is_deviation_edge(edge))

#     print("eq state")
#     dev_forces = ceq.forces.ravel() * ce_structure.deviation_edges
#     for edge, dev_force in zip(ce_structure.edges, dev_forces):
#         if topology.is_deviation_edge(edge):
#             print(edge, f"{dev_force=:.2f}", topology.is_deviation_edge(edge))

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

# ce goals
nodes_ce_opt = []
for node in topology.nodes():
    if topology.is_node_origin(node):
        continue
    if topology.is_node_support(node):
        neighbor = topology.neighbors(node).pop()
        if topology.is_node_origin(neighbor):
            continue
    nodes_ce_opt.append(node)

xyz_ce_target = []
indices_ce_opt = []
for node in nodes_ce_opt:
    index = ce_structure.node_index[node]
    indices_ce_opt.append(index)
    xyz_ce_target.append(topology.node_coordinates(node))
xyz_ce_target = jnp.asarray(xyz_ce_target)

# fd goals
indices_fd_res_opt = indices_fdm
indices_fd_xyz_opt = []
fd_xyz_target = []
for node in network.nodes_where({"is_target": True}):
    index = fd_structure.node_index[node]
    indices_fd_xyz_opt.append(index)
    xyz = network.node_coordinates(node)
    fd_xyz_target.append(xyz)

xyz_fd_target = jnp.asarray(fd_xyz_target)

if OPTIMIZE:
    # define loss function
    @jit
    def loss_fn(diff_model, static_model):
        """
        A loss function.
        """
        model = eqx.combine(diff_model, static_model)
        ce_eqstate, fd_eqstate = model(ce_structure, fd_structure)

        # cem loss
        xyz_pred = ce_eqstate.xyz[indices_ce_opt, :]
        loss_ce = jnp.sum((xyz_pred - xyz_ce_target) ** 2)

        # fd loss
        residuals_pred = fd_eqstate.residuals[indices_fd_res_opt, :]
        xyz_pred = fd_eqstate.xyz[indices_fd_xyz_opt, :]
        loss_fd = jnp.sum((residuals_pred - 0.0) ** 2) + jnp.sum((xyz_pred - xyz_fd_target) ** 2)

        return loss_ce + loss_fd

    # set tree filtering specification
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(lambda tree: (tree.cem.forces, tree.fdm.q), filter_spec, replace=(True, True))

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
    print("\n***Optimizing CEM and FDM jointly with scipy***")
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
    ce_eqstate_star, fd_eqstate_star = model_star(ce_structure, fd_structure)

    form_opt = form_from_eqstate(ce_structure, ce_eqstate_star)
    network_opt = network_updated(fd_structure.network, fd_eqstate_star)

# ------------------------------------------------------------------------------
# Plott loss function
# ------------------------------------------------------------------------------

    print("\nPlotting loss function...")
    plt.figure(figsize=(8, 5))
    start_time = time()

    losses = [loss_fn(h_model, static_model) for h_model in history]

    plt.plot(losses, label="Loss CEM+FDM")
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

# plotter.add(topology)
for _network in [network, topology]:
    nodes, edges = _network.to_nodes_and_edges()
    _network = Network.from_nodes_and_edges(nodes, edges)
    plotter.add(_network,
                show_nodes=False,
                edgewidth=0.5,
                edgecolor={edge: Color.grey() for edge in _network.edges()})

for xyzs in xyz_ce_target, fd_xyz_target:
    for xyz in xyzs:
        point = Point(*xyz)
        plotter.add(point, size=3)

plotter.add(form_opt,
            nodesize=2,
            edgewidth=(1., 3.),
            edgetext="key",
            show_edgetext=False,
            show_reactions=False,
            show_loads=False)

plotter.add(network_opt,
            nodesize=2,
            edgecolor="force",
            show_reactions=True,
            show_loads=False,
            edgewidth=(1., 3.),
            show_edgetext=False,
            show_nodes=True,
            reactionscale=1.0)

plotter.zoom_extents()
# plotter.save("net_deck_integrated.pdf")
plotter.show()
