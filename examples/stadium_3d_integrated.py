import os

import matplotlib.pyplot as plt

from time import time

from compas.colors import Color
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

OPTIMIZE = True

PLOT_LOSS = True
EXPORT_LOSS = False

EXPORT_JSON = False

q0 = 2.0
qmin, qmax = 1e-3, 30.0
fmin, fmax = -50.0, 50.0
fmin_cable = 1e-2
fmax_tie = -1e-2
zmin = 0.0
zmax = 0.5
pz_spoke = -0.1

# weights ce
weight_xyz_ce = 1.0
weight_residual = 5.0

# weights fd
weight_length = 1.0
target_length_ratio_fd = 0.8

weight_xyz_fd = 0.0

weight_force = 1.0
target_force_fd = 0.2

weight_reg_fd = 1e-2

# weights ce spoke
weight_xyz_ce_spoke = 0.1

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------

HERE = os.path.dirname(__file__)
IN_NET = os.path.abspath(os.path.join(HERE, "data/stadium_cablenet.json"))
IN_ARCH = os.path.abspath(os.path.join(HERE, "data/stadium_arch.json"))
IN_SPOKE = os.path.abspath(os.path.join(HERE, "data/stadium_spoke.json"))

# ------------------------------------------------------------------------------
# Load from JSON
# ------------------------------------------------------------------------------

network = FDNetwork.from_json(IN_NET)
topology = TopologyDiagram.from_json(IN_ARCH)
assert topology.number_of_indirect_deviation_edges() == 0
topology_spoke = TopologyDiagram.from_json(IN_SPOKE)
assert topology_spoke.number_of_indirect_deviation_edges() == 0

# ------------------------------------------------------------------------------
# Manipulate topology
# ------------------------------------------------------------------------------

network.edges_forcedensities(q=q0)

# ------------------------------------------------------------------------------
# Equilibrium structs
# ------------------------------------------------------------------------------

ce_structure = CEStructure.from_topology_diagram(topology)
ce_spoke_structure = CEStructure.from_topology_diagram(topology_spoke)
fd_structure = FDStructure.from_network(network)

# ------------------------------------------------------------------------------
# Indices pre calculation (arch and cablenet)
# ------------------------------------------------------------------------------

# add loads
nodes_fdm = []  # support nodes in fdm world where to get reaction force from
nodes_cem = []  # nodes in cem arch where to apply cem reaction as a load

gkey_key = network.gkey_key()

# search for interface nodes in topology diagram
for node in topology.nodes_where({"is_interface": True}):
    key = gkey_key.get(geometric_key(topology.node_coordinates(node)))

    if key is None:
        continue

    nodes_cem.append(node)
    nodes_fdm.append(key)

assert len(nodes_cem) == len(nodes_fdm)

indices_cem = []
for node in nodes_cem:
    indices_cem.append(ce_structure.node_index[node])

indices_fdm = []
for node in nodes_fdm:
    indices_fdm.append(fd_structure.node_index[node])

# ------------------------------------------------------------------------------
# Indices pre calculation (spoke wheel and cablenet)
# ------------------------------------------------------------------------------

# add loads
nodes_spoke_fdm = []  # support nodes in fdm world where to get reaction force from
nodes_spoke_cem = []  # nodes in cem arch where to apply cem reaction as a load

# search for interface nodes in topology diagram
for node in topology_spoke.nodes_where({"is_interface": True}):
    key = gkey_key.get(geometric_key(topology_spoke.node_coordinates(node)))

    if key is None:
        continue

    nodes_spoke_cem.append(node)
    nodes_spoke_fdm.append(key)

assert len(nodes_spoke_cem) == len(nodes_spoke_fdm)

indices_spoke_cem = []
for node in nodes_spoke_cem:
    indices_spoke_cem.append(ce_spoke_structure.node_index[node])

indices_spoke_fdm = []
for node in nodes_spoke_fdm:
    indices_spoke_fdm.append(fd_structure.node_index[node])

# ------------------------------------------------------------------------------
# Loads
# ------------------------------------------------------------------------------

load_spoke_weight = jnp.array([0.0, 0.0, pz_spoke])

# ------------------------------------------------------------------------------
# Mixed equilibrium model
# ------------------------------------------------------------------------------

class MixedEquilibriumModel(eqx.Module):
    """
    A custom equilibrium model
    """
    cem: CEModel
    fdm: FDModel
    cem2: CEModel

    def __call__(self, ce_structure, fd_structure, ce_spoke_structure):
        """
        Compute a state of static equilibrium.
        """
        fd_equilibrium = self.fdm(fd_structure)
        fd_reactions = fd_equilibrium.residuals[indices_fdm, :]
        fd_supports = fd_equilibrium.xyz[indices_fdm, :]
        fd_spoke_reactions = fd_equilibrium.residuals[indices_spoke_fdm, :]

        loads = self.cem.loads.at[indices_cem, :].set(fd_reactions)
        xyz = self.cem.xyz.at[indices_cem, :].set(fd_supports)
        cem = eqx.tree_at(lambda tree: (tree.loads, tree.xyz),
                          self.cem,
                          replace=(loads, xyz))
        ce_equilibrium = cem(ce_structure)

        # fd_spoke_reactions = fd_spoke_reactions + load_spoke_weight
        loads = self.cem2.loads.at[indices_spoke_cem, :].set(fd_spoke_reactions)
        cem2 = eqx.tree_at(lambda tree: (tree.loads), self.cem2, replace=(loads))
        ce_spoke_equilibrium = cem2(ce_spoke_structure)

        return ce_equilibrium, fd_equilibrium, ce_spoke_equilibrium

# ------------------------------------------------------------------------------
# Equilibrium models
# ------------------------------------------------------------------------------

ce_model = CEModel.from_topology_diagram(topology)
fd_model = FDModel.from_network(network)
ce_spoke_model = CEModel.from_topology_diagram(topology_spoke)

model = MixedEquilibriumModel(cem=ce_model, fdm=fd_model, cem2=ce_spoke_model)
ceq, fdq, ceq_spoke = model(ce_structure, fd_structure, ce_spoke_structure)

form_opt = form_from_eqstate(ce_structure, ceq)
network_opt = network_updated(fd_structure.network, fdq)
form_spoke_opt = form_from_eqstate(ce_spoke_structure, ceq_spoke)

print()
print("\nCablenet")
more_stats = {}
more_stats["CForce"] = [network_opt.edge_force(edge) for edge in network.edges_where({"group": "cable"})]
network_opt.print_stats(more_stats)

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
xyz_ce_target = jnp.asarray(xyz_ce_target)

nodes_ce_res_opt = []
indices_ce_res_opt = []
for node in topology.nodes_where({"interface": 1}):
    for neighbor in topology.neighbors(node):
        if topology.is_node_support(neighbor):
            nodes_ce_res_opt.append(neighbor)
            index = ce_structure.node_index[neighbor]
            indices_ce_res_opt.append(index)

# fd goals
# xyz
nodes_cable = []
for edge in network.edges_where({"group": "cable"}):
    nodes_cable.extend(edge)
nodes_cable = list(set(nodes_cable))

fd_xyz_target = []
indices_fd_xyz_opt = []
for node in nodes_cable:
    index = fd_structure.node_index[node]
    indices_fd_xyz_opt.append(index)
    xyz = network.node_coordinates(node)
    fd_xyz_target.append(xyz)

xyz_fd_target = jnp.asarray(fd_xyz_target)
lines_fd_target = (xyz_fd_target, xyz_fd_target + jnp.array([0.0, 0.0, 1.0]))

# lengths
indices_fd_length_opt = []
fd_lengths_target = []
# cablenet
for edge in network.edges_where({"group": "net"}):
    index = fd_structure.edge_index[edge]
    indices_fd_length_opt.append(index)
    length = network.edge_length(*edge)
    fd_lengths_target.append(length)

# cable ring
for edge in network.edges_where({"group": "cable"}):
    index = fd_structure.edge_index[edge]
    indices_fd_length_opt.append(index)
    length = network.edge_length(*edge)
    fd_lengths_target.append(length)

fd_lengths_target = jnp.asarray(fd_lengths_target)

# cable forces
indices_fd_force_opt = []
for edge in network.edges_where({"group": "cable"}):
    index = fd_structure.edge_index[edge]
    indices_fd_force_opt.append(index)

# ce spoke goals
nodes_ce_spoke_xyz_opt = []
for node in topology_spoke.support_nodes():
    nodes_ce_spoke_xyz_opt.append(node)

xyz_ce_spoke_target = []
indices_ce_spoke_xyz_opt = []
for node in nodes_ce_spoke_xyz_opt:
    index = ce_spoke_structure.node_index[node]
    indices_ce_spoke_xyz_opt.append(index)
    xyz_ce_spoke_target.append(topology_spoke.node_coordinates(node))

xyz_ce_spoke_target = jnp.asarray(xyz_ce_spoke_target)
# print(len(nodes_ce_spoke_xyz_opt), len(indices_ce_spoke_xyz_opt))

if OPTIMIZE:

    # define loss function
    @jit
    def loss_fn(diff_model, static_model):
        """
        A loss function.
        """
        model = eqx.combine(diff_model, static_model)
        ce_eqstate, fd_eqstate, ce_spoke_eqstate = model(ce_structure, fd_structure, ce_spoke_structure)

        # cem loss
        xyz_pred = ce_eqstate.xyz[indices_ce_xyz_opt, :]
        goal_xyz_ce = jnp.mean((xyz_pred - xyz_ce_target) ** 2)

        residuals_pred_ce = ce_eqstate.reactions[indices_ce_res_opt, :]
        goal_res_ce = jnp.mean((residuals_pred_ce - 0.0) ** 2)

        loss_ce = goal_xyz_ce * weight_xyz_ce + goal_res_ce * weight_residual

        # fd loss
        xyz_pred_fd = fd_eqstate.xyz[indices_fd_xyz_opt, :]
        xyz_fd_target = vmap(closest_point_on_line)(xyz_pred_fd, lines_fd_target)
        goal_xyz_fd = jnp.mean((xyz_pred_fd - xyz_fd_target) ** 2)

        # cablent lengths
        lengths_pred_fd = fd_eqstate.lengths[indices_fd_length_opt, :].ravel()
        lengths_diff = lengths_pred_fd - fd_lengths_target * target_length_ratio_fd
        goal_length_fd = jnp.mean(lengths_diff ** 2)

        # lengths_pred_fd = fd_eqstate.lengths[indices_fd_length_opt, :].ravel()
        # goal_length_fd = jnp.var(lengths_pred_fd) / jnp.mean(lengths_pred_fd)

        # cable forces
        forces_pred_fd = fd_eqstate.forces[indices_fd_force_opt, :].ravel()
        goal_force_fd = jnp.mean((forces_pred_fd - target_force_fd) ** 2)

        # fd regularizer
        goal_reg_fd = jnp.mean(jnp.square(diff_model.fdm.q))

        # loss_fd = goal_force_fd * weight_force + goal_length_fd * weight_length
        loss_fd = goal_length_fd * weight_length + goal_xyz_fd * weight_xyz_fd + goal_force_fd * weight_force + goal_reg_fd * weight_reg_fd

        # cem spoke loss
        xyz_spoke_pred = ce_spoke_eqstate.xyz[indices_ce_spoke_xyz_opt, :]
        goal_xyz_ce_spoke = jnp.mean((xyz_spoke_pred - xyz_ce_spoke_target) ** 2)
        loss_ce_spoke = goal_xyz_ce_spoke * weight_xyz_ce_spoke

        return loss_ce + loss_fd + loss_ce_spoke

    # set tree filtering specification
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(lambda tree: (tree.cem.forces,
                                            tree.fdm.q,
                                            tree.fdm.xyz_fixed,
                                            tree.cem2.forces
                                            ),
                              filter_spec,
                              replace=(True, True, True, True))

    # split model into differentiable and static submodels
    diff_model, static_model = eqx.partition(model, filter_spec)

    # define parameter bounds
    cem_bound_low = []
    for edge in topology.edges():
        if topology.edge_attribute(edge, "group") == "cable":
            cem_bound_low.append(fmin_cable)
        else:
            cem_bound_low.append(fmin)
    cem_bound_low = jnp.array(cem_bound_low)

    cem_bound_up = []
    for edge in topology.edges():
        if topology.edge_attribute(edge, "group") == "tie":
            cem_bound_up.append(fmax_tie)
        else:
            cem_bound_up.append(fmax)
    cem_bound_up = jnp.array(cem_bound_up)

    fdm_xyz_fixed_bound_low = []
    fdm_xyz_fixed_bound_up = []
    for node in network.nodes_supports():
        x, y, z = network.node_coordinates(node)
        if node in nodes_fdm:
            fdm_xyz_fixed_bound_low.append([x, y, z - zmin])
            fdm_xyz_fixed_bound_up.append([x, y, z + zmax])
        else:
            fdm_xyz_fixed_bound_low.append([x, y, z])
            fdm_xyz_fixed_bound_up.append([x, y, z])

    fdm_xyz_fixed_bound_low = jnp.array(fdm_xyz_fixed_bound_low)
    fdm_xyz_fixed_bound_up = jnp.array(fdm_xyz_fixed_bound_up)

    bound_low = eqx.tree_at(lambda tree: (tree.cem.forces,
                                          tree.fdm.q,
                                          tree.fdm.xyz_fixed,
                                          tree.cem2.forces),
                            diff_model,
                            replace=(cem_bound_low,
                                     jnp.ones_like(model.fdm.q) * qmin,
                                     fdm_xyz_fixed_bound_low,
                                     jnp.ones_like(model.cem2.forces) * fmin)
                            )

    bound_up = eqx.tree_at(lambda tree: (tree.cem.forces,
                                         tree.fdm.q,
                                         tree.fdm.xyz_fixed,
                                         tree.cem2.forces),
                           diff_model,
                           replace=(cem_bound_up,
                                    jnp.ones_like(model.fdm.q) * qmax,
                                    fdm_xyz_fixed_bound_up,
                                    jnp.ones_like(model.cem2.forces) * fmax)
                           )

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
    ce_eqstate_star, fd_eqstate_star, ce_spoke_eqstate_star = model_star(ce_structure, fd_structure, ce_spoke_structure)

    form_opt = form_from_eqstate(ce_structure, ce_eqstate_star)
    network_opt = network_updated(fd_structure.network, fd_eqstate_star)
    form_spoke_opt = form_from_eqstate(ce_spoke_structure, ce_spoke_eqstate_star)

# ------------------------------------------------------------------------------
# Plot loss function
# ------------------------------------------------------------------------------

if OPTIMIZE and EXPORT_JSON:

    filepath_arch = os.path.abspath(os.path.join(HERE, f"data/stadium_arch_opt.json"))
    form_opt.to_json(filepath_arch)
    filepath_net = os.path.abspath(os.path.join(HERE, f"data/stadium_cablenet_opt.json"))
    network_opt.to_json(filepath_net)
    filepath_spoke = os.path.abspath(os.path.join(HERE, f"data/stadium_spoke_opt.json"))
    form_spoke_opt.to_json(filepath_spoke)
    print(f"\nExported optimized arch JSON file to {filepath_arch}")
    print(f"\nExported optimized cablenet JSON file to {filepath_net}")
    print(f"\nExported optimized spoke JSON file to {filepath_spoke}")

# ------------------------------------------------------------------------------
# Plot loss function
# ------------------------------------------------------------------------------

if OPTIMIZE and PLOT_LOSS:

    print("\nPlotting loss function...")
    plt.figure(figsize=(8, 5))
    start_time = time()

    losses = [loss_fn(h_model, static_model) for h_model in history]

    if EXPORT_LOSS:
        filepath = f"stadium_integrated_targetforce{int(target_force_fd)}_loss.txt"
        with open(filepath, "w") as file:
            for loss in losses:
                file.write(f"{loss}\n")
        print(f"Saved loss history to {filepath}")

    plt.plot(losses, label="Loss MEM")
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

    # arch
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

    # spoke wheel
    form_spoke_opt_view = form_spoke_opt.copy(FDNetwork)
    for node in topology_spoke.nodes():
        load = form_spoke_opt.node_attributes(node, ["qx", "qy", "qz"])
        form_spoke_opt_view.node_load(node, load)

    for node in topology_spoke.nodes():
        if topology_spoke.is_node_support(node):
            form_spoke_opt_view.node_support(node)
            reaction = form_spoke_opt.node_attributes(node, ["rx", "ry", "rz"])
            form_spoke_opt_view.node_attributes(node, ["rx", "ry", "rz"], scale_vector(reaction, -1.0))

    for edge in topology_spoke.edges():
        length = form_spoke_opt.edge_length(*edge)
        form_spoke_opt_view.edge_attribute(edge, "length", length)
        force = form_spoke_opt.edge_force(edge)
        _q = force / length
        form_spoke_opt_view.edge_attribute(edge, "q", _q)

    print()
    print("\nCablenet")
    more_stats = {}
    more_stats["CForce"] = [network_opt.edge_force(edge) for edge in network.edges_where({"group": "cable"})]
    network_opt.print_stats(more_stats)

    print("\nArch")
    form_opt_view.print_stats()

    print("\nSpoke")
    form_spoke_opt_view.print_stats()

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

    # _nodes = nodes_ce_res_opt
    # _nodes = nodes_ce_xyz_opt
    _nodes = None
    _edges = list(form_opt_view.edges())

    viewer.add(form_opt_view,
               show_nodes=False,
               nodesize=0.1, # 0.1,
               nodecolor=Color.black(),
               # nodes=_nodes, # nodes_ce_res_opt, # , _nodes
               edges=_edges,
               edgecolor="force",
               edgewidth=(0.01, 0.1),
               show_reactions=True,
               show_loads=True,
               loadscale=1.0,
               reactionscale=1.0,
               reactioncolor=Color.pink(), # Color.from_rgb255(0, 150, 10)
               )

    # _nodes = nodes_cable
    # _nodes = None
    viewer.add(network_opt,
               show_nodes=False,
               nodesize=0.1,
               nodecolor=Color.purple(),
               edgecolor="force",
               show_reactions=True,
               show_loads=False,
               edgewidth=(0.01, 0.1),
               reactionscale=1.0,
               reactioncolor=Color(0.1, 0.1, 0.1), # Color.from_rgb255(0, 150, 10)
               )

    viewer.add(form_spoke_opt_view,
               show_nodes=False,
               nodesize=0.1, # 0.1,
               nodecolor=Color.black(),
               # nodes=_nodes, # nodes_ce_res_opt, # , _nodes
               # edges=_edges,
               edgecolor="force",
               edgewidth=(0.01, 0.1),
               show_reactions=True,
               show_loads=True,
               loadscale=1.0,
               reactionscale=1.0,
               reactioncolor=Color.orange(), # Color.from_rgb255(0, 150, 10)
               )

    # viewer.add(network, as_wireframe=True, show_points=False)
    # topology_view = topology.copy(FDNetwork)
    # viewer.add(topology_view, as_wireframe=True, show_points=False)
    # topology_spoke_view = topology_spoke.copy(FDNetwork)
    # viewer.add(topology_spoke_view, as_wireframe=True, show_points=False)

    viewer.show()
