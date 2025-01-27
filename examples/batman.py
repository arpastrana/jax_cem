import os

import matplotlib.pyplot as plt

from time import time

from compas.datastructures import Mesh

from jax import jit
from jax import vmap

import jaxopt

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import EquilibriumModel as FDModel
from jax_fdm.equilibrium import EquilibriumStructure as FDStructure
from jax_fdm.equilibrium import network_updated
from jax_fdm.visualization import Plotter as PlotterFD


OPTIMIZE = True
PLOT = False
OPT_METHOD = "L-BFGS-B"

q0 = 2.0
qmin, qmax = 1e-3, 100.0

length_factor = 0.75
length_target = 0.4

alpha_length = 1.0
alpha_fairness = 0.01  # 0.01


# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------

HERE = os.path.dirname(__file__)
IN_NET = os.path.abspath(os.path.join(HERE, "data/batman.json"))

# ------------------------------------------------------------------------------
# Load mesh from JSON
# ------------------------------------------------------------------------------

mesh = Mesh.from_json(IN_NET)
supports = [vkey for vkey in mesh.vertices_on_boundary() if len(mesh.vertex_neighbors(vkey)) > 3]

# ------------------------------------------------------------------------------
# Create network from mesh
# ------------------------------------------------------------------------------

nodes, faces = mesh.to_vertices_and_faces()
edges = list(mesh.edges())
network = FDNetwork.from_nodes_and_edges(nodes, edges)

network.edges_forcedensities(q0)
network.nodes_supports(supports)

# ------------------------------------------------------------------------------
# Equilibrium models
# ------------------------------------------------------------------------------

model = FDModel.from_network(network)
structure = FDStructure.from_network(network)
fdq = model(structure)

if not OPTIMIZE:
    network_opt = network_updated(network, fdq)

# ==========================================================================
# Extract vertices' ordered neighbors
# ==========================================================================

print("Computing neighborhoods")
neighborhoods = []

for vertex in structure.nodes:

    nbrs_indices = []
    for vkey in mesh.vertex_neighbors(vertex, ordered=True):

        if vkey == vertex:
            continue

        index = structure.node_index[vkey]
        nbrs_indices.append(index)

    neighborhoods.append(nbrs_indices)

# ==========================================================================
# Pad vertices' ordered neighbors
# ==========================================================================

neighborhoods_padded = []
largest_neighborhood = max(len(hood) for hood in neighborhoods)

for hood in neighborhoods:

    hood_size = len(hood)
    if hood_size == largest_neighborhood:
        neighborhoods_padded.append(hood)
        continue

    hood_padded = hood + [-1] * (largest_neighborhood - hood_size)
    neighborhoods_padded.append(hood_padded)

assert all(len(hood) == largest_neighborhood for hood in neighborhoods_padded)

hoods_padded = jnp.array(neighborhoods_padded)
hoods_size = jnp.array([len(hood) for hood in neighborhoods])
vertices_free_index = jnp.array([idx for idx, vkey in zip(range(network.number_of_nodes()), structure.nodes) if not network.is_node_support(vkey)])


# ==========================================================================
# Fairness
# ==========================================================================

def hood_xyz(hood, xyz):
    """
    Get the polygon formed by vertex neighborhood from the xyz vertices array.
    """
    xyz_hood = xyz[hood, :]
    xyz_repl = jnp.zeros_like(xyz_hood)

    hood_2d = jnp.reshape(hood, (-1, 1))
    hxyz = jnp.where(hood_2d >= 0, xyz_hood, xyz_repl)
    assert hxyz.shape == xyz_hood.shape, f"{hxyz.shape}"

    return hxyz


def vertex_hood_fairness_quad(vertex_xyz, hood_xyz):
    """
    Compute the fairness vector of a quad vertex neighborhood.
    """
    hood_xyz = hood_xyz[:4, :]  # take the first four coordinates
    assert hood_xyz.shape == (4, 3), f"{hood_xyz.shape}"

    diag_xyz_a = (hood_xyz[0, :] + hood_xyz[2, :]) / 2.0
    diag_xyz_b = (hood_xyz[1, :] + hood_xyz[3, :]) / 2.0

    assert diag_xyz_a.shape == vertex_xyz.shape
    assert diag_xyz_b.shape == vertex_xyz.shape

    fvector_a = vertex_xyz - diag_xyz_a
    fvector_b = vertex_xyz - diag_xyz_b

    fairness = jnp.sum(jnp.square(fvector_a)) + jnp.sum(jnp.square(fvector_b))

    return fairness


def vertex_hood_fairness_ngon(vertex_xyz, hood_xyz, hood_size):
    """
    Compute the fairness of an n-gon vertex neighborhood.
    """
    hvector = jnp.sum(hood_xyz, axis=0) / hood_size
    fvector = vertex_xyz - hvector
    assert fvector.shape == vertex_xyz.shape

    fairness = jnp.sum(jnp.square(fvector))

    return fairness


def vertex_hood_fairness(vertex, hood, hood_size, xyz):
    """
    Calculate the fairness of a vertex based on the position of its neighbors.
    """
    vxyz = xyz[vertex, :]
    hxyz = hood_xyz(hood, xyz)

    return jnp.where(hood_size == 4,
                     vertex_hood_fairness_quad(vxyz, hxyz),
                     vertex_hood_fairness_ngon(vxyz, hxyz, hood_size))


def vertices_hoods_fairness(vertices, hoods, hoods_size, xyz):
    """
    Calculate the fairness energy of the vertices of a mesh following Tang, et al. 2014.
    """
    fairness_fn = vmap(vertex_hood_fairness, in_axes=(0, 0, 0, None))
    fairnesses = fairness_fn(vertices, hoods, hoods_size, xyz)

    return jnp.sum(fairnesses)


def goal_fairness(eqstate, alpha):
    """
    Measure the planarity of the faces of a mesh structure.
    """
    xyz = eqstate.xyz
    indices = vertices_free_index

    return alpha * vertices_hoods_fairness(indices,
                                           hoods_padded[indices, :],
                                           hoods_size[indices],
                                           xyz)


# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

if OPTIMIZE:

    lengths_target = []
    for edge in network.edges():
        length = network.edge_length(*edge)
        lengths_target.append(length)

    lengths_target = jnp.asarray(lengths_target) * length_factor

    # define goal functions
    def goal_length(eq_state, alpha):
        return alpha * jnp.mean((eq_state.lengths - length_target) ** 2)

    # define loss function
    def loss_fn(diff_model, static_model):
        """
        A loss function.
        """
        model = eqx.combine(diff_model, static_model)
        eq_state = model(structure)

        return goal_length(eq_state, alpha_length) + goal_fairness(eq_state, alpha_fairness)

    # set tree filtering specification
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(lambda tree: (tree.q), filter_spec, replace=(True))

    # split model into differentiable and static submodels
    diff_model, static_model = eqx.partition(model, filter_spec)

    bounds_low_compas = jnp.ones_like(model.q) * qmin
    bounds_up_compas = jnp.ones_like(model.q) * qmax
    bound_low = eqx.tree_at(lambda tree: (tree.q),
                            diff_model,
                            replace=(bounds_low_compas))
    bound_up = eqx.tree_at(lambda tree: (tree.q),
                           diff_model,
                           replace=(bounds_up_compas))

    bounds = (bound_low, bound_up)

    # evaluate loss function at the start
    loss = loss_fn(diff_model, static_model)
    print(f"{loss=}")

    # solve optimization problem with scipy
    print("\n***Optimizing FDM with scipy***")
    # optimizer = jaxopt.ScipyMinimize
    optimizer = jaxopt.ScipyBoundedMinimize

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
    # opt_result = opt.run(diff_model, static_model)

    opt_result = opt.run(diff_model, bounds, static_model)
    print(f"Opt time: {time() - start:.4f} sec")
    diff_model_star, opt_state_star = opt_result

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

if OPTIMIZE and PLOT:

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

# for node in nodes_target:
#     point = Point(*network.node_coordinates(node))
#     plotter.add(point, size=3)

plotter.add(network_opt,
            nodesize=15,
            edgecolor="force",
            show_reactions=False,
            show_loads=False,
            edgewidth=(0.25, 2.5),
            reactioncolor=Color.from_rgb255(0, 150, 10),
            show_edgetext=False,
            show_nodes=True,
            reactionscale=0.5)

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
