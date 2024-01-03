import os

import matplotlib.pyplot as plt

from math import radians
from math import fabs

import numpy as np

from compas.datastructures import Mesh
from compas.datastructures import network_transformed
from compas.colors import Color

from compas.geometry import Line
from compas.geometry import add_vectors
from compas.geometry import subtract_vectors
from compas.geometry import scale_vector
from compas.geometry import Rotation
from compas.geometry import Transformation
from compas.geometry import multiply_matrices

from compas.utilities import geometric_key
from compas.utilities import remap_values

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

from jax_fdm.visualization import Plotter as PlotterFD
from jax_fdm.visualization import Viewer as ViewerFD


name_version = "opt"

name_arch = "stadium_arch"
name_cablenet = "stadium_cablenet"
name_spoke = "stadium_spoke"
name_mesh = "stadium_spoke_mesh"

DISPLAY_ARCH = True
DISPLAY_CABLENET = True
DISPLAY_SPOKE = True
DISPLAY_MESH = True

DELETE_AUX_TRAILS = True

VIEW = True

PLOT = True
PLOT_SAVE = False

PLOT_MESH = True
PLOT_DECK_LINES = False
PLOT_CHORD_TARGET_LINES = False

plot_transforms = {
                   "3d": {"figsize": (6, 6), "padding": 0.0},
                   "top": {"figsize": (6, 6), "padding": -0.5},
                   "y": {"figsize": (6, 6), "padding": -0.5},
                   }

edgewidth_view = (0.03, 0.06)
edgewidth_plot = (0.5, 2.5)
nodesize = 1.5
nodesize_factor = 8
nodesize_factor_xy = 6

color_mesh = Color(0.9, 0.9, 0.9)
color_pink = Color.from_rgb255(255, 123, 171)
color_support = Color.from_rgb255(0, 150, 10)
color_orange = Color.from_rgb255(255, 141, 65)
color_purple = Color.purple()
color_gray = Color(0.2, 0.2, 0.2)
color_white = Color.white()

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------

HERE = os.path.dirname(__file__)

IN_ARCH = os.path.abspath(os.path.join(HERE, f"data/{name_arch}_{name_version}.json"))
IN_NET = os.path.abspath(os.path.join(HERE, f"data/{name_cablenet}_{name_version}.json"))
IN_SPOKE = os.path.abspath(os.path.join(HERE, f"data/{name_spoke}_{name_version}.json"))

IN_ARCH_BASE = os.path.abspath(os.path.join(HERE, f"data/{name_arch}.json"))
IN_NET_BASE = os.path.abspath(os.path.join(HERE, f"data/{name_cablenet}.json"))
IN_SPOKE_BASE = os.path.abspath(os.path.join(HERE, f"data/{name_spoke}.json"))
IN_MESH_SPOKE_BASE = os.path.abspath(os.path.join(HERE, f"data/{name_mesh}.json"))

# ------------------------------------------------------------------------------
# Load from JSON
# ------------------------------------------------------------------------------

cablenet_base = FDNetwork.from_json(IN_NET_BASE)
arch_base = TopologyDiagram.from_json(IN_ARCH_BASE)
spoke_base = TopologyDiagram.from_json(IN_SPOKE_BASE)
mesh = Mesh.from_json(IN_MESH_SPOKE_BASE)

# ------------------------------------------------------------------------------
# Functions of functions
# ------------------------------------------------------------------------------

def form_cem_to_fdnetwork(form):
    """
    """
    network = form.copy(FDNetwork)

    for node in form.nodes():
        load = form.node_attributes(node, ["qx", "qy", "qz"])
        network.node_load(node, load)

    for node in form.nodes():
        if form.is_node_support(node):
            network.node_support(node)
            reaction = form.node_attributes(node, ["rx", "ry", "rz"])
            network.node_attributes(node, ["rx", "ry", "rz"], scale_vector(reaction, -1.0))

    for edge in form.edges():
        length = form.edge_length(*edge)
        network.edge_attribute(edge, "length", length)

        force = form.edge_force(edge)
        _q = force / length
        network.edge_attribute(edge, "q", _q)

    return network


def delete_topology_aux_trails(topology_to_delete, topology):
    """
    """
    deletable = []

    for edge in topology.edges():
        if not topology.is_auxiliary_trail_edge(edge):
            continue
        for node in edge:
            if topology.is_node_support(node):
                deletable.append(node)

    for node in deletable:
        topology_to_delete.delete_node(node)


def transform_network_vectors(network, network_ref, attr_names):
    """
    """
    for node in network.nodes():
        load = network_ref.node_attributes(node, attr_names)
        xyz = network_ref.node_coordinates(node)
        load_line = Line(xyz, add_vectors(xyz, load))
        load_line = load_line.transformed(T)
        load = subtract_vectors(load_line.end, load_line.start)
        network.node_attributes(node, attr_names, load)


def calculate_edge_widths(networks, width_min, width_max):
    """
    """
    forces_bag = []
    forces = []
    for network in networks:
        _forces = [fabs(network.edge_force(edge)) for edge in network.edges()]
        forces_bag.extend(_forces)
        forces.append(_forces)

    force_min = min(forces_bag)
    force_max = max(forces_bag)

    widths = []
    widths_bag = []
    for network, forces in zip(networks, forces):
        _widths = remap_values(forces, width_min, width_max, force_min, force_max)
        edgewidths = {edge: width for edge, width in zip(network.edges(), _widths)}
        widths.append(edgewidths)
        widths_bag.extend(_widths)

    return widths


# ------------------------------------------------------------------------------
# Indices pre calculation
# ------------------------------------------------------------------------------

# gkey_key = cablenet_base.gkey_key()

# # add loads
# nodes_cem = []  # support nodes in cem world where to get reaction force from
# nodes_fdm = []  # nodes in fdm where to apply cem reaction as a load

# for node in deck_base.nodes():
#     if deck_base.is_node_origin(node):
#         for neighbor in deck_base.neighbors(node):
#             if not deck_base.is_node_support(neighbor):
#                 continue

#             key = gkey_key.get(geometric_key(deck_base.node_coordinates(node)))

#             if key is None:
#                 continue

#             nodes_cem.append(neighbor)
#             nodes_fdm.append(key)

# # ce goals
# nodes_ce_xyz_opt = []
# for node in deck_base.nodes():
#     if deck_base.is_node_origin(node):
#         if any(deck_base.is_node_support(nbr) for nbr in deck_base.neighbors(node)):
#             continue
#     if deck_base.is_node_support(node):
#         neighbor = deck_base.neighbors(node).pop()
#         if deck_base.is_node_origin(neighbor):
#             continue
#     nodes_ce_xyz_opt.append(node)

# nodes_ce_res_opt = []
# for node in deck_base.nodes():
#     if node in nodes_cem or node in nodes_ce_xyz_opt:
#         continue
#     if not deck_base.is_node_support(node):
#         continue
#     nodes_ce_res_opt.append(node)

# ------------------------------------------------------------------------------
# Clean up
# ------------------------------------------------------------------------------

_networks = []
_networks_base = []

if DISPLAY_SPOKE:
    spoke = TopologyDiagram.from_json(IN_SPOKE)
    _networks.append(spoke)
    _networks_base.append(spoke_base)

if DISPLAY_CABLENET:
    cablenet = FDNetwork.from_json(IN_NET)
    _networks.append(cablenet)
    _networks_base.append(cablenet_base)

if DISPLAY_ARCH:
    arch = TopologyDiagram.from_json(IN_ARCH)
    _networks.append(arch)
    _networks_base.append(arch_base)

# ------------------------------------------------------------------------------
# Mesh update
# ------------------------------------------------------------------------------

    gkey_key = spoke_base.gkey_key()
    for vkey in mesh.vertices():
        gkey = geometric_key(mesh.vertex_coordinates(vkey))
        key = gkey_key[gkey]
        xyz = spoke.node_coordinates(key)
        mesh.vertex_attributes(vkey, "xyz", xyz)

# ------------------------------------------------------------------------------
# Delete auxiliary trails
# ------------------------------------------------------------------------------

if DELETE_AUX_TRAILS:
    for _network, _network_base in zip(_networks, _networks_base):
        if isinstance(_network_base, TopologyDiagram):
            delete_topology_aux_trails(_network, _network_base)

# ------------------------------------------------------------------------------
# Viewer
# ------------------------------------------------------------------------------

if VIEW:
    viewer = ViewerFD(width=1000,
                      height=1000,
                      show_grid=False,
                      viewmode="lighted")

    viewer.view.camera.distance = 15.0
    # viewer.view.camera.fov = 1.0
    # viewer.view.camera.position = (-13.859, 20.460, 14.682)
    # viewer.view.camera.target = (1.008, 4.698, -3.034)
    # viewer.view.camera.rotation = (0.885, 0.000, -2.385)
    # viewer.view.camera.rotation = (radians(45.0), radians(0.0), radians(45.0))

    width_min_view, width_max_view = edgewidth_view
    edgewidths = calculate_edge_widths(_networks, width_min_view, width_max_view)

    for _network, _edgewidth in zip(_networks, edgewidths):
        if isinstance(_network, TopologyDiagram):
            _network = form_cem_to_fdnetwork(_network)

        viewer.add(_network,
                   edgewidth=_edgewidth,
                   edgecolor="force", # Color.grey().darkened(),
                   show_reactions=False,
                   show_loads=False,
                   )

    if DISPLAY_MESH:
        viewer.add(mesh,
                   opacity=0.4,
                   show_points=False,
                   show_edges=False)

    viewer.show()

# ------------------------------------------------------------------------------
# Plotter
# ------------------------------------------------------------------------------

if PLOT:

    for plot_transform, plot_config in plot_transforms.items():

        print(f"\nPlotting with transform: {plot_transform}")
        figsize = plot_config["figsize"]
        plotter = PlotterFD(figsize=figsize, dpi=200)

        ns = nodesize

        if plot_transform == "3d" and VIEW:

            P = viewer.view.camera.projection(viewer.width, viewer.height)
            W = viewer.view.camera.viewworld()
            P[1, 1] = P[0, 0]
            T = P @ W

        elif plot_transform == "x":

            T = Rotation.from_axis_and_angle([1.0, 0.0, 0.0],
                                             radians(-90.0))

            ns = nodesize * nodesize_factor_xy

        elif plot_transform == "y":

            R1 = Rotation.from_axis_and_angle([1.0, 0.0, 0.0],
                                              radians(-90.0))

            R2 = Rotation.from_axis_and_angle([0.0, 1.0, 0.0],
                                              radians(90.0))

            T = Transformation.from_matrix(multiply_matrices(R2.matrix,
                                                             R1.matrix))
            ns = nodesize * nodesize_factor_xy

        elif plot_transform == "top":
            T = np.eye(4)
            ns = nodesize * nodesize_factor

        else:
            print("No transform!")
            T = np.eye(4)

        # calculate
        width_min_plot, width_max_plot = edgewidth_plot
        edgewidths = calculate_edge_widths(_networks, width_min_plot, width_max_plot)

        for i, network in enumerate(_networks):

            if isinstance(network, TopologyDiagram):
                network = form_cem_to_fdnetwork(network)

            network_plot = network_transformed(network, T)

            # transform loads
            transform_network_vectors(network_plot, network, ["px", "py", "pz"])
            transform_network_vectors(network_plot, network, ["rx", "ry", "rz"])

            rs = 0.3 if name_version != "fdm_opt" else 0.1
            if plot_transform != "3d":
                rs = rs / 2.0

            zorder = (i + 1) * 1000

            plotter.add(network_plot,
                        nodesize=ns,
                        show_nodes=True,
                        show_edges=True,
                        show_loads=False,
                        show_reactions=False,
                        edgewidth=edgewidth_plot,
                        edgecolor="force",
                        sizepolicy="absolute",
                        loadscale=1.0,
                        reactionscale=rs,
                        reactioncolor=color_gray,
                        zorder=zorder
                        )

        if PLOT_MESH:
            mesh_plot = mesh.transformed(T)
            plotter.add(mesh_plot,
                        show_edges=False,
                        show_vertices=False,
                        facecolor={fkey: color_mesh for fkey in mesh.faces()},
                        zorder=10,
                        )

        if PLOT_DECK_LINES:
            for edge in deck_base.edges():
                line = Line(*deck_base.edge_coordinates(*edge))
                line = line.transformed(T)
                plotter.add(line,
                            draw_as_segment=True,
                            linestyle="dashed",
                            lineweight=0.5,
                            zorder=2000,
                            )

        if PLOT_CHORD_TARGET_LINES:
            line_length = 7.0  # 3.0, 6.0
            line_length_up = 4.0  # 0.6,  2.0
            for node in nodes_ce_xyz_opt:
                xyz = deck_base.node_coordinates(node)
                start = add_vectors(xyz, [0.0, 0.0, line_length_up])
                end = add_vectors(xyz, [0.0, 0.0, -(line_length - line_length_up)])
                # start = add_vectors(xyz, [0.0, 0.0, line_length / 2.0])
                # end = add_vectors(xyz, [0.0, 0.0, -line_length / 2.0])
                line = Line(start, end).transformed(T)
                plotter.add(line,
                            draw_as_segment=True,
                            linecolor=color_orange,
                            color=color_orange,
                            linewidth=0.5,
                            linestyle="dotted")

        padding = plot_config["padding"]
        plotter.zoom_extents(padding=padding)

        if PLOT_SAVE:
            parts = []
            if DISPLAY_CABLENET:
                parts.append("net")
            if DISPLAY_ARCH:
                parts.append("arch")
            if DISPLAY_SPOKE:
                parts.append("spoke")

            name = "".join(parts)
            filename = f"{name}_3d_{name_version}_{plot_transform}_plot.pdf"
            FILE_OUT = os.path.abspath(os.path.join(HERE, filename))
            print(f"\nSaving plot to {filename}")
            plotter.save(FILE_OUT, bbox_inches=0.0, transparent=True)

        plotter.show()
