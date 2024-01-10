import os

import matplotlib.pyplot as plt

import math
from math import radians
from math import fabs

import numpy as np

from compas.datastructures import Mesh
from compas.datastructures import network_transformed
from compas.colors import Color

from compas.geometry import Line
from compas.geometry import Polyline
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
name_spoke_mesh = "stadium_spoke_mesh"
name_net_mesh = "stadium_cablenet_mesh"

DELETE_AUX_TRAILS = True

DISPLAY_ARCH = True
DISPLAY_CABLENET = False
DISPLAY_SPOKE = False
DISPLAY_MESHES = True

DISPLAY_FOCUS = "arch"  # 0: cablenet, 1: arch, 3: spoke
HALVE_NETWORK = False
HALVE_MESHES = False

VIEW = True

PLOT = True
PLOT_SAVE = True

PLOT_BASE_SPOKE = True

PLOT_TARGET_LINES = False
PLOT_TARGET_LINES_CIRCLE = False

PLOT_MESH_NET = False
PLOT_MESH_SPOKE = True

plot_transforms = {
                   "3d": {"figsize": (6, 6), "padding": -0.1},
                   # "top": {"figsize": (6, 6), "padding": -0.5},
                   # "y": {"figsize": (6, 6), "padding": -0.5},
                   # "x": {"figsize": (6, 6), "padding": -0.5},
                   }

edgewidth_view = (0.03, 0.06)
edgewidth_plot = (0.5, 2.5)
nodesize = 2.0
nodesize_factor = 8
nodesize_factor_xy = 6

color_mesh_net = Color(0.9, 0.9, 0.9)
color_mesh_spoke = Color(0.8, 0.8, 0.8)

color_pink = Color.from_rgb255(255, 123, 171)
color_support = Color.from_rgb255(0, 150, 10)
color_orange = Color.from_rgb255(255, 180, 130)
color_purple = Color.purple()
color_gray = Color(0.2, 0.2, 0.2)
color_gray_light = Color.from_rgb255(204, 204, 204)

color_white = Color(0.85, 0.85, 0.85)
# color_white = Color.white()

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------

HERE = os.path.dirname(__file__)

IN_ARCH = os.path.abspath(os.path.join(HERE, f"data/{name_arch}.json"))
IN_NET = os.path.abspath(os.path.join(HERE, f"data/{name_cablenet}.json"))
IN_SPOKE = os.path.abspath(os.path.join(HERE, f"data/{name_spoke}.json"))

IN_ARCH_BASE = os.path.abspath(os.path.join(HERE, f"data/{name_arch}.json"))
IN_NET_BASE = os.path.abspath(os.path.join(HERE, f"data/{name_cablenet}.json"))
IN_SPOKE_BASE = os.path.abspath(os.path.join(HERE, f"data/{name_spoke}.json"))
IN_MESH_SPOKE_BASE = os.path.abspath(os.path.join(HERE, f"data/{name_spoke_mesh}.json"))
IN_MESH_NET_BASE = os.path.abspath(os.path.join(HERE, f"data/{name_net_mesh}.json"))

# ------------------------------------------------------------------------------
# Load from JSON
# ------------------------------------------------------------------------------

cablenet_base = FDNetwork.from_json(IN_NET_BASE)
arch_base = TopologyDiagram.from_json(IN_ARCH_BASE)
spoke_base = TopologyDiagram.from_json(IN_SPOKE_BASE)
mesh_spoke = Mesh.from_json(IN_MESH_SPOKE_BASE)
mesh_net = Mesh.from_json(IN_MESH_NET_BASE)


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
    for network in networks.values():
        _forces = [fabs(network.edge_force(edge)) for edge in network.edges()]
        forces_bag.extend(_forces)
        forces.append(_forces)

    force_min = min(forces_bag)
    force_max = max(forces_bag)

    widths = []
    widths_bag = []
    for network, forces in zip(networks.values(), forces):
        _widths = remap_values(forces, width_min, width_max, force_min, force_max)
        edgewidths = {edge: width for edge, width in zip(network.edges(), _widths)}
        widths.append(edgewidths)
        widths_bag.extend(_widths)

    return widths


def calculate_edge_colors(network):
    """
    """
    edgecolor = color_pink
    if isinstance(network, TopologyDiagram):
        edgecolor = color_gray

    return edgecolor


def halve_network(network, axis="y"):
    """
    """
    deletable = []
    for node in network.nodes():
        if network.node_attribute(node, axis) <= 0.0:
            deletable.append(node)

    for node in deletable:
        network.delete_node(node)


def halve_mesh(mesh, axis="y"):
    """
    """
    deletable = []
    for node in mesh.vertices():
        if mesh.vertex_attribute(node, axis) <= 0.0:
            deletable.append(node)

    for node in deletable:
        mesh.delete_vertex(node)


def sort_points_around_origin(points):
    """
    """
    # Function to convert Cartesian coordinates to Spherical
    def cartesian_to_spherical(x, y, z):
        r = math.sqrt(x**2 + y**2 + z**2)
        theta = math.atan2(y, x)  # Azimuthal angle
        phi = math.acos(z / r)  # Polar angle
        return r, theta, phi

    # Function to convert Spherical coordinates to Cartesian
    def spherical_to_cartesian(r, theta, phi):
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        return x, y, z

    # Convert each point to spherical coordinates
    spherical_points = [cartesian_to_spherical(x, y, z) for x, y, z in points]

    # Sort points by their azimuthal angle and then by their polar angle
    spherical_points.sort(key=lambda p: (p[1], p[2]))

    # Convert the sorted points back to Cartesian coordinates (if needed)
    return [spherical_to_cartesian(r, theta, phi) for r, theta, phi in spherical_points]


# ------------------------------------------------------------------------------
# Clean up
# ------------------------------------------------------------------------------

_networks = {}
_networks_base = {}

# spoke
spoke = TopologyDiagram.from_json(IN_SPOKE)
_networks["spoke"] = spoke
_networks_base["spoke"] = spoke_base

# update mesh coordinates
gkey_key = spoke_base.gkey_key()
for vkey in mesh_spoke.vertices():
    gkey = geometric_key(mesh_spoke.vertex_coordinates(vkey))
    key = gkey_key[gkey]
    xyz = spoke.node_coordinates(key)
    mesh_spoke.vertex_attributes(vkey, "xyz", xyz)

# cablenet
cablenet = FDNetwork.from_json(IN_NET)
_networks["cablenet"] = cablenet
_networks_base["cablenet"] = cablenet_base

gkey_key = cablenet_base.gkey_key()
for vkey in mesh_net.vertices():
    gkey = geometric_key(mesh_net.vertex_coordinates(vkey))
    key = gkey_key[gkey]
    xyz = cablenet.node_coordinates(key)
    mesh_net.vertex_attributes(vkey, "xyz", xyz)

# arch
arch = TopologyDiagram.from_json(IN_ARCH)
_networks["arch"] = arch
_networks_base["arch"] = arch_base

# ------------------------------------------------------------------------------
# Delete auxiliary trails
# ------------------------------------------------------------------------------

if DELETE_AUX_TRAILS:
    for _network, _network_base in zip(_networks.values(), _networks_base.values()):
        if isinstance(_network_base, TopologyDiagram):
            delete_topology_aux_trails(_network, _network_base)

# ------------------------------------------------------------------------------
# Viewer
# ------------------------------------------------------------------------------

if VIEW:
    viewer = ViewerFD(width=1000,
                      height=1000,
                      show_grid=True,
                      viewmode="lighted")

    viewer.view.camera.distance = 15.0
    viewer.view.camera.rotation = (radians(45.0), radians(0.0), radians(-45.0))

    width_min_view, width_max_view = edgewidth_view
    edgewidths = calculate_edge_widths(_networks, width_min_view, width_max_view)

    for _network, _edgewidth in zip(_networks.values(), edgewidths):
        if isinstance(_network, TopologyDiagram):
            _network = form_cem_to_fdnetwork(_network)

        viewer.add(_network,
                   edgewidth=_edgewidth,
                   edgecolor=None,
                   show_reactions=False,
                   show_loads=False,
                   )

    if DISPLAY_MESHES:
        viewer.add(mesh_net,
                   opacity=0.4,
                   show_points=False,
                   show_edges=False)

        viewer.add(mesh_spoke,
                   opacity=0.6,
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
        for i, (name, network) in enumerate(_networks.items()):

            # reaction size
            rs = 0.1
            if plot_transform != "3d":
                rs = rs / 2.0

            # color, width and zorder
            edgewidth = 0.75
            edgecolor = color_gray_light
            zorder = (i + 1) * 1_000
            show_nodes = False

            if DISPLAY_FOCUS == name:
                edgecolor = calculate_edge_colors(network)
                edgewidth = 1.0
                zorder = 10_000
                show_nodes = True

            elif name == "cablenet" and not DISPLAY_CABLENET:
                edgecolor = color_white
                edgewidth = 0.75
                # zorder = 1
            elif name == "spoke" and not DISPLAY_SPOKE:
                edgecolor = color_white
                edgewidth = 0.75
                # zorder = 1
            elif name == "arch" and not DISPLAY_ARCH:
                edgecolor = color_white
                # edgewidth = 0.75
                # zorder = 1

                edgewidth = {}
                for edge in network.edges():
                    if network.edge_attribute(edge, "group") == "cable":
                        edgewidth[edge] = 0.25
                    else:
                        edgewidth[edge] = 0.75
                # deletable = []
                # for edge in network.edges_where({"group": "cable"}):
                #     deletable.append(edge)
                # for edge in deletable:
                #     network.delete_edge(*edge)

            # halve network
            if HALVE_NETWORK:
                halve_network(network, axis="y")
            if HALVE_MESHES:
                halve_mesh(mesh_net, axis="y")
                halve_mesh(mesh_spoke, axis="y")

            if isinstance(network, TopologyDiagram):
                network = form_cem_to_fdnetwork(network)

            # transform network
            network_plot = network_transformed(network, T)

            # transform loads
            transform_network_vectors(network_plot, network, ["px", "py", "pz"])
            transform_network_vectors(network_plot, network, ["rx", "ry", "rz"])

            plotter.add(network_plot,
                        nodesize=ns,
                        show_nodes=show_nodes,
                        show_edges=True,
                        show_loads=False,
                        show_reactions=False,
                        edgewidth=edgewidth,
                        edgecolor=edgecolor,
                        sizepolicy="absolute",
                        loadscale=1.0,
                        reactionscale=rs,
                        reactioncolor=color_gray,
                        zorder=zorder
                        )

        if PLOT_MESH_NET:
            mesh_plot = mesh_net.transformed(T)
            plotter.add(mesh_plot,
                        show_edges=False,
                        show_vertices=False,
                        facecolor={fkey: color_mesh_net for fkey in mesh_net.faces()},
                        zorder=10,
                        )

        if PLOT_MESH_SPOKE:
            mesh_plot = mesh_spoke.transformed(T)
            plotter.add(mesh_plot,
                        show_edges=False,
                        show_vertices=False,
                        facecolor={fkey: color_mesh_spoke for fkey in mesh_spoke.faces()},
                        zorder=10,
                        )

        if PLOT_BASE_SPOKE:
            points = [spoke.node_coordinates(node) for node in spoke.nodes() if spoke.is_node_support(node)]
            sorted_endpoints = sort_points_around_origin(points)
            sorted_endpoints = sorted_endpoints + sorted_endpoints[:1]
            polyline = Polyline(sorted_endpoints).transformed(T)
            plotter.add(polyline,
                        draw_points=False,
                        color=color_gray_light,
                        linewidth=0.5,
                        linestyle="solid",
                        zorder=20)

        if PLOT_TARGET_LINES:
            line_length = 3.0
            line_length_up = 0.1  # 1.5
            line_length_down = 1.5

            nodes_cable = []
            for edge in cablenet.edges_where({"group": "cable"}):
                nodes_cable.extend(edge)

            end_points = []
            for node in nodes_cable:
                xyz = cablenet.node_coordinates(node)
                start = add_vectors(xyz, [0.0, 0.0, line_length_up])
                # end = add_vectors(xyz, [0.0, 0.0, -(line_length - line_length_up)])
                end = add_vectors(xyz, [0.0, 0.0, -line_length_down])
                end_points.append(end)
                # start = add_vectors(xyz, [0.0, 0.0, line_length / 2.0])
                # end = add_vectors(xyz, [0.0, 0.0, -line_length / 2.0])
                line = Line(start, end).transformed(T)
                plotter.add(line,
                            draw_as_segment=True,
                            linecolor=color_orange,
                            color=color_orange,
                            linewidth=0.5,
                            linestyle="dotted",
                            zorder=20_000)

            if PLOT_TARGET_LINES_CIRCLE:
                sorted_endpoints = sort_points_around_origin(end_points)
                sorted_endpoints = sorted_endpoints + sorted_endpoints[:1]
                polyline = Polyline(sorted_endpoints).transformed(T)
                plotter.add(polyline,
                            draw_points=False,
                            color=color_orange,
                            linewidth=0.5,
                            linestyle="solid",
                            zorder=20_000)

        padding = plot_config["padding"]
        plotter.zoom_extents(padding=padding)

        if PLOT_SAVE:
            filename = f"{DISPLAY_FOCUS}_3d_constraints_{plot_transform}_plot.pdf"
            FILE_OUT = os.path.abspath(os.path.join(HERE, filename))
            print(f"\nSaving plot to {filename}")
            plotter.save(FILE_OUT, bbox_inches=0.0, transparent=True)

        plotter.show()
