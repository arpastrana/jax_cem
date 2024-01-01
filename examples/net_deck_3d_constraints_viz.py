import os

import matplotlib.pyplot as plt

from time import time

from compas.datastructures import Network
from compas.datastructures import Mesh
from compas.datastructures import network_transformed
from compas.colors import Color
from compas.geometry import Line
from compas.geometry import add_vectors
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

from jax_fdm.visualization import Plotter as PlotterFD
from jax_fdm.visualization import Viewer as ViewerFD


DISPLAY_DECK = True
DISPLAY_CABLENET = True

VIEW = True
PLOT = True
PLOT_MESH = False
PLOT_LINES = True
PLOT_SAVE = True

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
IN_DECK = os.path.abspath(os.path.join(HERE, "data/deck_3d.json"))
IN_NET = os.path.abspath(os.path.join(HERE, "data/net_hexagon_3d.json"))

IN_MESH_DECK = os.path.abspath(os.path.join(HERE, "data/deck_mesh_3d.json"))

# ------------------------------------------------------------------------------
# Load from JSON
# ------------------------------------------------------------------------------

cablenet = FDNetwork.from_json(IN_NET)
deck = TopologyDiagram.from_json(IN_DECK)
mesh = Mesh.from_json(IN_MESH_DECK)

# ------------------------------------------------------------------------------
# Indices pre calculation
# ------------------------------------------------------------------------------

gkey_key = cablenet.gkey_key()

# add loads
nodes_cem = []  # support nodes in cem world where to get reaction force from
nodes_fdm = []  # nodes in fdm where to apply cem reaction as a load

for node in deck.nodes():
    if deck.is_node_origin(node):
        for neighbor in deck.neighbors(node):
            if not deck.is_node_support(neighbor):
                continue

            key = gkey_key.get(geometric_key(deck.node_coordinates(node)))

            if key is None:
                continue

            nodes_cem.append(neighbor)
            nodes_fdm.append(key)

# ce goals
nodes_ce_xyz_opt = []
for node in deck.nodes():
    if deck.is_node_origin(node):
        if any(deck.is_node_support(nbr) for nbr in deck.neighbors(node)):
            continue
    if deck.is_node_support(node):
        neighbor = deck.neighbors(node).pop()
        if deck.is_node_origin(neighbor):
            continue
    nodes_ce_xyz_opt.append(node)

nodes_ce_res_opt = []
for node in deck.nodes():
    if node in nodes_cem or node in nodes_ce_xyz_opt:
        continue
    if not deck.is_node_support(node):
        continue
    nodes_ce_res_opt.append(node)

# ------------------------------------------------------------------------------
# Clean up
# ------------------------------------------------------------------------------

# delete auxiliary trail edges from deck
deletable = []
nodes_cem_residual = []
for edge in deck.edges():
    if not deck.is_auxiliary_trail_edge(edge):
        continue
    for node in edge:
        if deck.is_node_support(node):
            deletable.append(node)
        else:
            nodes_cem_residual.append(node)
for node in deletable:
    deck.delete_node(node)

_networks = []
if DISPLAY_DECK:
    _networks.append(deck)
if DISPLAY_CABLENET:
    _networks.append(cablenet)


# ------------------------------------------------------------------------------
# Viewer
# ------------------------------------------------------------------------------

if VIEW:
    viewer = ViewerFD(width=1000,
                      height=1000,
                      show_grid=False,
                      viewmode="lighted")

    viewer.view.camera.distance = 28.0
    viewer.view.camera.position = (-13.859, 20.460, 14.682)
    viewer.view.camera.target = (1.008, 4.698, -3.034)
    viewer.view.camera.rotation = (0.885, 0.000, -2.385)

    for _network in _networks:
        viewer.add(_network,
                   show_points=False,
                   linewidth=6.0,
                   linecolor=Color.grey().darkened())

    viewer.add(mesh, opacity=0.4, show_points=False, show_edges=False)

    viewer.show()

# ------------------------------------------------------------------------------
# Plotter
# ------------------------------------------------------------------------------

if PLOT:
    plotter = PlotterFD(figsize=(6, 6), dpi=200)

    P = viewer.view.camera.projection(viewer.width, viewer.height)
    W = viewer.view.camera.viewworld()
    P[1, 1] = P[0, 0]
    T = P @ W

    for network in _networks:
        network_plot = network
        network_plot = network_transformed(network, T)

        if isinstance(network, TopologyDiagram):
            network_plot = FDNetwork.from_data(network_plot.data)
            edgecolor = color_gray
            nodecolor = {}
            for node in network.nodes():
                if network_plot.is_node_support(node):
                    color = color_support
                elif node in nodes_ce_xyz_opt:
                    color = color_orange
                elif node in nodes_ce_res_opt:
                    color = color_orange
                else:
                    color = color_white

                nodecolor[node] = color
        else:
            nodecolor = None
            edgecolor = {}
            for edge in network.edges():
                if network.edge_attribute(edge, "group") == "cable":
                    color = color_pink
                elif network.edge_attribute(edge, "group") == "hangers":
                    color = color_pink
                else:
                    color = color_gray
                edgecolor[edge] = color

        plotter.add(network_plot,
                    nodesize=1.7,
                    nodecolor=nodecolor,
                    show_nodes=True,
                    show_edges=True,
                    show_loads=False,
                    edgewidth=1.0,
                    edgecolor=edgecolor,
                    sizepolicy="absolute",
                    show_reactions=False,
                    reactionscale=0.1,
                    )

    if PLOT_LINES:
        line_length = 6.0
        line_length_up = 2.0
        for node in nodes_ce_xyz_opt:
            xyz = deck.node_coordinates(node)
            start = add_vectors(xyz, [0.0, 0.0, line_length_up])
            end = add_vectors(xyz, [0.0, 0.0, -(line_length - line_length_up)])
            # start = add_vectors(xyz, [0.0, 0.0, line_length / 2.0])
            # end = add_vectors(xyz, [0.0, 0.0, -line_length / 2.0])
            line = Line(start, end).transformed(T)
            plotter.add(line,
                        draw_as_segment=True,
                        linecolor=color_purple,
                        color=color_purple,
                        linewidth=0.5,
                        linestyle="dotted")

    if PLOT_MESH:
        mesh = mesh.transformed(T)
        plotter.add(mesh,
                    show_edges=False,
                    show_vertices=False,
                    facecolor={fkey: color_mesh for fkey in mesh.faces()},
                    )

    plotter.zoom_extents()
    if PLOT_SAVE:
        plotter.save("net_deck_3d_constraints.pdf", transparent=True, bbox_inches=0.0)
    plotter.show()
