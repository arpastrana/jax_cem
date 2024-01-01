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

from jax_fdm.visualization import Plotter as PlotterFD
from jax_fdm.visualization import Viewer as ViewerFD


DISPLAY_DECK = True
DISPLAY_CABLENET = True

VIEW = False
PLOT = False
PLOT_SAVE = False

PLOT_CONSTRAINTS = True
PLOT_CONSTRAINTS_SAVE = True

color_orange = Color.from_rgb255(255, 141, 65)
color_purple = Color.purple()
color_pink = Color.from_rgb255(255, 123, 171)
color_support = Color.from_rgb255(0, 150, 10)

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------

HERE = os.path.dirname(__file__)
IN_DECK = os.path.abspath(os.path.join(HERE, "data/deck.json"))
IN_NET = os.path.abspath(os.path.join(HERE, "data/net.json"))

# ------------------------------------------------------------------------------
# Load from JSON
# ------------------------------------------------------------------------------

deck = TopologyDiagram.from_json(IN_DECK)
cablenet = FDNetwork.from_json(IN_NET)

_networks = []

if DISPLAY_DECK:
    _networks.append((deck, "deck"))
if DISPLAY_CABLENET:
    _networks.append((cablenet, "cablenet"))

# ------------------------------------------------------------------------------
# Viewer
# ------------------------------------------------------------------------------

if VIEW:
    viewer = ViewerFD(width=1000, height=1000, show_grid=False, viewmode="lighted")

    viewer.view.camera.distance = 28.0
    viewer.view.camera.position = (-13.859, 20.460, 14.682)
    viewer.view.camera.target = (1.008, 4.698, -3.034)
    viewer.view.camera.rotation = (0.885, 0.000, -2.385)

    for _network in _networks:

        _network, _ = _network
        edges = list(_network.edges())

        if isinstance(_network, TopologyDiagram):
            _edges = []
            for edge in edges:
                u, v = edge
                edge_rev = v, u
                if _network.is_auxiliary_trail_edge(edge) or _network.is_auxiliary_trail_edge(edge_rev):
                    continue
                _edges.append(edge)
            edges = _edges

        key_index = dict((key, index) for index, key in enumerate(_network.nodes()))
        nodes = [_network.node_coordinates(key) for key in _network.nodes()]
        edges = [(key_index[u], key_index[v]) for u, v in edges]

        _network = Network.from_nodes_and_edges(nodes, edges)
        viewer.add(_network, show_points=False, linewidth=6.0, linecolor=Color.grey().darkened())

    viewer.show()

# ------------------------------------------------------------------------------
# Plotter
# ------------------------------------------------------------------------------

if PLOT:

    for _network in _networks:

        plotter = PlotterFD(figsize=(8, 5), dpi=200)
        _network, name = _network

        nodes, edges = _network.to_nodes_and_edges()

        edges = list(_network.edges())
        if isinstance(_network, TopologyDiagram):
            _edges = []
            for edge in edges:
                u, v = edge
                edge_rev = v, u
                if _network.is_auxiliary_trail_edge(edge) or _network.is_auxiliary_trail_edge(edge_rev):
                    continue
                _edges.append(edge)

            edges = _edges

        key_index = dict((key, index) for index, key in enumerate(_network.nodes()))
        nodes = [_network.node_coordinates(key) for key in _network.nodes()]
        edges = [(key_index[u], key_index[v]) for u, v in edges]

        _network = Network.from_nodes_and_edges(nodes, edges)

        plotter.add(_network,
                    show_nodes=True,
                    edgewidth=1.0,
                    nodesize=3.0,
                    edgecolor={edge: Color(0.2, 0.2, 0.2) for edge in _network.edges()},
                    nodecolor={node: Color.white() for node in _network.nodes()},
                    sizepolicy="relative",
                    )

        plotter.zoom_extents()
        if PLOT_SAVE:
            filename = f"{name}_topology.pdf"
            plotter.save(filename, transparent=True)
            print(f"Saved pdf to {filename}")
        plotter.show()

# ------------------------------------------------------------------------------
# Indices pre calculation
# ------------------------------------------------------------------------------

if PLOT_CONSTRAINTS:

    # fdm goals
    nodes_fdm = []  # nodes in fdm where to apply cem reaction as a load

    deck.build_trails(True)

    gkey_key = cablenet.gkey_key()
    for node in deck.nodes():
        if deck.is_node_origin(node):
            neighbor = deck.neighbors(node).pop()
            if deck.is_node_support(neighbor):
                key = gkey_key[geometric_key(deck.node_coordinates(node))]
                nodes_fdm.append(key)

    # ce goals
    nodes_cem = []
    for node in deck.nodes():
        if deck.is_node_origin(node):
            neighbor = deck.neighbors(node).pop()
            if deck.is_node_support(neighbor):
                continue
            # continue
        if deck.is_node_support(node):
            neighbor = deck.neighbors(node).pop()
            if deck.is_node_origin(neighbor):
                continue
        nodes_cem.append(node)

    plotter = PlotterFD(figsize=(8, 5), dpi=200)

    for _network in _networks:

        _network, name = _network

        if isinstance(_network, TopologyDiagram):

            deletable = []
            for node in _network.nodes():
                if deck.is_node_support(node):
                    neighbor = deck.neighbors(node).pop()
                    if deck.is_node_origin(neighbor):
                        deletable.append(node)

            for node in deletable:
                _network.delete_node(node)

            # _network = FDNetwork.from_data(_network.data)

        nodecolor = {}
        edgecolor = {}
        if isinstance(_network, TopologyDiagram):
            ns = 20.0
            gkey_key = deck.gkey_key()
            for node in _network.nodes():
                if deck.is_node_support(node):
                    c = color_support
                elif node in nodes_cem:
                    c = color_pink
                else:
                    c = color_support
                nodecolor[node] = c

            edgecolor = {edge: Color(0.2, 0.2, 0.2) for edge in _network.edges()}
            _network = FDNetwork.from_data(_network.data)

        else:
            ns = 20.0
            gkey_key = cablenet.gkey_key()
            for node in _network.nodes():
                key = gkey_key.get(geometric_key(_network.node_coordinates(node)))
                if key is None:
                    continue

                if key in nodes_fdm:
                    c = color_pink
                elif cablenet.is_node_support(node):
                    c = color_support
                else:
                    c = Color.white()
                nodecolor[node] = c

            for edge in _network.edges():
                if _network.edge_attribute(edge, "group") == "cable":
                    c = color_orange
                    c = color_pink
                elif _network.edge_attribute(edge, "group") == "hangers":
                    c = color_purple
                    c = Color(0.2, 0.2, 0.2)
                    c = color_pink
                    c = color_orange
                else:
                    c = Color(0.2, 0.2, 0.2)

                edgecolor[edge] = c

        plotter.add(_network,
                    show_nodes=True,
                    edgewidth=1.0,
                    nodesize=ns,
                    edgecolor=edgecolor,
                    nodecolor=nodecolor,
                    sizepolicy="absolute",
                    )

    plotter.zoom_extents()
    if PLOT_CONSTRAINTS_SAVE:
        filename = "netdeck_constraints.pdf"
        plotter.save(filename, transparent=True)
        print(f"Saved pdf to {filename}")

    plotter.show()
