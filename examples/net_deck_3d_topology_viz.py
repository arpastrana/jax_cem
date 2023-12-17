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


DISPLAY_DECK = False
DISPLAY_CABLENET = True

VIEW = True
PLOT = False

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------

HERE = os.path.dirname(__file__)
IN_DECK = os.path.abspath(os.path.join(HERE, "data/deck_topology_3d.json"))
IN_NET = os.path.abspath(os.path.join(HERE, "data/net_hexagon_topology_3d.json"))

# ------------------------------------------------------------------------------
# Load from JSON
# ------------------------------------------------------------------------------

deck = TopologyDiagram.from_json(IN_DECK)
cablenet = FDNetwork.from_json(IN_NET)

_networks = []
if DISPLAY_DECK:
    _networks.append(deck)
if DISPLAY_CABLENET:
    _networks.append(cablenet)

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
    plotter = PlotterFD(figsize=(8, 5), dpi=200)

    # plotter.add(topology)
    # for _network in [network, topology]:
    #     nodes, edges = _network.to_nodes_and_edges()
    #     _network = Network.from_nodes_and_edges(nodes, edges)
    #     plotter.add(_network,
    #                 show_nodes=False,
    #                 edgewidth=0.5,
    #                 edgecolor={edge: Color.grey() for edge in _network.edges()})

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
                show_reactions=False,
                show_loads=False,
                edgewidth=(1., 3.),
                show_edgetext=False,
                show_nodes=True,
                reactionscale=1.0)

    plotter.zoom_extents()
    # plotter.save("net_deck_integrated.pdf")
    plotter.show()
