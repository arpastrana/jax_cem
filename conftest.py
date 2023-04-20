import pytest
import compas
import jax_cem
import compas_cem
import math
import numpy

from math import sqrt

from compas_cem.diagrams import TopologyDiagram
from compas_cem.elements import Node
from compas_cem.elements import TrailEdge
from compas_cem.elements import DeviationEdge
from compas_cem.loads import NodeLoad
from compas_cem.supports import NodeSupport


def pytest_ignore_collect(path):
    if "rhino" in str(path):
        return True

    if "blender" in str(path):
        return True

    if "ghpython" in str(path):
        return True


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(autouse=True)
def add_compas(doctest_namespace):
    doctest_namespace["compas"] = compas


@pytest.fixture(autouse=True)
def add_jax_cem(doctest_namespace):
    doctest_namespace["jax_cem"] = jax_cem


@pytest.fixture(autouse=True)
def add_compas_cem(doctest_namespace):
    doctest_namespace["compas_cem"] = compas_cem


@pytest.fixture(autouse=True)
def add_math(doctest_namespace):
    doctest_namespace["math"] = math


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace["np"] = numpy


@pytest.fixture
def compression_strut():
    """
    A one-edge, single-trail compression strut.
    """
    topology = TopologyDiagram()

    # add nodes
    topology.add_node(Node(0, [0.0, 0.0, 0.0]))
    topology.add_node(Node(1, [0.0, 2.0, 0.0]))
    # add edge with length 1, in compression
    topology.add_edge(TrailEdge(0, 1, length=-1.0))
    # add support
    topology.add_support(NodeSupport(0))
    # add loads at the unsupported edge
    topology.add_load(NodeLoad(1, [0, -1.0, 0.0]))
    # build trails
    topology.build_trails()

    return topology


@pytest.fixture
def tension_chain():
    """
    A chain with three edges in tension.
    The lengths of the first and the last edges are implicitly pulled to planes.
    """
    topology = TopologyDiagram()

    # add nodes
    topology.add_node(Node(0, [0.0, 0.0, 0.0]))
    topology.add_node(Node(1, [1.0, 0.0, 0.0]))
    topology.add_node(Node(2, [2.0, 0.0, 0.0]))
    topology.add_node(Node(3, [3.0, 0.0, 0.0]))
    # add edges
    topology.add_edge(TrailEdge(0, 1, length=1, plane=([1.5, 0.0, 0.0], [1.0, 0.0, 0.0])))
    topology.add_edge(TrailEdge(1, 2, length=1))  # unit length in tension
    topology.add_edge(TrailEdge(2, 3, length=1, plane=([4.0, 0.0, 0.0], [1.0, 0.0, 0.0])))
    # add support
    topology.add_support(NodeSupport(3))
    # add load
    topology.add_load(NodeLoad(0, [-1, 0, 0]))
    # build trails
    topology.build_trails()

    return topology


@pytest.fixture
def compression_chain():
    """
    A chain with three edges in compression.
    The lengths of the first and the last edges are implicitly pulled to planes.
    """
    topology = TopologyDiagram()

    # add nodes
    topology.add_node(Node(0, [0.0, 0.0, 0.0]))
    topology.add_node(Node(1, [1.0, 0.0, 0.0]))
    topology.add_node(Node(2, [2.0, 0.0, 0.0]))
    topology.add_node(Node(3, [3.0, 0.0, 0.0]))
    # add edges
    topology.add_edge(TrailEdge(0, 1, length=-1, plane=([1.5, 0.0, 0.0], [1.0, 0.0, 0.0])))
    topology.add_edge(TrailEdge(1, 2, length=-1))  # unit length in tension
    topology.add_edge(TrailEdge(2, 3, length=-1, plane=([4.0, 0.0, 0.0], [1.0, 0.0, 0.0])))
    # add support
    topology.add_support(NodeSupport(3))
    # add load
    topology.add_load(NodeLoad(0, [1, 0, 0]))
    # build trails
    topology.build_trails()

    return topology


@pytest.fixture
def threebar_funicular():
    """
    The simplest possible two-trail funicular structure in the CEM.
    """
    # create a topology diagram
    topology = TopologyDiagram()

    # add nodes
    topology.add_node(Node(0, [0.0, 0.0, 0.0]))
    topology.add_node(Node(1, [1.0, 0.0, 0.0]))
    topology.add_node(Node(2, [2.5, 0.0, 0.0]))
    topology.add_node(Node(3, [3.5, 0.0, 0.0]))

    # add edges with negative values for a compression-only structure
    topology.add_edge(TrailEdge(0, 1, length=-1.0))
    topology.add_edge(DeviationEdge(1, 2, force=-1.0))
    topology.add_edge(TrailEdge(2, 3, length=-1.0))

    # add supports
    topology.add_support(NodeSupport(0))
    topology.add_support(NodeSupport(3))

    # add loads
    topology.add_load(NodeLoad(1, [0.0, -1.0, 0.0]))
    topology.add_load(NodeLoad(2, [0.0, -1.0, 0.0]))

    # build trails
    topology.build_trails()

    return topology


@pytest.fixture
def braced_tower_2d():
    """
    A braced tower in 2d.
    """
    points = [
        (0, [0.0, 0.0, 0.0]),
        (1, [0.0, 1.0, 0.0]),
        (2, [0.0, 2.0, 0.0]),
        (3, [1.0, 0.0, 0.0]),
        (4, [1.0, 1.0, 0.0]),
        (5, [1.0, 2.0, 0.0]),
    ]

    trail_edges = [(0, 1), (1, 2), (3, 4), (4, 5)]

    deviation_edges = [(1, 4), (2, 5)]

    load = [0.0, -1.0, 0.0]

    topology = TopologyDiagram()

    for key, point in points:
        topology.add_node(Node(key, point))

    for u, v in trail_edges:
        topology.add_edge(TrailEdge(u, v, length=-1.0))

    for u, v in deviation_edges:
        topology.add_edge(DeviationEdge(u, v, force=-1.0))

    topology.add_edge(DeviationEdge(1, 5, force=1.0))
    topology.add_edge(DeviationEdge(1, 3, force=1.0))
    topology.add_edge(DeviationEdge(2, 4, force=1.0))

    topology.add_support(NodeSupport(0))
    topology.add_support(NodeSupport(3))

    topology.add_load(NodeLoad(2, load))
    topology.add_load(NodeLoad(5, load))

    # build trails
    topology.build_trails()

    return topology


@pytest.fixture
def tree_2d_needs_auxiliary_trails():
    """
    An planar tree that is missing two auxiliary trails to be valid topologically.
    """
    width = 4
    height = width / 2

    # Topology diagram
    topology = TopologyDiagram()

    # add nodes
    topology.add_node(Node(1, [-width / 2, height, 0.0]))
    topology.add_node(Node(2, [width / 2, height, 0.0]))
    topology.add_node(Node(3, [0.0, height / 2, 0.0]))
    topology.add_node(Node(4, [0.0, 0.0, 0.0]))

    # add edges with negative values for a compression-only structure
    topology.add_edge(TrailEdge(3, 4, length=-height / 2))

    topology.add_edge(DeviationEdge(1, 3, force=-sqrt(4.0)))
    topology.add_edge(DeviationEdge(2, 3, force=-sqrt(2.0)))
    topology.add_edge(DeviationEdge(1, 2, force=2.0))

    # add supports
    topology.add_support(NodeSupport(4))

    # add loads
    topology.add_load(NodeLoad(1, [0.0, -1.0, 0.0]))
    topology.add_load(NodeLoad(2, [0.0, -1.0, 0.0]))

    return topology


@pytest.fixture
def support_missing_topology():
    """
    A topology with three edges supposed to topology two trails. One support is missing.
    """
    topology = TopologyDiagram()
    # add five nodes
    for node_key in range(5):
        topology.add_node(Node(node_key, xyz=[0.0, float(node_key), 0.0]))

    # add two trail edges and one weird deviation edge
    topology.add_edge(TrailEdge(0, 1, length=1))
    topology.add_edge(TrailEdge(1, 2, length=1))
    topology.add_edge(DeviationEdge(3, 4, force=1))

    # add load
    topology.add_load(NodeLoad(0, [0, -1.0, 0.0]))

    # add only one support
    topology.add_support(NodeSupport(2))

    return topology


@pytest.fixture
def no_trails_topology():
    """
    A topology with only two deviation edges.
    """
    topology = TopologyDiagram()
    # add five nodes
    for node_key in range(3):
        topology.add_node(Node(node_key, xyz=[0.0, float(node_key), 0.0]))

    # add two trail edges and one weird deviation edge
    topology.add_edge(DeviationEdge(0, 1, force=1))
    topology.add_edge(DeviationEdge(1, 2, force=1))

    # add load
    topology.add_load(NodeLoad(0, [0, -1.0, 0.0]))

    # add only one support
    topology.add_support(NodeSupport(2))

    return topology


@pytest.fixture
def unsupported_topology():
    """
    A topology with one trail edge and a node load, but no supports.
    """
    topology = TopologyDiagram()
    # add five nodes
    for node_key in range(2):
        topology.add_node(Node(node_key, xyz=[0.0, float(node_key), 0.0]))

    # add two trail edges and one weird deviation edge
    topology.add_edge(TrailEdge(0, 1, length=-1))

    # add load
    topology.add_load(NodeLoad(0, [0, -1.0, 0.0]))

    return topology


@pytest.fixture
def topology_shifted_sequences():
    """
    A topology with shifted sequences of two different lengths.
    """
    points = [
        (0, [1.0, 0.0, 0.0]),
        (1, [1.0, -1.0, 0.0]),
        (2, [1.0, -2.0, 0.0]),
        (3, [1.0, -3.0, 0.0]),
        (4, [2.0, 0.0, 0.0]),
        (5, [2.0, -1.0, 0.0]),
        (6, [2.0, -2.0, 0.0]),
        (7, [2.0, -3.0, 0.0]),
    ]

    # key: plane
    trail_edges = {
        (0, 1): ([0.0, -1.0, 0.0], [0.0, -1.0, 0.0]),
        (1, 2): ([0.0, -2.0, 0.0], [0.0, -1.0, 0.0]),
        (2, 3): ([0.0, -3.0, 0.0], [0.0, -1.0, 0.0]),
    }

    deviation_edges = [(0, 4), (1, 5), (2, 6), (3, 7)]

    # create topology diagram
    topology = TopologyDiagram()

    for key, point in points:
        topology.add_node(Node(key, point))

    for (u, v), plane in trail_edges.items():
        topology.add_edge(TrailEdge(u, v, length=-1.0, plane=plane))

    for u, v in deviation_edges:
        topology.add_edge(DeviationEdge(u, v, force=-2.0))

    topology.add_support(NodeSupport(3))

    for node in range(4):
        topology.add_load(NodeLoad(0, [0.0, -1.0, 0.0]))

    # build trails swith auxiliary trails
    topology.build_trails(auxiliary_trails=True)

    # shift auxiliary trails (only one iteration is needed)
    for node in topology.origin_nodes():
        edges = topology.connected_edges(node)
        for edge in edges:
            if not topology.is_indirect_deviation_edge(edge):
                continue
            u, v = edge
            node_other = u if node != u else v
            sequence = topology.node_sequence(node)
            sequence_other = topology.node_sequence(node_other)

            if sequence_other != sequence:
                topology.shift_trail(node, sequence_other)

    return topology
