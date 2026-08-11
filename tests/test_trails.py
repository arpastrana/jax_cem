import numpy as np
import pytest
from converters import structure_from_topology
from pytest_lazy_fixtures import lf

from jax_cem.datastructures import align_trails
from jax_cem.datastructures import build_trails
from jax_cem.datastructures import is_edge_deviation_direct
from jax_cem.datastructures import sequences_from_trails

# ==============================================================================
# Helpers
# ==============================================================================


def trail_inputs(topology):
    """
    The array inputs the trail search takes, read off a topology diagram.
    """
    structure = structure_from_topology(topology)

    return (
        np.asarray(structure.nodes),
        np.asarray(structure.supports),
        np.asarray(structure.edges_trail),
        np.asarray(structure.edges_deviation),
    )


def reference_sequences(topology):
    """
    The sequences a COMPAS CEM topology diagram laid its own trails into.
    """
    shape = (topology.number_of_sequences(), topology.number_of_trails())
    sequences = np.full(shape, -1, dtype=int)
    for index, (_, trail) in enumerate(topology.trails(True)):
        for node in trail:
            sequences[topology.node_sequence(node)][index] = node

    return sequences


def columns(sequences):
    """
    The trails of a sequences array, order-independent.
    """
    return sorted(tuple(int(node) for node in column) for column in sequences.T)


# ==============================================================================
# Tests - Trail search
# ==============================================================================


@pytest.mark.parametrize(
    "topology",
    [
        lf("compression_strut"),
        lf("tension_chain"),
        lf("compression_chain"),
        lf("threebar_funicular"),
        lf("braced_tower_2d"),
        lf("topology_shifted_sequences"),
    ],
)
def test_trail_search_reproduces_the_reference_sequences(topology):
    """
    The native trail search lays out the nodes as COMPAS CEM does.
    """
    reference = reference_sequences(topology)
    nodes, supports, edges_trail, _ = trail_inputs(topology)

    trails = build_trails(nodes, supports, edges_trail)

    # the shift of a trail is a choice the diagram made, so read it off it
    shifts = [int(np.nonzero(reference == trail[0])[0][0]) for trail in trails]

    sequences, origin_nodes = sequences_from_trails(trails, np.asarray(shifts))

    assert sequences.shape == reference.shape
    assert columns(sequences) == columns(reference)
    assert sorted(origin_nodes.tolist()) == sorted(trail[0] for trail in trails)


def test_every_node_lands_on_exactly_one_trail(braced_tower_2d):
    """
    The trails partition the nodes.
    """
    nodes, supports, edges_trail, _ = trail_inputs(braced_tower_2d)
    trails = build_trails(nodes, supports, edges_trail)

    visited = [node for trail in trails for node in trail]
    assert sorted(visited) == sorted(int(node) for node in nodes)


def test_a_trail_runs_from_origin_to_support(braced_tower_2d):
    """
    The last node of every trail is its support.
    """
    nodes, supports, edges_trail, _ = trail_inputs(braced_tower_2d)
    trails = build_trails(nodes, supports, edges_trail)

    assert sorted(trail[-1] for trail in trails) == sorted(int(s) for s in supports)


# ==============================================================================
# Tests - Sequence layout
# ==============================================================================


def test_a_slot_names_the_edge_that_leaves_its_node(braced_tower_2d):
    """
    The edge of a slot joins the node of that slot to the next one on its trail.

    Notes
    -----
    This fixture holds every trail edge against its trail direction, so an edge
    is matched as a pair of nodes rather than in the order the structure keeps.
    """
    structure = structure_from_topology(braced_tower_2d)
    nodes = np.asarray(structure.sequences.nodes)
    edges = np.asarray(structure.sequences.edges)
    edges_trail = np.asarray(structure.edges_trail)

    for row, pair in enumerate(zip(nodes[:-1], nodes[1:])):
        sequence, sequence_next = pair
        for column, edge in enumerate(edges[row]):
            if edge < 0:
                continue
            assert set(edges_trail[edge]) == {sequence[column], sequence_next[column]}


def test_the_slot_of_an_edge_reads_the_edge_back(topology_shifted_sequences):
    """
    The two directions of the slot map agree, on a layout that pads and shifts.
    """
    structure = structure_from_topology(topology_shifted_sequences)
    edges = np.asarray(structure.sequences.edges)
    slots = np.asarray(structure.sequences.edges_slot)
    trail_edges = np.arange(structure.num_edges_trail)

    # the last sequence a trail reaches holds its support, which no edge leaves
    assert np.all(edges[-1] < 0)
    assert np.array_equal(edges[:-1].ravel()[slots], trail_edges)
    assert np.array_equal(np.sort(edges[edges >= 0]), trail_edges)


# ==============================================================================
# Tests - Trail alignment
# ==============================================================================


def test_align_trails_reproduces_the_reference_layout(topology_shifted_sequences):
    """
    Aligning arrives at the layout the diagram was shifted into by hand.
    """
    reference = reference_sequences(topology_shifted_sequences)
    aligned = align_trails(structure_from_topology(topology_shifted_sequences))

    assert aligned.sequences.nodes.shape == reference.shape
    assert columns(np.asarray(aligned.sequences.nodes)) == columns(reference)


def test_aligning_twice_changes_nothing(topology_shifted_sequences):
    """
    Alignment reads the trails, not the shift already applied to them.
    """
    once = align_trails(structure_from_topology(topology_shifted_sequences))
    twice = align_trails(once)

    assert np.array_equal(
        np.asarray(once.sequences.nodes),
        np.asarray(twice.sequences.nodes),
    )


def test_align_trails_leaves_an_aligned_structure_alone(threebar_funicular):
    """
    A structure whose deviation edges are all direct needs no shift.
    """
    structure = structure_from_topology(threebar_funicular)
    aligned = align_trails(structure)

    assert np.array_equal(
        np.asarray(structure.sequences.nodes),
        np.asarray(aligned.sequences.nodes),
    )


def test_align_trails_can_trade_one_indirect_edge_for_another(braced_tower_2d):
    """
    A single pass shifts a trail without always reducing the indirect edges.
    """
    structure = structure_from_topology(braced_tower_2d)
    aligned = align_trails(structure)

    def indirect(candidate):
        return int(np.sum(~np.asarray(is_edge_deviation_direct(candidate))))

    assert (
        np.asarray(aligned.sequences.nodes).shape[0]
        > np.asarray(structure.sequences.nodes).shape[0]
    )
    assert indirect(aligned) == indirect(structure)


def test_align_trails_returns_a_new_structure(topology_shifted_sequences):
    """
    The transform does not touch the structure it is given.
    """
    structure = structure_from_topology(topology_shifted_sequences)
    before = np.asarray(structure.sequences.nodes).copy()

    align_trails(structure)

    assert np.array_equal(np.asarray(structure.sequences.nodes), before)


# ==============================================================================
# Tests - Rejected topologies
# ==============================================================================


def test_a_node_off_every_trail_is_rejected():
    """
    A node reachable only through deviation edges needs an auxiliary trail.
    """
    nodes = np.array([0, 1, 2])
    supports = np.array([1])
    edges_trail = np.array([[0, 1]])

    with pytest.raises(ValueError, match="do not lie on a trail"):
        build_trails(nodes, supports, edges_trail)


def test_a_branching_trail_is_rejected():
    """
    A trail is a path, so a node cannot continue into two trail edges.
    """
    nodes = np.array([0, 1, 2])
    supports = np.array([1])
    edges_trail = np.array([[0, 1], [1, 2], [0, 2]])

    with pytest.raises(ValueError, match="ambiguous"):
        build_trails(nodes, supports, edges_trail)


def test_no_supports_is_rejected():
    """
    Trails grow from supports, so there must be one.
    """
    with pytest.raises(ValueError, match="No supports"):
        build_trails(np.array([0, 1]), np.empty(0, dtype=int), np.array([[0, 1]]))


def test_no_trail_edges_is_rejected():
    """
    A structure of deviation edges alone has no trail to step through.
    """
    with pytest.raises(ValueError, match="No trail edges"):
        build_trails(np.array([0, 1]), np.array([1]), np.empty((0, 2), dtype=int))


def test_a_self_looping_trail_edge_is_rejected():
    """
    A trail edge must join two distinct nodes.
    """
    with pytest.raises(ValueError, match="self-loop"):
        build_trails(np.array([0, 1]), np.array([1]), np.array([[0, 0], [0, 1]]))


def test_mismatched_shifts_are_rejected(braced_tower_2d):
    """
    One shift per trail, no more and no fewer.
    """
    nodes, supports, edges_trail, _ = trail_inputs(braced_tower_2d)
    trails = build_trails(nodes, supports, edges_trail)

    with pytest.raises(ValueError, match="they must match"):
        sequences_from_trails(trails, np.array([0]))


def test_a_negative_shift_is_rejected(braced_tower_2d):
    """
    A trail cannot start before the first sequence.
    """
    nodes, supports, edges_trail, _ = trail_inputs(braced_tower_2d)
    trails = build_trails(nodes, supports, edges_trail)

    with pytest.raises(ValueError, match="before the first sequence"):
        sequences_from_trails(trails, np.array([-1, 0]))
