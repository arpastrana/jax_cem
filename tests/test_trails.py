import jax.numpy as jnp
import numpy as np
import pytest
from converters import structure_from_topology
from pytest_lazy_fixtures import lf

from jax_cem.datastructures import align_trails
from jax_cem.datastructures import gather_padded
from jax_cem.datastructures import indices_beyond
from jax_cem.datastructures import is_edge_deviation_direct
from jax_cem.datastructures import search_trails
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

    trail_nodes = search_trails(nodes, supports, edges_trail)

    # the shift of a trail is a choice the diagram made, so read it off it
    shifts = [int(np.nonzero(reference == trail[0])[0][0]) for trail in trail_nodes]

    sequences, origin_nodes = sequences_from_trails(trail_nodes, np.asarray(shifts))

    assert sequences.shape == reference.shape
    assert columns(sequences) == columns(reference)
    assert sorted(origin_nodes.tolist()) == sorted(trail[0] for trail in trail_nodes)


def test_every_node_lands_on_exactly_one_trail(braced_tower_2d):
    """
    The trails partition the nodes.
    """
    nodes, supports, edges_trail, _ = trail_inputs(braced_tower_2d)
    trail_nodes = search_trails(nodes, supports, edges_trail)

    visited = [node for trail in trail_nodes for node in trail]
    assert sorted(visited) == sorted(int(node) for node in nodes)


def test_a_trail_runs_from_origin_to_support(braced_tower_2d):
    """
    The last node of every trail is its support.
    """
    nodes, supports, edges_trail, _ = trail_inputs(braced_tower_2d)
    trail_nodes = search_trails(nodes, supports, edges_trail)

    assert sorted(trail[-1] for trail in trail_nodes) == sorted(
        int(s) for s in supports
    )


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
    nodes = np.asarray(structure.trails.sequences.nodes)
    edges = np.asarray(structure.trails.sequences.trail_edge_index)
    edges_trail = np.asarray(structure.edges_trail)

    for row, pair in enumerate(zip(nodes[:-1], nodes[1:])):
        sequence, sequence_next = pair
        for column, edge in enumerate(edges[row]):
            if edge < 0:
                continue
            assert set(edges_trail[edge]) == {sequence[column], sequence_next[column]}


def test_the_origin_nodes_agree_with_the_layout_that_holds_them(
    topology_shifted_sequences,
):
    """
    The stored origin nodes are the first node every column of the layout holds.

    Notes
    -----
    They are read off the trails and the layout is built from the same trails, so
    this is the one pin that the two cannot state different things.
    """
    structure = structure_from_topology(topology_shifted_sequences)
    nodes = np.asarray(structure.trails.sequences.nodes)
    origins = np.asarray(structure.origin_nodes)

    first = np.argmax(nodes >= 0, axis=0)

    assert np.array_equal(origins, nodes[first, np.arange(nodes.shape[-1])])


def test_a_shift_leaves_the_origin_nodes_alone(topology_shifted_sequences):
    """
    A shift moves a trail down the sequences without reordering it.
    """
    structure = structure_from_topology(topology_shifted_sequences)
    aligned = align_trails(structure)

    assert (
        aligned.trails.sequences.nodes.shape != structure.trails.sequences.nodes.shape
    )
    assert np.array_equal(
        np.asarray(structure.origin_nodes),
        np.asarray(aligned.origin_nodes),
    )


def test_the_slot_of_an_edge_reads_the_edge_back(topology_shifted_sequences):
    """
    The two directions of the slot map agree, on a layout that pads and shifts.
    """
    structure = structure_from_topology(topology_shifted_sequences)
    edges = np.asarray(structure.trails.sequences.trail_edge_index)
    slots = np.asarray(structure.trails.trail_edge_index)
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

    assert aligned.trails.sequences.nodes.shape == reference.shape
    assert columns(np.asarray(aligned.trails.sequences.nodes)) == columns(reference)


def test_aligning_twice_changes_nothing(topology_shifted_sequences):
    """
    Alignment reads the trails, not the shift already applied to them.
    """
    once = align_trails(structure_from_topology(topology_shifted_sequences))
    twice = align_trails(once)

    assert np.array_equal(
        np.asarray(once.trails.sequences.nodes),
        np.asarray(twice.trails.sequences.nodes),
    )


def test_align_trails_leaves_an_aligned_structure_alone(threebar_funicular):
    """
    A structure whose deviation edges are all direct needs no shift.
    """
    structure = structure_from_topology(threebar_funicular)
    aligned = align_trails(structure)

    assert np.array_equal(
        np.asarray(structure.trails.sequences.nodes),
        np.asarray(aligned.trails.sequences.nodes),
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
        np.asarray(aligned.trails.sequences.nodes).shape[0]
        > np.asarray(structure.trails.sequences.nodes).shape[0]
    )
    assert indirect(aligned) == indirect(structure)


def test_align_trails_returns_a_new_structure(topology_shifted_sequences):
    """
    The transform does not touch the structure it is given.
    """
    structure = structure_from_topology(topology_shifted_sequences)
    before = np.asarray(structure.trails.sequences.nodes).copy()

    align_trails(structure)

    assert np.array_equal(np.asarray(structure.trails.sequences.nodes), before)


# ==============================================================================
# Tests - Padded slots
# ==============================================================================


def test_a_padded_slot_is_sent_past_the_end():
    """
    The sentinel leaves the index space, and a real index is left alone.
    """
    indices = jnp.asarray([1, -1, 0])

    assert np.asarray(indices_beyond(indices, 2)).tolist() == [1, 2, 0]


def test_a_padded_slot_reads_nothing():
    """
    A gather takes zero at a padded slot and the addressed row everywhere else.
    """
    values = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])

    gathered = np.asarray(gather_padded(values, jnp.asarray([1, -1, 0])))

    assert np.allclose(gathered, [[3.0, 4.0], [0.0, 0.0], [1.0, 2.0]])


def test_a_padded_slot_writes_nothing():
    """
    A scatter through a padded slot leaves every row of the target alone.
    """
    values = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    padded = indices_beyond(jnp.asarray([-1, -1]), values.shape[0])

    written = values.at[padded, :].set(jnp.full((2, 2), 9.0), mode="drop")

    assert np.allclose(np.asarray(written), np.asarray(values))


def test_a_negative_index_reads_a_real_row_even_when_filled():
    """
    The sentinel cannot stay an index, which is why it is sent out of bounds.

    Notes
    -----
    A fill catches an index past the end and not a negative one, which wraps onto
    the last row first. The kernel rests on that, so it is pinned here rather than
    assumed of a dependency.
    """
    values = jnp.asarray([[1.0], [2.0]])

    wrapped = jnp.take(values, jnp.asarray([-1]), axis=0, mode="fill", fill_value=0.0)

    assert np.allclose(np.asarray(wrapped), [[2.0]])


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
        search_trails(nodes, supports, edges_trail)


def test_a_branching_trail_is_rejected():
    """
    A trail is a path, so a node cannot continue into two trail edges.
    """
    nodes = np.array([0, 1, 2])
    supports = np.array([1])
    edges_trail = np.array([[0, 1], [1, 2], [0, 2]])

    with pytest.raises(ValueError, match="ambiguous"):
        search_trails(nodes, supports, edges_trail)


def test_no_supports_is_rejected():
    """
    Trails grow from supports, so there must be one.
    """
    with pytest.raises(ValueError, match="No supports"):
        search_trails(np.array([0, 1]), np.empty(0, dtype=int), np.array([[0, 1]]))


def test_no_trail_edges_is_rejected():
    """
    A structure of deviation edges alone has no trail to step through.
    """
    with pytest.raises(ValueError, match="No trail edges"):
        search_trails(np.array([0, 1]), np.array([1]), np.empty((0, 2), dtype=int))


def test_a_self_looping_trail_edge_is_rejected():
    """
    A trail edge must join two distinct nodes.
    """
    with pytest.raises(ValueError, match="self-loop"):
        search_trails(np.array([0, 1]), np.array([1]), np.array([[0, 0], [0, 1]]))


def test_mismatched_shifts_are_rejected(braced_tower_2d):
    """
    One shift per trail, no more and no fewer.
    """
    nodes, supports, edges_trail, _ = trail_inputs(braced_tower_2d)
    trail_nodes = search_trails(nodes, supports, edges_trail)

    with pytest.raises(ValueError, match="they must match"):
        sequences_from_trails(trail_nodes, np.array([0]))


def test_a_negative_shift_is_rejected(braced_tower_2d):
    """
    A trail cannot start before the first sequence.
    """
    nodes, supports, edges_trail, _ = trail_inputs(braced_tower_2d)
    trail_nodes = search_trails(nodes, supports, edges_trail)

    with pytest.raises(ValueError, match="before the first sequence"):
        sequences_from_trails(trail_nodes, np.array([-1, 0]))
