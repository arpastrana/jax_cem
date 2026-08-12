import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from jax_cem.datastructures import Structure
from jax_cem.datastructures import is_edge_deviation_direct
from jax_cem.equilibrium import EquilibriumModel
from jax_cem.equilibrium.models import nodes_deviation_force
from jax_cem.parameters import Parameters

# ==============================================================================
# Helpers
# ==============================================================================


def two_trails():
    """
    Two one-edge trails, their origin nodes joined by one deviation edge.
    """
    return Structure(
        nodes=np.arange(4),
        supports=np.array([1, 3]),
        edges_trail=np.array([[0, 1], [2, 3]]),
        edges_deviation=np.array([[0, 2]]),
    )


def uneven_trails():
    """
    A two-edge trail beside a one-edge trail, which pads the sequences.
    """
    return Structure(
        nodes=np.arange(5),
        supports=np.array([2, 4]),
        edges_trail=np.array([[0, 1], [1, 2], [3, 4]]),
        edges_deviation=np.array([[0, 3]]),
    )


def crossing_trails():
    """
    Uneven trails joined by one direct and one indirect deviation edge.
    """
    return Structure(
        nodes=np.arange(5),
        supports=np.array([2, 4]),
        edges_trail=np.array([[0, 1], [1, 2], [3, 4]]),
        edges_deviation=np.array([[0, 3], [1, 3]]),
    )


def parameters(structure):
    """
    A downward unit load at every node, with the origin nodes spread along x.

    Notes
    -----
    Every trail edge takes a unit length. The origins are held apart so that no
    deviation edge starts out with zero length.
    """
    num_nodes = int(structure.num_nodes)
    num_edges_trail = int(structure.num_edges_trail)
    num_trails = int(structure.num_trails)

    xyz_origin = np.zeros((num_trails, 3))
    xyz_origin[:, 0] = np.arange(num_trails)

    return Parameters(
        xyz_origin=jnp.asarray(xyz_origin),
        loads=jnp.tile(jnp.array([0.0, -1.0, 0.0]), (num_nodes, 1)),
        forces=jnp.full(int(structure.num_edges_deviation), 0.5),
        lengths=jnp.ones(num_edges_trail),
        planes=jnp.zeros((num_edges_trail, 6)),
    )


STRUCTURES = [two_trails, uneven_trails, crossing_trails]


# ==============================================================================
# Tests - Deviation forces
# ==============================================================================


def test_a_deviation_force_pushes_a_node_toward_the_far_end():
    """
    A tension force pulls each node of a deviation edge toward the other.

    Notes
    -----
    The two scatters of the accumulation carry opposite signs. Swapping them
    leaves every magnitude intact, so this pins the direction directly rather
    than through an equilibrium the baselines check.
    """
    structure = two_trails()
    xyz = jnp.array(
        [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [1.0, -1.0, 0.0]],
    )

    deviation_forces = nodes_deviation_force(structure, xyz, jnp.array([1.0]))

    # node 0 sits at the tail of the edge and node 2 at its head
    assert np.allclose(deviation_forces[0], [1.0, 0.0, 0.0])
    assert np.allclose(deviation_forces[2], [-1.0, 0.0, 0.0])
    # the support nodes carry no deviation edge
    assert np.allclose(np.asarray(deviation_forces)[[1, 3]], 0.0)


def test_an_indirect_deviation_edge_is_left_out_of_the_first_pass():
    """
    The first pass sees only the deviation edges that stay within a sequence.
    """
    structure = uneven_trails()
    direct = np.asarray(is_edge_deviation_direct(structure))

    assert direct.shape == (structure.num_edges_deviation,)
    assert direct.dtype == bool
    assert direct.all()


def test_a_deviation_edge_across_two_sequences_is_indirect():
    """
    The mask tells the two kinds of deviation edge apart, edge by edge.

    Notes
    -----
    Nothing else pins which edges those are. A mask that reported every edge
    direct would leave the equilibrium unchanged, since the classification only
    holds edges out of the first pass and the iteration puts them back, so this
    reads the sequence of a node rather than an equilibrium computed from it.
    Verified to bite by putting every node in the same sequence.
    """
    structure = crossing_trails()
    direct = np.asarray(is_edge_deviation_direct(structure))

    # edge (0, 3) joins two origin nodes; edge (1, 3) reaches back a sequence
    assert direct.tolist() == [True, False]


# ==============================================================================
# Tests - Nodal equilibrium
# ==============================================================================


@pytest.mark.parametrize("build", STRUCTURES, ids=lambda build: build.__name__)
def test_the_free_nodes_are_in_equilibrium(build):
    """
    The residual vanishes at every node no support holds.

    Notes
    -----
    The residual is assembled from the edge forces rather than carried out of the
    sweep, so this closes the loop on the whole kernel: the trail edge forces it
    recovers, the deviation forces it applies, and the positions it lands on have
    to balance the loads node by node.
    """
    structure = build()
    state = EquilibriumModel(tmax=100)(parameters(structure), structure)

    nodes = np.arange(int(structure.num_nodes))
    free = np.setdiff1d(nodes, np.asarray(structure.supports))

    assert np.allclose(np.asarray(state.residuals)[free], 0.0, atol=1e-6)


@pytest.mark.parametrize("build", STRUCTURES, ids=lambda build: build.__name__)
def test_the_structure_carries_its_loads_to_the_supports(build):
    """
    The residuals sum to the applied load, so the supports absorb all of it.

    Notes
    -----
    The internal forces cancel in pairs over the whole structure, which leaves
    the loads. No single step of the solver enforces that sum.
    """
    structure = build()
    state = EquilibriumModel(tmax=100)(parameters(structure), structure)

    residuals = np.asarray(state.residuals)
    loads = np.asarray(state.loads)

    assert np.allclose(residuals.sum(axis=0), loads.sum(axis=0), atol=1e-6)


@pytest.mark.parametrize("build", STRUCTURES, ids=lambda build: build.__name__)
def test_the_origin_positions_land_on_the_origin_nodes(build):
    """
    The origin positions are column-aligned with the trails, not keyed by node.

    Notes
    -----
    The origins are spread along x, so a column read in the wrong order moves a
    trail rather than leaving the result unchanged.
    """
    structure = build()
    params = parameters(structure)

    state = EquilibriumModel(tmax=100)(params, structure)
    origins = np.asarray(structure.origin_nodes)

    assert np.allclose(np.asarray(state.xyz)[origins], np.asarray(params.xyz_origin))


# ==============================================================================
# Tests - Padded sequences
# ==============================================================================


def test_an_empty_slot_of_the_layout_takes_no_length():
    """
    A slot that no trail edge leaves stretches nothing.

    Notes
    -----
    Every trail edge here is a unit length, so a slot that read one by addressing
    an edge it does not own would report that length rather than nothing. The
    empty slots outnumber the padded nodes, since the last node of a trail holds a
    slot that no edge leaves.
    """
    structure = uneven_trails()
    is_empty = np.asarray(structure.sequence_edges) < 0
    assert np.any(is_empty), "no empty slot to exercise"

    xyz = jnp.zeros((int(structure.num_nodes), 3))
    trails_state = EquilibriumModel().trails_equilibrium(
        parameters(structure),
        structure,
        xyz,
        use_indirect=False,
    )

    assert np.allclose(np.asarray(trails_state.lengths)[is_empty], 0.0)


def test_a_padded_sequence_still_builds_every_trail_edge():
    """
    Trails of unequal length pad the sequences, which must not reach the result.

    Notes
    -----
    The lengths differ per edge, and the slot the short trail leaves empty reads
    the last of them, so a padded slot that escaped the mask would stretch an
    edge rather than leave the result untouched.
    """
    structure = uneven_trails()
    assert np.any(np.asarray(structure.sequence_nodes) < 0), "no padding to exercise"
    assert np.any(np.asarray(structure.sequence_edges) < 0), "no empty slot"

    params = Parameters(
        xyz_origin=jnp.zeros((2, 3)),
        loads=jnp.tile(jnp.array([0.0, -1.0, 0.0]), (5, 1)),
        forces=jnp.array([0.5]),
        lengths=jnp.array([1.0, 2.0, 3.0]),
        planes=jnp.zeros((3, 6)),
    )

    state = EquilibriumModel()(params, structure)
    edge_index = structure.edge_index

    for edge, length in (((0, 1), 1.0), ((1, 2), 2.0), ((3, 4), 3.0)):
        assert np.allclose(state.lengths[edge_index[edge]], length), edge

    assert np.all(np.isfinite(np.asarray(state.xyz)))


# ==============================================================================
# Tests - Trail edge lengths
# ==============================================================================


def test_a_trail_edge_with_a_plane_ignores_its_length():
    """
    A plane drives the edge it is given to, and the length of that edge goes unread.

    Notes
    -----
    The length here would carry the node five units away, and the plane sits two
    units off the origin, so the node lands on one or the other and not on both.
    Verified to bite by asking the length whether it is zero, which is what the
    plane used to be reached through.
    """
    structure = two_trails()
    planes = np.zeros((2, 6))
    planes[0, :] = [0.0, -2.0, 0.0, 0.0, 1.0, 0.0]

    replaced = (jnp.array([5.0, 1.0]), jnp.asarray(planes))
    params = eqx.tree_at(
        lambda tree: (tree.lengths, tree.planes),
        parameters(structure),
        replace=replaced,
    )

    state = EquilibriumModel()(params, structure)

    assert np.allclose(np.asarray(state.xyz)[1, 1], -2.0)


def test_a_normal_shorter_than_the_tolerance_is_no_plane():
    """
    A normal that rounds to zero leaves its edge to the length instead.

    Notes
    -----
    The normal arrives unnormalized, so how short it is says nothing about what it
    means, and only a tolerance separates one meant to be read from one left at
    zero. Verified to bite by comparing the normal against zero exactly, which
    takes this plane for a real one and puts the node on it.
    """
    structure = two_trails()
    planes = np.zeros((2, 6))
    planes[0, :] = [0.0, -2.0, 0.0, 0.0, 1e-12, 0.0]

    replaced = (jnp.array([5.0, 1.0]), jnp.asarray(planes))
    params = eqx.tree_at(
        lambda tree: (tree.lengths, tree.planes),
        parameters(structure),
        replace=replaced,
    )

    state = EquilibriumModel()(params, structure)
    xyz = np.asarray(state.xyz)

    assert np.allclose(np.linalg.norm(xyz[1] - xyz[0]), 5.0)


# ==============================================================================
# Tests - Rejected structures
# ==============================================================================


@pytest.mark.parametrize("far", [-1, 5])
def test_an_edge_that_misses_a_node_is_rejected(far):
    """
    A key outside the nodes would wrap or drop silently in the accumulation.
    """
    with pytest.raises(ValueError, match="index existing nodes"):
        Structure(
            nodes=np.arange(2),
            supports=np.array([1]),
            edges_trail=np.array([[0, 1]]),
            edges_deviation=np.array([[0, far]]),
        )
