import jax.numpy as jnp
import numpy as np
import pytest

from jax_cem.datastructures import EquilibriumStructure
from jax_cem.equilibrium import EquilibriumModel
from jax_cem.parameters import ParameterState

# ==============================================================================
# Helpers
# ==============================================================================


def two_trails():
    """
    Two one-edge trails, their origin nodes joined by one deviation edge.
    """
    return EquilibriumStructure(
        nodes=np.arange(4),
        supports=np.array([1, 3]),
        edges_trail=np.array([[0, 1], [2, 3]]),
        edges_deviation=np.array([[0, 2]]),
    )


def uneven_trails():
    """
    A two-edge trail beside a one-edge trail, which pads the sequences.
    """
    return EquilibriumStructure(
        nodes=np.arange(5),
        supports=np.array([2, 4]),
        edges_trail=np.array([[0, 1], [1, 2], [3, 4]]),
        edges_deviation=np.array([[0, 3]]),
    )


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

    deviations = EquilibriumModel().nodes_deviation(structure, xyz, jnp.array([1.0]))

    # node 0 sits at the tail of the edge and node 2 at its head
    assert np.allclose(deviations[0], [1.0, 0.0, 0.0])
    assert np.allclose(deviations[2], [-1.0, 0.0, 0.0])
    # the support nodes carry no deviation edge
    assert np.allclose(np.asarray(deviations)[[1, 3]], 0.0)


def test_an_indirect_deviation_edge_is_left_out_of_the_first_pass():
    """
    The first pass sees only the deviation edges that stay within a sequence.
    """
    structure = uneven_trails()
    direct = np.asarray(structure.is_edge_deviation_direct)

    assert direct.shape == (structure.num_edges_deviation,)
    assert direct.dtype == bool
    assert direct.all()


# ==============================================================================
# Tests - Padded sequences
# ==============================================================================


def test_a_padded_sequence_still_builds_every_trail_edge():
    """
    Trails of unequal length pad the sequences, which must not reach the result.
    """
    structure = uneven_trails()
    assert np.any(np.asarray(structure.sequences) < 0), "no padding to exercise"

    params = ParameterState(
        xyz=jnp.zeros((5, 3)),
        loads=jnp.tile(jnp.array([0.0, -1.0, 0.0]), (5, 1)),
        forces=jnp.array([0.5]),
        lengths=jnp.array([[1.0], [2.0], [0.0], [3.0], [0.0]]),
        planes=jnp.zeros((5, 6)),
    )

    state = EquilibriumModel()(params, structure)
    edge_index = structure.edge_index

    for edge, length in (((0, 1), 1.0), ((1, 2), 2.0), ((3, 4), 3.0)):
        assert np.allclose(state.lengths[edge_index[edge]], length), edge

    assert np.all(np.isfinite(np.asarray(state.xyz)))


# ==============================================================================
# Tests - Rejected structures
# ==============================================================================


@pytest.mark.parametrize("far", [-1, 5])
def test_an_edge_that_misses_a_node_is_rejected(far):
    """
    A key outside the nodes would wrap or drop silently in the accumulation.
    """
    with pytest.raises(ValueError, match="index existing nodes"):
        EquilibriumStructure(
            nodes=np.arange(2),
            supports=np.array([1]),
            edges_trail=np.array([[0, 1]]),
            edges_deviation=np.array([[0, far]]),
        )
