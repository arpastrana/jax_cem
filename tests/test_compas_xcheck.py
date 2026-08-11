"""
Cross-check the equilibrium of a structure against COMPAS CEM, solver to solver.

The baselines in `test_equilibrium` are literals captured once, at one setting.
These run `compas_cem.equilibrium.static_equilibrium` on the same topology, at the
same iteration limit and convergence threshold, and compare every node and every
edge, so a divergence that only shows up while iterating has somewhere to surface.

The equilibrium invariants at the end call no solver of their own. `test_kernel`
holds them too, over structures built without COMPAS CEM; they run here as well
because these six fixtures are the richest ones available, and the two meet once
the fixtures are ported.
"""

import numpy as np
import pytest
from compas_cem.equilibrium import static_equilibrium
from converters import parameters_from_topology
from converters import structure_from_topology
from pytest_lazy_fixtures import lf

from jax_cem.equilibrium import EquilibriumModel

pytestmark = pytest.mark.compas_xcheck

TMAX = 100
ETA = 1e-6

TOPOLOGIES = [
    lf("compression_strut"),
    lf("tension_chain"),
    lf("compression_chain"),
    lf("threebar_funicular"),
    lf("braced_tower_2d"),
    lf("topology_shifted_sequences"),
]

# ==============================================================================
# Helpers
# ==============================================================================


def equilibrate(topology):
    """
    Solve one topology with both implementations, at the same solver settings.
    """
    form = static_equilibrium(topology, tmax=TMAX, eta=ETA, verbose=False)

    structure = structure_from_topology(topology)
    params = parameters_from_topology(topology, structure)
    eqstate = EquilibriumModel(tmax=TMAX, eta=ETA)(params, structure)

    return form, structure, eqstate


def edge_indices(structure):
    """
    Map an edge of a form diagram onto its row in the arrays of a structure.

    Notes
    -----
    A diagram orders its edges as they were added and may hold either node of an
    edge first, while the structure puts every trail edge before every deviation
    edge, so the two orders agree only by lookup.
    """
    index = structure.edge_index

    return lambda u, v: index.get((u, v), index.get((v, u)))


# ==============================================================================
# Tests - Node positions
# ==============================================================================


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_the_node_positions_match_compas_cem(topology):
    """
    Every node lands where COMPAS CEM puts it.
    """
    form, _, eqstate = equilibrate(topology)

    xyz = np.asarray(eqstate.xyz)
    for node in form.nodes():
        assert np.allclose(form.node_coordinates(node), xyz[node]), node


# ==============================================================================
# Tests - Edge forces and lengths
# ==============================================================================


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_the_edge_forces_match_compas_cem(topology):
    """
    Every edge carries the force COMPAS CEM finds in it.

    Notes
    -----
    A deviation force is an input to both solvers, so this pins the edge order
    the structure lays out as much as the force the trail edges recover.
    """
    form, structure, eqstate = equilibrate(topology)
    index_of = edge_indices(structure)

    forces = np.asarray(eqstate.forces)
    for u, v in form.edges():
        assert np.allclose(form.edge_force((u, v)), forces[index_of(u, v)]), (u, v)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_the_edge_lengths_match_compas_cem(topology):
    """
    Every edge is as long as it is in COMPAS CEM.
    """
    form, structure, eqstate = equilibrate(topology)
    index_of = edge_indices(structure)

    lengths = np.asarray(eqstate.lengths)
    for u, v in form.edges():
        assert np.allclose(form.edge_length((u, v)), lengths[index_of(u, v)]), (u, v)


# ==============================================================================
# Tests - Residuals
# ==============================================================================


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_the_support_reactions_match_compas_cem(topology):
    """
    The reaction at a support matches the one COMPAS CEM reports there.

    Notes
    -----
    A reaction is the negation of a residual, and COMPAS CEM reports a reaction,
    so the comparison carries the sign change.
    """
    form, structure, eqstate = equilibrate(topology)

    reactions = -np.asarray(eqstate.residuals)
    for node in np.asarray(structure.supports):
        node = int(node)
        assert np.allclose(form.reaction_force(node), reactions[node]), node


# ==============================================================================
# Tests - Equilibrium invariants
# ==============================================================================


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_the_free_nodes_are_in_equilibrium(topology):
    """
    The residual vanishes at every node no support holds.
    """
    _, structure, eqstate = equilibrate(topology)

    nodes = np.arange(int(structure.num_nodes))
    free = np.setdiff1d(nodes, np.asarray(structure.supports))

    assert np.allclose(np.asarray(eqstate.residuals)[free], 0.0, atol=1e-6)


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_the_structure_carries_its_loads_to_the_supports(topology):
    """
    The residuals sum to the applied load, so the supports absorb all of it.

    Notes
    -----
    The internal forces cancel in pairs over the whole structure, which leaves
    the loads. No single step of the solver enforces that sum.
    """
    _, _, eqstate = equilibrate(topology)

    residuals = np.asarray(eqstate.residuals)
    loads = np.asarray(eqstate.loads)

    assert np.allclose(residuals.sum(axis=0), loads.sum(axis=0), atol=1e-6)
