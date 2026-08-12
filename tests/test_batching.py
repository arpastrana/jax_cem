import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from jax_cem.datastructures import Structure
from jax_cem.equilibrium import EquilibriumModel
from jax_cem.parameters import Parameters

# ==============================================================================
# Helpers
# ==============================================================================


def chain(edges_deviation):
    """
    A four node chain hanging from one support, braced by one deviation edge.
    """
    return Structure(
        nodes=np.arange(4),
        supports=np.array([3]),
        edges_trail=np.array([[0, 1], [1, 2], [2, 3]]),
        edges_deviation=np.array(edges_deviation),
    )


def parameters(force):
    """
    The parameters of a four node chain, with one force on the deviation edge.
    """
    return Parameters(
        xyz_origin=jnp.zeros((1, 3)),
        loads=jnp.tile(jnp.array([0.0, -1.0, 0.0]), (4, 1)),
        forces=jnp.array([force]),
        lengths=jnp.ones(3),
        planes=jnp.zeros((3, 6)),
    )


def stack(*trees):
    """
    Stack a set of pytrees along a new leading axis.
    """
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *trees)


# ==============================================================================
# Tests - Stacked structures
# ==============================================================================


def test_a_stacked_structure_reports_the_counts_of_one():
    """
    The counts read the trailing axes, so a batch axis does not shift them.
    """
    structure = chain([[0, 2]])
    batched = stack(structure, chain([[1, 3]]))

    counts = (
        "num_nodes",
        "num_edges",
        "num_edges_trail",
        "num_edges_deviation",
        "num_trails",
        "num_sequences",
    )

    for count in counts:
        assert getattr(batched, count) == getattr(structure, count), count


def test_a_stacked_structure_keeps_the_edge_order():
    """
    The edges concatenate along the edge axis, not the batch axis.
    """
    structures = chain([[0, 2]]), chain([[1, 3]])
    edges = jax.vmap(lambda structure: structure.edges)(stack(*structures))

    assert edges.shape == (2, 4, 2)
    for index, structure in enumerate(structures):
        assert np.array_equal(edges[index], structure.edges)


# ==============================================================================
# Tests - Transformations
# ==============================================================================


def test_vmap_over_structures_matches_one_call_at_a_time():
    """
    A batch of structures in equilibrium equals the structures in equilibrium.
    """
    model = EquilibriumModel(tmax=10)
    structures = chain([[0, 2]]), chain([[1, 3]])
    params = parameters(0.5), parameters(-0.5)

    batched = jax.vmap(model)(stack(*params), stack(*structures))

    for index, pair in enumerate(zip(params, structures, strict=True)):
        assert np.allclose(batched.xyz[index], model(*pair).xyz)


def test_vmap_over_parameters_matches_one_call_at_a_time():
    """
    One structure carries a batch of parameters.
    """
    model = EquilibriumModel(tmax=10)
    structure = chain([[0, 2]])
    params = parameters(0.5), parameters(-0.5)

    batched = jax.vmap(lambda p: model(p, structure))(stack(*params))

    for index, param in enumerate(params):
        assert np.allclose(batched.xyz[index], model(param, structure).xyz)


def test_jit_matches_the_eager_call():
    """
    The structure passes through a jit boundary as a pytree.
    """
    model = EquilibriumModel(tmax=10)
    structure = chain([[0, 2]])
    params = parameters(0.5)

    assert np.allclose(
        jax.jit(model)(params, structure).xyz,
        model(params, structure).xyz,
    )


def test_the_gradient_of_a_batch_is_finite():
    """
    Differentiating through the iterative equilibrium under vmap stays finite.
    """
    model = EquilibriumModel(tmax=10)
    structure = chain([[0, 2]])

    def loss(length):
        params = parameters(0.5)
        lengths = params.lengths.at[0].set(length)

        return jnp.sum(model(params._replace(lengths=lengths), structure).xyz ** 2)

    gradients = jax.vmap(jax.grad(loss))(jnp.array([1.0, 2.0]))

    assert np.all(np.isfinite(gradients))
    assert np.all(gradients != 0.0)


# ==============================================================================
# Tests - Rejected inputs
# ==============================================================================


def test_the_constructor_rejects_a_batched_edge_array():
    """
    The constructor runs on the host, so a batch of structures is built one at a
    time and stacked afterwards.

    The suite installs jaxtyping's import hook, which rejects the shape before the
    guard in the constructor reaches it. Both are a rejection, and the guard is
    what a caller without the hook meets.
    """
    with pytest.raises((TypeCheckError, ValueError)):
        Structure(
            nodes=np.arange(4),
            supports=np.array([3]),
            edges_trail=np.stack([np.array([[0, 1], [1, 2], [2, 3]])] * 2),
            edges_deviation=np.stack([np.array([[0, 2]])] * 2),
        )
