# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Added `jax_cem.datastructures.trails`, a native trail search that replaces the
  reliance on `TopologyDiagram.build_trails`. `trails_from_edges` orders the nodes of
  a structure into trails and `sequences_from_trails` lays those trails out into
  sequences.
- Added `align_trails`, which starts every trail at the sequence of the node it
  deviates to and returns a new structure, mirroring `shift_trail` on a topology
  diagram. One pass makes an origin node's deviation edges direct without making
  every edge in the structure direct.
- Added `EquilibriumStructure.__check_init__`, which rejects self-loops, a support
  count that does not match the trail count, and out-of-range supports.
- Added `connectivity_matrix` to `jax_cem.datastructures.structures`.
- Added `beartype` to the dev group and a root `conftest.py` that installs
  jaxtyping's import hook, so every test run checks the shape annotations against
  what the code returns. The hook cannot be wired through the pytest `addopts`: the
  plugin parses the raw command line before the ini options merge into it.
- Added `tests/test_batching.py`, which pins the batching contract: a structure is
  built one at a time on the host, stacked into a pytree, and batched with `vmap`.
- Added `docs/roadmap.md`, recording the design decisions of the refresh.
- Added a `compas_xcheck` marker for tests that cross-check against COMPAS CEM.
- Added `tests/converters.py`, holding the COMPAS CEM topology diagram converters
  that the test fixtures need. Scaffolding, deleted once the native array
  constructor replaces it.

### Changed

- Changed every array annotation in the package to a jaxtyping shape, replacing the
  `# N x 3` comments and the bare `jax.Array` declarations. The shapes were verified
  by running the suite under jaxtyping's import hook with a runtime typechecker,
  which turns each annotation into a check.
- Changed the fields of `EquilibriumStructure` from NumPy to JAX arrays. The trail
  search still computes with NumPy and `sequence_data` converts once on the way out,
  so the shipped structure is device-resident.
- Changed `EquilibriumStructure` to require an edge array of node key pairs. It used
  to reshape whatever it was given, which let a flat array through and made the
  declared shape unenforceable. That reshape also flattened a batch of structures
  into one, which then failed inside the trail search with an error about the node
  keys; a batched input is now rejected where it enters.
- Changed the counts on `EquilibriumStructure` to read the trailing axes rather than
  the leading one, and `edges` to concatenate on the edge axis. On a stacked
  structure the counts used to return the size of the batch.
- Changed the `number_of_*` count methods to `num_*` properties, following `jax_fdm`:
  `num_nodes`, `num_edges`, `num_edges_trail`, `num_edges_deviation`, `num_trails`,
  and `num_sequences`. The two edge counts take the word order of the `edges_trail`
  and `edges_deviation` fields they count.
- Changed the license from MIT to Apache-2.0.
- Changed the packaging to `pyproject.toml` with `uv`, replacing `setup.py`,
  `setup.cfg`, `MANIFEST.in`, `.bumpversion.cfg`, and the `requirements` files.
- Changed the package layout from `src/jax_cem` to a top-level `jax_cem`.
- Changed the linter and formatter to `ruff`, replacing `flake8`, `black`,
  `isort`, `autopep8`, and `pydocstyle`.
- Changed continuous integration to `test.yml` and `publish.yml`, replacing the
  `compas-actions` workflows, and added a `pyright` job to `pr-checks.yml`.
- Changed the supported Python range to 3.11 through 3.13.
- Changed `connectivity_matrix` to take the node count, fixing an `IndexError`
  raised when the highest-indexed node touches no edge.
- Changed the annotations that misreported their runtime type, which makes `pyright`
  clean: `EquilibriumModel` took the empty `Structure` base class where it reads
  `EquilibriumStructure` attributes, the `vmap`-ed `index` parameters of
  `node_equilibrium` and `node_length_plane` were declared `int` where they receive a
  traced scalar, and `node_index`, `edge_index`, `sequences_edges` and
  `sequences_edges_indices` were declared `jax.Array` where they hold dicts and
  NumPy arrays.
- Changed the test suite to `pytest-lazy-fixtures`, replacing the abandoned
  `pytest-lazy-fixture` and its `pytest<8` ceiling.
- Changed `EquilibriumStructure` to take four arrays: `nodes`, `supports`,
  `edges_trail`, and `edges_deviation`, down from eight arguments. Trail and deviation
  edges are held apart rather than as one edge array and a mask, `edges` concatenates
  them with trail edges first, and everything else is derived. `align_trails` applies
  a shift with `equinox.tree_at`, so no shift reaches the constructor.
- Changed `support_nodes` to `supports`, matching the Formax contract.
- Changed `indirect_edges` to `edges_deviation_direct`, which is what the mask holds.
- Removed `incidence` in favour of negating `connectivity` at its one call site, since
  the two were the same array up to sign.

### Removed

- Removed `EquilibriumStructure.trail_edges` and `EquilibriumStructure.deviation_edges`.
  Nothing read the first, and their names differed from the `edges_trail` and
  `edges_deviation` fields only in word order. The second was a mask over a
  contiguous block, so `node_equilibrium` now slices the deviation edges out of the
  connectivity instead of multiplying every edge by a zero, and `nodes_equilibrium`
  builds edge vectors for that block alone rather than for every edge.
- Removed the `keepdims` flag of `vector_length`. Nothing passed it, and it changed
  the return shape, which left the function without one shape to declare.
- Removed `trail_length` from `jax_cem.equilibrium.models`. It was unreachable and
  its comment described a shape it did not return.
- Removed `EquilibriumStructure.from_topology_diagram` and
  `ParameterState.from_topology_diagram`. They made `compas_cem` a hard import of
  `jax_cem.parameters`; the conversion now lives in `tests/converters.py` until the
  native array constructor replaces it.
- Removed the dependency on `compas`. The library no longer imports
  `compas.numerical` or `compas.utilities`, which lifts the `compas==1.17.10` pin.
- Removed the Sphinx documentation sources, `tasks.py`, and the `temp`, `scripts`,
  and `data` placeholder directories.
- Removed the `HOME`, `DATA`, `DOCS`, and `TEMP` globals from `jax_cem`.
