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
- Added `docs/roadmap.md`, recording the design decisions of the refresh.
- Added a `compas_xcheck` marker for tests that cross-check against COMPAS CEM.
- Added `tests/converters.py`, holding the COMPAS CEM topology diagram converters
  that the test fixtures need. Scaffolding, deleted once the native array
  constructor replaces it.

### Changed

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

- Removed `EquilibriumStructure.from_topology_diagram` and
  `ParameterState.from_topology_diagram`. They made `compas_cem` a hard import of
  `jax_cem.parameters`; the conversion now lives in `tests/converters.py` until the
  native array constructor replaces it.
- Removed the dependency on `compas`. The library no longer imports
  `compas.numerical` or `compas.utilities`, which lifts the `compas==1.17.10` pin.
- Removed the Sphinx documentation sources, `tasks.py`, and the `temp`, `scripts`,
  and `data` placeholder directories.
- Removed the `HOME`, `DATA`, `DOCS`, and `TEMP` globals from `jax_cem`.
