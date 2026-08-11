# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Added `EquilibriumState.vectors`, the vector of every edge, which the state used
  to leave the caller to recompute from the node positions.
- Added `EquilibriumStructure.is_edge_deviation_direct`, a property that masks the
  deviation edges whose two nodes share a sequence. It replaces a stored field:
  which edges those are follows from the sequences, so deriving it means it cannot
  fall out of step with a trail that shifts.
- Added `EquilibriumModel.nodes_deviation`, which accumulates the deviation force at
  every node, and `EquilibriumModel.nodes_residual`, which assembles the residual at
  every node from the edge forces and the loads.
- Added an edge range check to `EquilibriumStructure.__check_init__`. `scipy` used to
  reject a negative node key on the caller's behalf while building the connectivity
  matrix; without the matrix a negative key would wrap and the accumulation would drop
  the edge, giving a wrong equilibrium with no error.
- Added `tests/test_kernel.py`, which pins the direction of a deviation force, the
  padding of a sequence a trail does not reach, and the edge range check. The frozen
  baselines catch a swapped force direction only indirectly.
- Added `tests/test_compas_xcheck.py`, which solves each fixture with
  `compas_cem.equilibrium.static_equilibrium` at the same iteration limit and
  convergence threshold and compares every node and every edge, giving the marker a
  test to carry. The frozen baselines record one setting; a live solver also covers
  the path between the settings. Verified to bite by swapping the two scatters of the
  deviation accumulation, which fails eleven of the thirty.
- Added two equilibrium invariants, to `tests/test_kernel.py` over structures built
  without COMPAS CEM and to `tests/test_compas_xcheck.py` over the six fixtures: the
  residual vanishes at every free node, and the residuals sum to the applied load.
  Neither calls a second solver. The first measures convergence, since the residual is
  assembled from the edge forces rather than carried out of the sweep: on the braced
  tower it reads 1.9 after one pass and 4e-9 once converged. The second holds to
  machine precision at any iteration count, because the internal forces cancel in
  pairs whether or not the sweep has settled, so it checks the bookkeeping instead.
- Added `jax_cem.datastructures.sequences`, holding `SequenceData`, `build_sequences`,
  and `sequences_from_trails`, which `jax_cem.datastructures.trails` used to carry.
  The trail search orders the nodes; laying that order out into the sequences the
  equilibrium scan steps through is a separate concern, and only the second one
  depends on the shifts.
- Added `jax_cem.datastructures.trails`, a native trail search that replaces the
  reliance on `TopologyDiagram.build_trails`. `build_trails` takes the name of the
  method it stands in for, and orders the nodes of a structure into trails;
  `build_sequences` derives everything that the layout of those trails settles.
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

- Changed the equilibrium kernel from matrix products to gathers and scatters. Edge
  vectors are now `xyz[edges[:, 1]] - xyz[edges[:, 0]]`, and the deviation force is a
  `segment_sum` over the deviation edges rather than a dot product per node. Timed on
  synthetic trail grids with `jax.jit`: 0.47 ms to 0.12 ms at 100 nodes, 50 ms to
  1.4 ms at 900, and 966 ms to 11 ms at 3600. The old form cost `edges * nodes` in
  time and in memory, so the matrix was also the largest array the structure carried.
  The six regression fixtures come out identical; a scatter reduces in an order a dot
  product need not follow, so equality is to rounding rather than to the bit in
  general, and double precision is what keeps the difference far below the tolerance
  the baselines are compared at.
- Changed `edges_vector` to take the edges of a graph rather than its connectivity
  matrix. This half of the conversion is exact: the matrix product added a zero for
  every node an edge does not touch, and adding an exact zero cannot perturb a float.
- Changed `sequences_equilibrium` to resolve `use_indirect` once, before the scan,
  into the force array it selects. The flag reached three signatures and was consulted
  at every node of every sequence, though it settles the same question every time.
- Changed `EquilibriumState` to the Formax field set: `residuals` in place of
  `reactions` and defined at every node rather than only at the supports, an added
  `vectors`, and `forces` and `lengths` shaped `"edges"` rather than `"edges 1"`. The
  container stays a named tuple where Formax uses an equinox module.
- Changed `EquilibriumState.residuals` to the quantity Formax and `jax_fdm` mean by
  the name: the load at a node plus the forces the edges meeting it exert, which is
  zero at a free node in equilibrium. It used to hold the trail residual the sweep
  carries, which is nonzero almost everywhere and is the negation of a residual where
  the two coincide, at a support. Reading it as a residual is what made the earlier
  claim here that COMPAS CEM stores a residual under a reaction's name; COMPAS CEM
  stores a reaction. The frozen baselines are unmoved, since the converter that writes
  them now negates.
- Changed the trail residual to the name `residuals_trail`, on
  `EquilibriumSequenceState` and throughout the kernel, so that `residuals` means one
  thing across the Formax libraries. The trail residual stays off `EquilibriumState`:
  the force in a trail edge and the vector of that edge already hold it.
- Changed `deviation_vector` to `resultant_vector`, which takes the edge set it
  accumulates over. Nothing in it was specific to a deviation edge, and the nodal
  residual needs the same accumulation over every edge.
- Changed `ParameterState.forces` to cover the deviation block alone, shaped
  `"edges_deviation"`. The trail entries were overwritten on every call and never read.
- Changed `vector_length` to return a scalar rather than a one-element vector, which
  is what flattens the per-edge quantities of the state.
- Changed `build_sequences` to take the trails, the edges and the shifts alone. The
  node and trail edge counts were only needed by the mask that is now derived.
- Changed every array annotation in the package to a jaxtyping shape, replacing the
  `# N x 3` comments and the bare `jax.Array` declarations. The shapes were verified
  by running the suite under jaxtyping's import hook with a runtime typechecker,
  which turns each annotation into a check.
- Changed the fields of `EquilibriumStructure` from NumPy to JAX arrays. The trail
  search still computes with NumPy and `build_sequences` converts once on the way out,
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

- Removed `EquilibriumStructure.connectivity` and `connectivity_matrix`, which the
  gather and scatter kernel leaves with no reader. `connectivity_matrix` was public
  through the star import of `jax_cem.datastructures`, which now carries an `__all__`.
- Removed `EquilibriumModel.node_equilibrium`. With the accumulation done for every
  node at once, the node index only selects a row, and keeping the function would have
  forced the vectorization to stay.
- Removed `SequenceData.edges_deviation_direct` and the structure field it fed, in
  favour of the derived `is_edge_deviation_direct`. The stored mask spanned every edge
  though its whole trail half was zero by construction.
- Removed the dependency on `scipy`, which only `connectivity_matrix` imported.
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
