# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Added `Trails.nodes_sequence`, the sequence that each node is put in equilibrium at,
  inverted in `build_trails` out of the layout that reads a node off a slot. It is the
  quantity `is_edge_deviation_direct` used to rebuild on every call, from a
  `jnp.broadcast_to` of the sequence indices scattered into a `jnp.full` table, so that
  function is now two gathers and a comparison and `jax_cem.datastructures.sequences`
  needs neither `indices_beyond` nor `jnp` at all. It is stored rather than derived
  because it is a function of the layout alone, and a shift replaces the layout whole;
  the mask that reads it stays derived, since that also reads the deviation edges, which
  are replaced apart from it.
- Added `Structure.sequences`, `Structure.sequence_nodes`, `Structure.sequence_edges`,
  `Structure.nodes_sequence`, and `Structure.edges_sequence`, which reach the layout in
  one step where a caller used to write `structure.trails.sequences.nodes`. The two grids
  are flat properties rather than one accessor for the container, since reading a grid off
  that is two steps again. The four maps name the direction they run in, keyed entity
  first: `sequence_nodes` and `sequence_edges` read a node and a trail edge off a slot,
  and `nodes_sequence` and `edges_sequence` read a sequence and a slot back off those.
- Added a test that a deviation edge spanning two sequences is reported indirect, edge by
  edge, on the crossing fixture. Nothing pinned which edges those are: a mask that called
  every edge direct passed all 94 tests, because the classification only holds edges out
  of the first pass and the iteration puts them back, so no equilibrium moves. The count
  in `test_align_trails_can_trade_one_indirect_edge_for_another` compared two numbers that
  both collapsed to zero under that mask, and now asserts the count is nonzero first.
- Added a test that the stored origin nodes are the first node the trail search found,
  which replaces one that compared them against the layout. The origins are now read off
  the layout, so that comparison restated the construction; the search is the reference
  that can still disagree.
- Added five tests for the padding that addresses nothing: that a sentinel leaves the
  index space and a real index does not, that a gather through one reads zero, that a
  scatter through one writes nothing, that a slot no trail edge leaves takes no length,
  and that a negative index still reads a real row even under a fill, which is the
  reason the sentinel has to be sent out of bounds rather than left as an index. The
  last two are the ones that bite: the length test fails under the previous convention,
  which read the last trail edge, and making the sentinel an index again fails ten tests.
- Added `Sequence`, one sequence of a structure: the node of every trail in it and the
  trail edge outgoing from each. It names what the scan steps over, which was an
  anonymous pair of arrays reaching `sequence_equilibrium` as two arguments, and
  `Trails.sequences` stacks every one of them, so the scan is handed the container while
  the step is handed one row of it and reads `sequence.nodes` and `sequence.edges` rather
  than unpacking a tuple by position. Its shapes describe that row, which is the contract
  a stacked structure under `vmap` already carries.
- Added an `__all__` to `jax_cem.equilibrium.models`, holding the model alone, and one
  to `jax_cem.equilibrium.states`, holding the three states, which the modules of
  `jax_cem.datastructures` already carry. The star imports that `jax_cem.equilibrium`
  runs over the two had no list to read, so they published every helper and every
  imported symbol: `jnp`, `vmap`, `scan`, and `segment_sum` from the first, and `Array`,
  `Float`, and `NamedTuple` from the second. The functions stay reachable by their
  module path.
- Added `Sequence.edges`, the trail edge outgoing from the node at each slot of a
  sequence and `-1` where a slot has none. It is the slot-to-edge map `build_trails`
  already builds in order to invert it into `Trails.edges_sequence`, kept rather than
  discarded, so the two directions of one bijection cannot fall out of step with a trail
  that shifts. It sits beside `Sequence.nodes`, which is what lets the scan step over
  the node of a slot and the edge that leaves it together.
- Added a test that the position of every origin node in the equilibrium state is the
  parameter given for it, over the three native structures, which pins that the origin
  positions are read by trail column and not by node key. Verified to bite by reversing
  the columns, which fails fourteen tests.
- Added two tests that the stored origin nodes are the first node every column of the
  layout holds, and that a shift leaves them alone. They are the pin that the layout and
  the origin nodes it is built beside cannot state different things.
- Added two layout tests to `tests/test_trails.py`: the edge of a slot joins the node
  of that slot to the next one on its trail, and the slot of an edge reads that edge
  back. The first runs on the braced tower, which holds every trail edge against its
  trail direction, and the second on the shifted fixture, whose layout both pads and
  shifts. Verified to bite by padding the map at the top rather than the bottom, which
  fails sixteen tests.
- Added `EquilibriumState.vectors`, the vector of every edge, which the state used
  to leave the caller to recompute from the node positions.
- Added `Trails`, which `Structure` holds as its one derived field,
  in place of the four it used to spread across. `SequenceData` was already this grouping
  and was splatted apart on arrival, so the constructor and `align_trails` had to
  replace four fields in step to keep them consistent; a shift rewrites the layout, and
  one field cannot fall out of step with itself.
- Added `is_edge_deviation_direct` as a function, taking a structure, where it was a
  property. It scatters and gathers, which attribute syntax hid.
- Added `EquilibriumTrailsState`, which names the triple that `equilibrium_iterative`
  and `trails_equilibrium` return and `equilibrium_state` consumes. The triple
  crossed those boundaries unnamed, which cost a repeated inline annotation at every
  one of them and left the callers unpacking it positionally. A
  sequence state holds one stage across all the trails, so no trail is whole in it;
  stacking every stage is what completes them, which is what this holds.
- Added `Structure.is_edge_deviation_direct`, a property that masks the
  deviation edges whose two nodes share a sequence. It replaces a stored field:
  which edges those are follows from the sequences, so deriving it means it cannot
  fall out of step with a trail that shifts.
- Added `nodes_deviation_force`, which accumulates the deviation force at every node, and
  `nodes_residual`, which assembles the residual at every node from the edge forces
  and the loads. The first names the quantity it returns and not the edges it reads,
  since a deviation is a kind of edge rather than something a node carries; a residual
  is already the quantity, so its name needs no such suffix.
- Added an edge range check to `Structure.__check_init__`. `scipy` used to
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
- Added `jax_cem.datastructures.sequences`, holding `Sequence`,
  `sequences_from_trails`, and `is_edge_deviation_direct`, which
  `jax_cem.datastructures.trails` used to carry. The trail search orders the nodes;
  laying that order out into the sequences the equilibrium scan steps through is a
  separate concern, and only the second one depends on the shifts.
- Added `jax_cem.datastructures.trails`, a native trail search that replaces the
  reliance on `TopologyDiagram.build_trails`. `search_trails` orders the nodes of a
  structure into trails; `build_trails` derives everything that the layout of those
  trails settles, which is the other half of what the method it stands in for does.
- Added `align_trails`, which starts every trail at the sequence of the node it
  deviates to and returns a new structure, mirroring `shift_trail` on a topology
  diagram. One pass makes an origin node's deviation edges direct without making
  every edge in the structure direct.
- Added `Structure.__check_init__`, which rejects self-loops, a support
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

- Changed the masking of the indirect deviation forces to build the parameters it passes
  on with `equinox.tree_at` rather than `NamedTuple._replace`, which is private, and to
  bind the result to `params_pass` rather than rebinding the `params` argument. The pass
  and the parameters it was handed are then two names, and what the scan closes over says
  which one it reads. The three tests that reached for `_replace` build theirs the same
  way; two of them replace a length and a plane together, which `tree_at` takes as a tuple
  of leaves and which spelling the container out would have restated five fields to say.
- Changed `edges_force` to decode a slot with the counts the structure states,
  `num_sequences` and `num_trails`, rather than by unpacking the shape of the array it
  then reads. The unpacking bound two counts to names that read as the things counted,
  `sequences, trails = trail_forces.shape`, and a count carries `num_`; naming it that way
  is what shows the structure already answers it. The unpacking also served as a rank
  guard against a stacked grid, which the shape annotation on the arguments now rejects
  one step earlier, though only while the import hook that checks it is installed. That
  is how the suite runs, and a stacked grid cannot arrive here from the one caller, since
  a batch is taken by `vmap` and the function sees one structure at a time.
- Changed `origin_nodes` from a field of `Trails`, forwarded by a property, into a field
  of `Structure` that `build_trails` returns beside the trails. A shift rewrites the
  layout and leaves the origins alone, so they are not part of the layout, and holding
  them there was defensive against a drift no shift could cause. `align_trails` replaces
  both fields together, which is what turns `test_a_shift_leaves_the_origin_nodes_alone`
  back into a check: discarding the recomputed origins would have made it assert only
  that the transform failed to reach a field.
- Changed the origin nodes to be read off the layout, at the first slot each column
  occupies, rather than walked out of the ragged trails one tuple at a time. A shift moves
  a trail down its column without reordering it, so that slot holds the node the trail
  starts at whatever the shift, and the read is one `argmax` over the grid.
  `sequences_from_trails` returns the grid alone, since nothing else wanted the origins it
  used to compute beside it.
- Changed the trail edge that a plane drives to be marked by the plane rather than by a
  zero length. `sequence_equilibrium` used to ask whether an edge had a length of zero,
  which read a sentinel out of the array an optimizer varies: a length walked onto zero
  hands the edge over to its plane mid-optimization, and the gradient with respect to
  that length is zero at the crossing, so nothing reports the handover. A zero normal is
  a degenerate plane rather than a value anyone means, so the question is now asked of
  the plane. The precedence this gives a plane over a length is the one the COMPAS CEM
  converter already applied when it zeroed the length of an edge it gave a plane to, so
  no structure built through it changes. An edge handed both directly through
  `Parameters` now takes its plane where it used to take its length.
- Changed the zero test on a plane normal into `is_plane_absent`, which both the branch
  that selects the plane and the guard inside `node_length_plane` now ask, so the two
  cannot answer differently and leave an edge whose plane is read by one and treated as
  degenerate by the other. The normal is compared against zero within an absolute
  tolerance, since it arrives unnormalized and an exact test takes a normal of `1e-12`
  for a real plane and puts the node on it, which moves the node by the same order.
- Changed six functions of `jax_cem.equilibrium.models` for legibility, with every
  value and gradient unchanged: `nodes_resultant` names its two scatters `at_tail` and
  `at_head` rather than subtracting one call from another across five lines;
  `trails_force` keeps the two axes of the layout rather than collapsing them with a
  `jnp.concatenate` of an array against itself, and stops rebinding its own arguments,
  so `edges_force` addresses a slot as the grid coordinate it is;
  `node_length_plane` renames `cos_nop` and `cos_nres`, neither of which was a cosine,
  to `offset_plane` and `advance_normal`, which is the ratio the length actually is;
  `nodes_equilibrium` names the two gathers it passes on; `trails_equilibrium` drops a
  nullary closure that built the initial scan state and stops shadowing `xyz` twice; and
  `equilibrium_iterative` names the two iterates `current` and `previous` and factors the
  sweep it runs three times into one local.
- Changed `ParameterState` to `Parameters`. The states an equilibrium returns are named
  for what they are, and the parameters it takes are not one of them: they are the input,
  where a state is an output, so carrying the suffix put them in the wrong family.
- Changed `EquilibriumStructure` to `Structure`, and deleted the empty `Structure` base
  it inherited from. The base declared nothing and nothing subclassed it or annotated
  against it; the contract it stood in for is `formax.AbstractStructure`, which a
  structure will inherit directly when it adopts it. Formax refers to the old name and
  is updated upstream.
- Changed `build_trails` to look an occupied slot up among the trail edges rather than
  among all of them. Only a trail edge can occupy a slot, and the map was built over the
  concatenation, so a deviation edge joining the same two nodes overwrote the trail
  edge's entry, being the later of the two. The slot grid then held an edge index the
  inverse map had no room for, and the raw `IndexError` from that overflow arrived two
  lines before the check written to diagnose it. The inverse is now sized by the trail
  edges, so the check is reached, and it names the edges that occupy no slot. A pair of
  nodes carrying more than one edge is rejected in `__check_init__`, which is where it
  belongs and what the overflow was accidentally doing: the edge index is keyed by the
  node pair, so a repeat silently collapses two edges into one.
- Changed `gather_padded` and `indices_beyond` into `jax_cem.datastructures.indexing`.
  They index an array whose indices carry a padded slot, which is neither what a
  sequence is nor what one settles, so they read as a module of their own.
- Changed the padding sentinel from an index into a value that addresses nothing, which
  removes the dummy row and every slice that served it. A padded slot is marked `-1`,
  and a negative index is valid and wraps onto the last entry, so the sweep allocated
  the node positions one row too long for the wrap to land on and sliced that row off
  again afterwards, in four places and with `[:-1]` meaning the dummy row in three of
  them and the last sequence in the fourth. `indices_beyond` now sends a padded slot one
  past the end instead, where a scatter drops it and `gather_padded` fills it with zero,
  so a padded slot reads nothing and writes nowhere. `jax_cem.equilibrium.models` no
  longer contains a negative index, a negative slice, or the padded row, and
  `nodes_padded` is gone from the annotations, including the one on
  `EquilibriumTrailsState.xyz`, whose docstring described the dummy row and told a
  consumer to drop the last entry of the trailing axes. `is_edge_deviation_direct`
  carried the same dummy row and lost it too, and the layout grid is now named
  `sequences` in the two functions that called the same axis `sequences_edges`, which
  was one shorter before the slice came out. Verified value- and gradient-preserving
  against a
  reconstruction of the previous kernel over eleven structures carrying real padding,
  including the shifted COMPAS fixture: positions, forces, lengths, residuals and
  vectors agree to 0.0 at one, two and a hundred iterations, and the gradients agree to
  0.0 everywhere but one case at one unit in the last place.
- Changed every `reshape` out of the library, of which there were three, none of them
  stating which axes it collapsed. `build_trails` builds the slot grid a row at a time
  rather than appending to a flat list and folding it afterwards; the padding mask of a
  sequence takes its trailing axis from an index rather than from a reshape; and
  `trails_force` no longer flattens the layout at all. A `-1` in a reshape also accepts
  an array with an extra axis and answers on it, so a stacked grid reaching `edges_force`
  outside `vmap` used to return a plausible array of the right length; unpacking the two
  axes rejects it instead.
- Changed `trail_nodes_of` to `read_trail_nodes`. The suffix said which argument the
  function takes, which its signature already states, where the name now says what it
  does: it reads the trails back off a layout that already holds them, rather than
  searching for them as `search_trails` does.
- Changed every construction that sat in a `return` statement to bind its object to a
  name first, in `build_trails`, `trails_equilibrium`, `read_trail_nodes`, and the
  `node_index` and `edge_index` properties. A name states what the thing is where it
  comes into being, and gives it somewhere to be inspected. This extends the rule the
  JAX array construction already followed to containers and comprehensions.
- Changed `Sequences` to `Trails`, which `Structure` now holds as `trails`,
  and nested the two grids inside it as `Trails.sequences`. The trails are the entity
  the CEM is about, the channels a force flows down, and a sequence is the step a
  computation takes across them, so the containment runs that way; the plural also
  advertised a stack that it was not, and left two containers a single character apart.
- Changed the sequences to stacked arrays rather than one container per sequence, which
  is what lets the scan stay a scan. A scan slices a leading axis and a collection has
  none, so the sweep would unroll: measured on trail grids, a collection reaches 1400
  jaxpr equations and 349 ms of compile at 5 sequences and 15370 and 4376 ms at 60,
  where the scan holds 124 equations and 184 ms at any size. The answers are identical
  and the run time is a wash, so what a collection costs is the graph and the compile.
- Changed `Sequences.edges` and `Sequences.edges_slot` to `Sequence.edges` and
  `Trails.edges_sequence`, the two directions of the bijection between slots and trail
  edges: keyed by a slot the first holds the edge, and keyed by the edge the second holds
  the slot. Both hold indices rather than endpoints, unlike `Structure.edges`, which holds
  node key pairs; the container is what says which is meant, and the stacked forms a
  caller reads say it outright, `sequence_edges` against `edges_sequence`.

  This passed through `trail_edge_index` for both, one name carrying the bijection in
  either direction on the grounds that the two cannot be confused silently: they differ
  in rank once the sequences are stacked and can never hold the same number of entries,
  so substituting one for the other raises. That holds, and it was still the wrong name.
  What a shared name costs is not a wrong answer but a docstring paragraph that has to
  be read before either field can be used, which is a convention living in prose rather
  than in what a caller types.
- Changed `Structure.__init__` to convert once, on the way out, rather than
  coercing every argument on the way in and rebinding it. The four `np.asarray` calls
  were no-ops on the NumPy arrays the signature declares, since `np.asarray` returns an
  ndarray unchanged, so all they did was widen the declared type in silence and shadow
  the arguments they converted. The host work now reads what the caller passed and the
  conversion to JAX is one block. A list of pairs reaches the pair check instead of
  being coerced, so it raises there rather than being accepted against the annotation.
- Changed `Trails` and `build_trails` to live in `jax_cem.datastructures.trails`, beside
  the search that finds the trails and the shift that aligns them, where they had been
  defined in the module that holds a sequence. The two modules split on the entity each
  is about, which also turns their dependency one way: the trails read a sequence, and a
  sequence reads nothing of theirs.
- Changed `build_trails` to `search_trails` and `build_sequences` to `build_trails`. The
  first finds the trails and returns their nodes, the second lays those out and returns
  the container a caller asks for. They stay two functions because only the layout
  depends on the shifts and only the layout runs again on alignment. The ragged form is
  `trail_nodes` wherever it appears, including `trails_of`, which became
  `read_trail_nodes`, so a bare `trails` always means the container.
- Changed `Trails.origin_nodes` from a property to a field. The property
  searched the layout for the first node of every column with `jnp.argmax` and a gather,
  which is index data computed with `jnp` against the rule for it, and both of its
  callers pull the result straight back to the host: 156 us per read against 0.3 us for
  a stored field. `sequences_from_trails` already returns the same fact, read off the
  trails, and `build_trails` discarded it. Storing it removes the search rather than
  duplicating it, and it cannot fall out of step with a shift, which moves a trail down
  the sequences without reordering it.
- Changed `sequences_equilibrium` to `trails_equilibrium`, after the state it returns.
  The CEM computes equilibrium along the trails of a structure, one sequence at a time,
  and the old name said the second half of that: it read as the plural of the step
  beneath it, `sequence_equilibrium`, where the two differ in what they range over and
  not in how many of it they take.
- Changed `EquilibriumModel.sequence_equilibrium` to five arguments from eight. Three
  pairs of them were one thing each: the node row and the edge row of a sequence are a
  `Sequence`, the position and the trail residual it carries are the sequence state it
  already returns, and the masked deviation forces are the forces of the pass, which
  the pass now settles into the parameters it hands down rather than threading beside
  them. Taking and returning one state is also what a scan carries, so the step reads
  as the transition it is.
- Changed `nodes_equilibrium` to five arguments from six, reading the forces of the
  pass off the parameters for the same reason.
- Changed `EquilibriumModel.equilibrium_iterative` to read the iteration settings off
  the model, from six arguments to three. It was handed `tmax`, `eta`, and `scale` by
  the one caller that holds them, so the call site named each of them twice. Holding
  the settings is what the class is for, which leaves every member of it either reading
  one or driving the control flow they configure.
- Changed `EquilibriumModel` to hold the sweep alone, moving the nine quantities it
  computed to functions of `jax_cem.equilibrium.models`: `nodes_equilibrium`,
  `nodes_deviation_force`, `nodes_position`, `nodes_length_plane`, `node_length_plane`,
  `edges_length`, `nodes_residual`, `edges_force`, and `trails_force`. The class held
  seventeen members and one of them read a setting, `__call__`; every other mention of
  `self` was a lookup of a sibling, which is a namespace rather than an object. Six
  members remain. A test that reached a quantity by constructing a model it had no use
  for now calls the function.
- Changed `Parameters.lengths` and `Parameters.planes` to the trail edge each
  one drives, shaped `"edges_trail"` and `"edges_trail 6"`, rather than the node that
  edge leaves. COMPAS CEM carries both on the trail edge and both of its kernels read
  them by edge key, so the nodewise keying was an artifact of the converter that
  filled them rather than a decision. It cost a row per support, where no trail edge
  leaves and nothing is read, and every caller had to zero those rows to stop
  `length == 0.0` from meaning no outgoing edge as well as plane-driven; the trailing
  `1` axis of `lengths`, which existed to be raveled off after the gather, goes with
  them. Exact on the six fixtures: node positions, edge forces, edge lengths, and
  residuals agree to 0.0, as do the gradients with respect to lengths, planes,
  positions, and forces, under `jit`, `vmap`, and `jacobian`.
- Changed `sequence_equilibrium` to take the trail edges of a sequence beside its
  nodes, which the scan steps over together. A slot that no trail edge leaves reads
  the last edge, which the padding mask and the slot the forces are gathered from then
  drop, in the same way a padded node key already reads the last node.
- Changed `Parameters.xyz` to `xyz_origin`, the position of the origin node of
  every trail, shaped `"trails 3"`. The sweep computes the position of every node it
  steps onto from a node array seeded with zeros, so the origins were the only
  positions the parameters were read for, and the remaining rows were named as inputs
  without being any. The axis is `trails` rather than a node subset because the array
  is the initial value of the per-trail position the scan carries, which makes a
  mismatched count a shape error rather than a wrong answer.
- Changed `node_length_plane` to take the plane that drives a trail edge, and
  `nodes_length_plane` to take one plane per trail, in place of the parameter state and
  an index into it. It is the one geometric step of the sweep, and taking a plane rather
  than a state it indexes leaves the keying with the caller that owns it: the same
  arithmetic no longer has to be reached through whichever entity the parameters happen
  to be keyed by. Timed on trail grids under `jax.jit`, the two forms are level at 16,
  400, and 3600 nodes, so this buys clarity and not speed.
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
- Changed `trails_equilibrium` to resolve `use_indirect` once, before the scan, into
  the force array it selects. The flag reached three signatures and was consulted at
  every node of every sequence, though it settles the same question every time.
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
- Changed `sequences_edges` and `sequences_edges_indices` to one
  `Trails.edges_sequence`, the slot of the layout that each trail edge occupies. The pair
  mapped a slot to an edge and then selected the occupied slots, so reading a per-slot
  quantity back in edge order took a gather and a scatter; inverting the map once, in
  the constructor, turns it into a single gather. The old name was wrong
  rather than loose: on the shifted fixture `sequences_edges_indices` held the value
  10 for a structure with seven trail edges, because its values indexed the flattened
  grid, not the edges.
- Changed `Structure.origin_nodes` to a property reading the layout, which
  states it, rather than a field of the structure beside it.
- Changed `deviation_vector` to `nodes_resultant`, which takes the edge set it
  accumulates over. Nothing in it was specific to a deviation edge, and the nodal
  residual needs the same accumulation over every edge.
- Changed `residual_trail_vector` to `residual_trail_next`, and settled the rule the
  two renames follow: a helper carries an entity prefix when it is bound to that axis,
  by scattering into it or gathering along it, and carries none when its arithmetic
  broadcasts. The `_vector` suffix claimed a single vector where `nodes_resultant`
  returns one per node, so it stated cardinality, which the shape annotation already
  states, and stated it wrongly.
- Changed `Parameters.forces` to cover the deviation block alone, shaped
  `"edges_deviation"`. The trail entries were overwritten on every call and never read.
- Changed `vector_length` to return a scalar rather than a one-element vector, which
  is what flattens the per-edge quantities of the state.
- Changed `build_trails` to take the trails, the edges and the shifts alone. The
  node and trail edge counts were only needed by the mask that is now derived.
- Changed every array annotation in the package to a jaxtyping shape, replacing the
  `# N x 3` comments and the bare `jax.Array` declarations. The shapes were verified
  by running the suite under jaxtyping's import hook with a runtime typechecker,
  which turns each annotation into a check.
- Changed the fields of `Structure` from NumPy to JAX arrays. The trail
  search still computes with NumPy and `build_trails` converts once on the way out,
  so the shipped structure is device-resident.
- Changed `Structure` to require an edge array of node key pairs. It used
  to reshape whatever it was given, which let a flat array through and made the
  declared shape unenforceable. That reshape also flattened a batch of structures
  into one, which then failed inside the trail search with an error about the node
  keys; a batched input is now rejected where it enters.
- Changed the counts on `Structure` to read the trailing axes rather than
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
  `Structure` attributes, the `vmap`-ed `index` parameters of
  `node_equilibrium` and `node_length_plane` were declared `int` where they receive a
  traced scalar, and `node_index`, `edge_index`, `sequences_edges` and
  `sequences_edges_indices` were declared `jax.Array` where they hold dicts and
  NumPy arrays.
- Changed the test suite to `pytest-lazy-fixtures`, replacing the abandoned
  `pytest-lazy-fixture` and its `pytest<8` ceiling.
- Changed `Structure` to take four arrays: `nodes`, `supports`,
  `edges_trail`, and `edges_deviation`, down from eight arguments. Trail and deviation
  edges are held apart rather than as one edge array and a mask, `edges` concatenates
  them with trail edges first, and everything else is derived. `align_trails` applies
  a shift with `equinox.tree_at`, so no shift reaches the constructor.
- Changed `support_nodes` to `supports`, matching the Formax contract.
- Changed `indirect_edges` to `edges_deviation_direct`, which is what the mask holds.
- Removed `incidence` in favour of negating `connectivity` at its one call site, since
  the two were the same array up to sign.

### Removed

- Removed `EquilibriumModel.node_position` and `EquilibriumModel.equilibrium`, which
  forwarded rather than computed: the first called `position_vector` with the arguments
  it was given, and the second called the sweep with `use_indirect=False`. A wrapper
  that lifts a scalar function onto an entity axis was kept, since the `vmap` is the
  work; one that only renames its callee was not.
- Removed `EquilibriumModel.verbose`, which was assigned in the constructor and read
  nowhere.
- Removed `Structure.connectivity` and `connectivity_matrix`, which the
  gather and scatter kernel leaves with no reader. `connectivity_matrix` was public
  through the star import of `jax_cem.datastructures`, which now carries an `__all__`.
- Removed `EquilibriumModel.node_equilibrium`. With the accumulation done for every
  node at once, the node index only selects a row, and keeping the function would have
  forced the vectorization to stay.
- Removed `SequenceData.edges_deviation_direct` and the structure field it fed, in
  favour of the derived `is_edge_deviation_direct`. The stored mask spanned every edge
  though its whole trail half was zero by construction.
- Removed the dependency on `scipy`, which only `connectivity_matrix` imported.
- Removed `Structure.trail_edges` and `Structure.deviation_edges`.
  Nothing read the first, and their names differed from the `edges_trail` and
  `edges_deviation` fields only in word order. The second was a mask over a
  contiguous block, so `node_equilibrium` now slices the deviation edges out of the
  connectivity instead of multiplying every edge by a zero, and `nodes_equilibrium`
  builds edge vectors for that block alone rather than for every edge.
- Removed the `keepdims` flag of `vector_length`. Nothing passed it, and it changed
  the return shape, which left the function without one shape to declare.
- Removed `trail_length` from `jax_cem.equilibrium.models`. It was unreachable and
  its comment described a shape it did not return.
- Removed `Structure.from_topology_diagram` and
  `Parameters.from_topology_diagram`. They made `compas_cem` a hard import of
  `jax_cem.parameters`; the conversion now lives in `tests/converters.py` until the
  native array constructor replaces it.
- Removed the dependency on `compas`. The library no longer imports
  `compas.numerical` or `compas.utilities`, which lifts the `compas==1.17.10` pin.
- Removed the Sphinx documentation sources, `tasks.py`, and the `temp`, `scripts`,
  and `data` placeholder directories.
- Removed the `HOME`, `DATA`, `DOCS`, and `TEMP` globals from `jax_cem`.
