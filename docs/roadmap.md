# Roadmap

The refresh of `jax_cem`: remove COMPAS from the library, adopt the packaging and
tooling of `jax_fdm`, `smax`, and `formax`, and expose an array-based structure
that satisfies the Formax equilibrium contract.

Work happens on the `refresh` branch, cut from `refactor`. That base carries two
unmerged fixes worth naming, because both are behavioral rather than cosmetic:
`7089b11` removes the `nan` raised when a residual is perpendicular to a plane
normal, and `f81156c` vectorizes the edge length and force computation for the
same reason.


## Design decisions

### The structure is built from arrays

`Structure` is constructed from arrays, not from a declarative layer of
`Node` and `Edge` objects.

```python
class Structure(AbstractStructure):
    nodes:           Int[Array, "nodes"]
    supports:        Int[Array, "nodes_fixed"]
    edges_trail:     Int[Array, "edges_trail 2"]
    edges_deviation: Int[Array, "edges_deviation 2"]

    @property
    def edges(self) -> Int[Array, "edges 2"]:
        return jnp.concatenate((self.edges_trail, self.edges_deviation))
```

Everything else — origin nodes, sequences, indirect deviation edges, the
connectivity and incidence matrices, and the sequence-to-edge index maps — is
derived in the constructor by the native trail search.

Trail and deviation edges are held as two disjoint arrays rather than one edge
array plus a boolean mask. An edge is a trail edge or a deviation edge and never
both, and two arrays make that structurally true instead of a rule a validator
has to enforce. `edges` is derived from them, so the three cannot drift apart.

The concatenation is trail-major, and that order is canonical: `trail_edge_index`
indexes it, and any source with its own edge order — JSON, a mesh, a COMPAS CEM
diagram — is permuted into it at the boundary.

Two consequences reach past the constructor. Trail and deviation edges occupy
contiguous blocks, so the masks the kernel used to multiply through are gone:
`deviation_edges` because the deviation edges are their own array, and
`edges_deviation_direct` because it is derived from the sequences on demand.

The edges are **not** reordered into `[trail | deviation_direct | deviation_indirect]`,
which an earlier draft of this roadmap proposed so that the first equilibrium pass
became a slice. Whether a deviation edge is direct is a property of the current
sequence layout, not of the edge: `align_trails` shifts trails and changes which edges
qualify, which `test_align_trails_can_trade_one_indirect_edge_for_another` pins. A
three-block order would therefore have to permute the edge array on alignment, and any
per-edge parameter built against the earlier order would then address the wrong edges,
silently. The order stays fixed and the distinction stays a mask.

Trail-edge forces are outputs, recovered
from the residuals, so `Parameters.forces` covers only the deviation block;
the trail entries the old parameter array carried were overwritten on every call
and never read.

A declarative layer remains possible later as a constructor-side module that emits
these arrays. It is deferred rather than rejected: it adds nothing the kernel or
the Formax adapter needs, and building it first would mean merging topology and
parameters in one object only to separate them again one function later.

### Trail edge lengths and planes are edgewise

`Parameters.lengths` and `Parameters.planes` are keyed by the trail edge each
one drives, shaped `"edges_trail"` and `"edges_trail 6"`, not by the node that edge
leaves.

The nodewise keying was a translation artifact rather than a decision. COMPAS CEM
carries both on the trail edge, and both of its kernels read them by edge key after
resolving the node from its trail and its sequence; nothing upstream keys a length by
the node whose outgoing edge it governs. The re-keying happened once, in the converter,
and the kernel was written against its output.

Nodewise the arrays are oversized and the sentinel carries a second meaning. Every
trail ends at one support and every other node leaves exactly one trail edge, so a
structure has `nodes - trails` trail edges and the support rows are never read: a
length set at a support does nothing, and every caller has to zero those rows to keep
`length == 0.0` from meaning three things at once — plane-driven, no outgoing edge, or
genuinely zero. Edgewise the second meaning is gone, and the trailing `1` axis with
it, which leaves the parameters split by entity: `xyz` and `loads` per node, `forces`,
`lengths`, and `planes` per edge.

This lands before the sentinel that phase 2 deferred, not after. The mask that
replaces the reserved value is one flag per trail edge, so it has a shape only once
the parameters it masks have one.

A sequence gains `trail_edge_index`, the trail edge outgoing from the node at each of its
slots and `-1` where a slot has none. It is the slot-to-edge map `build_trails` already
computes in order to invert it into `trail_edge_index`, kept rather than discarded, so
neither can fall out of step with a trail that shifts, and the scan steps over the
nodes and the edges of a sequence together.

Two alternatives were rejected. Parameters shaped to the layout need no map at all,
but a shift rewrites the layout and would silently re-address them, which is the
failure the fixed edge order exists to avoid. A node-to-edge map on the structure
leaves the scan signature untouched and is invariant under a shift, but it reaches a
length through two gathers rather than one, and it states a fact about the trails in a
second place.

Which edge a node's length governs follows from the trail direction and not from the
stored orientation of that edge: a structure may hold a trail edge as `(v, u)`, and on
the braced tower fixture all four are held that way. The orientation never reaches the
equilibrium, because the length is applied along the trail residual, but whatever
fills the arrays has to walk the trails rather than read the edge array.

The change is exact. Measured against the nodewise kernel on the six fixtures: node
positions, edge forces, edge lengths, and residuals agree to 0.0, as do the gradients
with respect to lengths, planes, positions, and forces, under `jit`, `vmap`, and
`jacobian`. A padded slot gathered at `-1` and wrapped to the last row, as a padded
node key did, and what it computed there reached only the layout slots that the
`trail_edge_index` gather drops. That wrap is gone; see "Padding addresses nothing".

### Only the origin positions are parameters

`Parameters.xyz_origin` holds the position of the origin node of every trail,
shaped `"trails 3"`, where the field was `xyz` over every node.

A sweep computes the position of every node it steps onto, and the node array it steps
through is seeded with zeros, so the origins were the only positions the parameters
were ever read for. The rest of the rows were named as inputs and were not, which is
the same oversizing the lengths and the planes carried, in the other entity.

The axis is `trails` and not a node subset, because the array is the initial value of
the per-trail position the scan carries and shares that axis with it, which is what
lets a mismatched count fail as a shape rather than as an answer. The order is the
order of the trails, which `Structure.origin_nodes` states; a shift moves a
trail down the sequences without reordering the trails, so alignment leaves it alone.
`jax_fdm` restricts the same field the same way and names it for the subset it holds,
`xyz_fixed`.

### The model holds the sweep, not the quantities

`EquilibriumModel` holds the settings of the solver and the control flow they drive:
the sweep over the sequences, the fixed point that resolves the indirect deviation
edges, and the state a call returns. Every quantity a step of that sweep computes is a
function of `jax_cem.equilibrium.models`.

What `self` carried is what exposed the split. Of the seventeen members the class held,
exactly one read a setting, `__call__`. Every other mention of `self` was a lookup of a
sibling method, which makes the class a namespace rather than an object.

The line is drawn on the entity and not on whether a member happens to read `self`,
which is an accident that drifts as soon as someone adds a call. A quantity is a
function and the sweep that orders them is the class. The functions that move out are
the entity-level forms of the primitives that already sat at the bottom of the file —
`trails_force` over `trail_force`, `nodes_deviation` and `nodes_residual` over
`nodes_resultant` — so the module gains one layer rather than a list.

What stays does not stay because it needs the settings, since `equilibrium_state` and
`sequence_equilibrium` read none. It stays because it is the seam a variant model
overrides, which is how the edgewise keying was prototyped before it landed. A variant
quantity is a different function, and composing functions needs no seam. A member that
only forwarded to a helper went rather than moved.

### The model is a plain class

`EquilibriumModel` is a plain Python class and not an `eqx.Module`, though the state
and the structure are modules and the Formax model protocol is one.

Nothing the model holds is differentiable. Its fields are solver settings, and `tmax`
gates a Python branch as well as the step count of the iteration; as a leaf of a module
it would be traced, and the branch would fail the moment a model reached a transform as
an argument rather than as a closure. Declaring the field static is the fix, and
nothing needs it yet. `eta` and `scale` reach only traced arithmetic, so they could be
leaves the day a batch over tolerances is wanted.

The module identity belongs to the wrapper rather than to the kernel. Formax makes a
model a module so that a model can be a field of another model, which is what its mixed
model is, and so that its abstract protocol is nominal rather than structural. Its own
audit of `jax_fdm` records the plain class there as the configuration and parameter
split it wants, and this follows it.

### `edges` is a property

Verified against equinox 0.13.8: a property satisfies `eqx.AbstractVar`. The class
instantiates, `isinstance` against the abstract contract holds, and omitting the
implementation still raises at instantiation. The pytree carries only the stored
fields, so no derived copy of `edges` can go stale under `tree_map`, `tree_at`,
or `vmap`.

### Index arrays are JAX arrays

Structure fields are annotated and stored as `Int[Array, ...]`. Index data is
computed with NumPy in the trail-search helpers and converted once, at the
constructor. This follows `smax`, whose shipped containers are all-JAX and whose
NumPy annotations appear only on construction helpers, and it matches the Formax
contract. `jax_fdm` keeps key arrays as NumPy on the container; `jax_cem` does not
follow it here.

The rule cuts both ways, and a derived index that reads as a property is where it was
being broken. `origin_nodes` searched the layout for the first node of every column
with `jnp.argmax` and a gather on every read, while the trail search already had the
same fact on the host and threw it away. It is stored now: 156 us per read became
0.3 us, and both of its callers pull it back to the host anyway. What may stay derived
is a view over stored fields, like `edges`, or a mask nothing can author, like
`is_edge_deviation_direct`. What a NumPy pass already computed is stored.

### The trails hold their sequences

A structure holds `trails`, and the trails hold their `sequences`. The trails are the
entity the CEM is about — the channels a force flows down — and a sequence is the step
the computation takes across them, so the containment runs that way and not the other.

```python
class Trails(NamedTuple):
    sequences:    Sequence               # every sequence, stacked
    trail_edge_index: Int[Array, "edges_trail"]
    origin_nodes: Int[Array, "trails"]
```

The sequences are **stacked arrays and not a collection of `Sequence`**. A scan slices
a leading axis; a tuple of one container per sequence has nothing to slice, so the
sweep would unroll. Measured on trail grids, against the 124 equations and 184 ms the
scan compiles to at any size: 1 400 equations and 349 ms at 5 sequences, 5 210 and
1 296 ms at 20, and 15 370 and 4 376 ms at 60. The answers are identical and the run
time is a wash, since XLA flattens it either way, so what a collection costs is the
graph and the compile.

`Sequence` therefore does honest double duty. Its shapes describe one sequence, which
is the view the scan presents to the step it drives, and `Trails.sequences` is that
container with a leading sequence axis. It is the contract a stacked structure under
`vmap` already carries.

One name carries the bijection between slots and trail edges, in both directions.
`Sequence.trail_edge_index` is keyed by a slot and holds the trail edge leaving it;
`Trails.trail_edge_index` is keyed by the trail edge and holds the slot it occupies.
The shared word is the entity the pair is about, and the container says which way it is
being read. Both hold indices rather than endpoints, which an `edges` would not have
said: `Structure.edges` holds node key pairs, so that word in two containers
would have meant two things.

Sharing the name puts the difference in the docstrings, which each state their key,
their shape, and the direction. What keeps that from being a hazard is that the two
cannot be confused silently: they differ in rank once the sequences are stacked, a grid
against a flat array, and they can never hold the same number of entries, since a trail
of `n` nodes fills `n` slots and owns `n - 1` edges. Substituting one for the other
raises rather than answering.

Two other pairs were tried. `trail_edge_of_slot` with `slot_of_trail_edge` spells both
halves and cannot be misread, at the cost of length. `trail_edge_index` with
`slot_index` names what each holds and leaves the key to the shape, but reads as the
mapping that `node_index` and `edge_index` already are, in the opposite direction. The
shared name reads as the duality it is, which is what a caller holding one of them is
usually thinking about.

The trail search and the layout stay separate functions, since only the second depends
on the shifts and only the second runs again on alignment. `search_trails` finds the
trails and returns their nodes; `build_trails` lays those out and returns the container.
The ragged form is `trail_nodes` wherever it appears, including `read_trail_nodes`, so a
bare **trails** always means the container.

### Padding addresses nothing

A trail that is shifted, or shorter than the tallest one, leaves slots of the sequence
grid empty, and those slots are marked `-1`. The mark is a value in the layout, and it
is not an index: `indices_beyond` sends it one past the end of whatever it would have
addressed, where a scatter drops it and `gather_padded` fills it with zero.

The alternative, which the kernel carried until it did not, is to let `-1` index. It is
a valid index that wraps onto the last entry, so the node positions were allocated one
row longer than the structure has nodes, the wrap landed there, and the row was sliced
off again on the way out. That cost a slice everywhere the positions were read, put the
same `[:-1]` token in one function with two meanings, and left the arrays that carry no
dummy row — the lengths, the planes, the loads — silently reading their last real entry
for a padded slot, correct only because a mask further down dropped it. The mask was
derived from the node of a slot while the entry it protected was keyed by the edge, and
those two disagree at the last node of a trail.

What replaces it is narrower than a mask: a padded slot contributes zero because it
reads zero, so the arithmetic it feeds carries it through rather than having to be told
about it. The sentinel stays `-1` in the layout, where it reads as absence and tests as
`< 0`, and stops being an index at the one place it was.

### The iteration is a checkpointed while loop

`equilibrium_iterative` runs on `equinox.internal.while_loop(kind="checkpointed")`,
which is what `jax_fdm` iterates on as well. It is the only import the library takes
from a private module of a dependency, and it stays because nothing public covers what
it does: a bounded loop that exits as soon as the nodes stop moving and still
differentiates in reverse, with memory logarithmic in the step count.

Replacing it with `scan` is exact — the suite passes and the values and gradients are
unchanged — and costs the early exit. Structures converge in one to sixteen iterations
where `tmax` defaults to a hundred, so a fixed trip count pays several times over: on a
four hundred node grid, five times the run time, seven times the gradient time, and
forty times the peak memory of the gradient, growing linearly in `tmax` where the
checkpointed loop grows logarithmically. Skipping the map with `lax.cond` recovers the
run time and neither the memory nor anything under `vmap`, where the branch lowers to a
select and both sides run.

### Batching goes through `vmap`

A structure is built one at a time and batched by stacking the modules into a
pytree, which `vmap` then maps over. The constructor runs the trail search on the
host, in Python, over dictionaries, so it cannot be traced and cannot be `vmap`-ed;
a batched array reaching it is rejected rather than reshaped. Stacking does not run
the constructor again, because equinox rebuilds a module from its leaves.

The shape annotations describe one structure, which is the view `vmap` presents to
the kernel. They do not hold on a stacked structure read outside `vmap`, and that is
the JAX contract rather than a gap. What is a gap is a stacked structure that
answers wrongly instead of failing, so the counts read the trailing axes and `edges`
concatenates on the edge axis. Before that, `number_of_nodes` on a stacked structure
returned the size of the batch.

### License

Apache-2.0, following `formax` and `smax`. The patent grant is the stronger choice
for a library `formax` is built on. `jax_fdm` and `compas_cem` remain MIT.

### COMPAS

No COMPAS or COMPAS CEM code inside the `jax_cem` package. `compas_cem` stays as an
optional dependency used by the examples and by cross-check tests marked
`compas_xcheck`, on the pattern `jax_fdm` uses for deletable scaffolding.

A shift changes which deviation edges are indirect, not the equilibrium the model
finds. Measured on the shifted fixture: the aligned and unaligned structures agree to
1e-8 at `tmax=10` and differ by 6.0 in a single pass. Alignment buys iterations, not
a different answer, which is why the baselines pass either way.

`compas.numerical.connectivity_matrix` gains a local NumPy implementation. This is
forced rather than chosen: COMPAS 2.x removed `compas.numerical` entirely, which is
the only reason the package was pinned to `compas==1.17.10`. `compas_cem` supports
COMPAS 2.x and numpy 2 on `main`, but its published 0.8.6 does not, so the dependency
is pinned to git until the next release.

The native trail search is the substantive piece of work in the refresh. `jax_cem`
did not own trail sequencing; it read `trails()`, `node_sequence()`, and
`is_indirect_deviation_edge()` off a COMPAS CEM topology diagram. Trail discovery,
sequence assignment, shifted-trail padding, and the direct versus indirect deviation
edge classification now live here. Auxiliary trails are deferred.


## Phases

Every phase lands green. That constraint sets the order: phase 1 does the packaging
and the smallest decoupling that lets the existing suite run in a modern
environment, and phase 2 replaces the core with the suite already passing. Phase 3
follows 2, since there is nothing to annotate until the containers settle. Phase 4
gates 5.

The point of a green phase 1 is not tidiness. It produces a run of the current
algorithm against the frozen baselines, on modern dependencies, before phase 2
rewrites trail sequencing. That run is the only reference the rewrite can be
diffed against.

Phases 1 through 4 are closed. Phase 2 kept the plane sentinel and auxiliary trails,
which are the same deferral rather than two: both are authoring decisions with nowhere
to live until the constructor settles.

### Phase 1 — Packaging and CI

Remove every COMPAS trace outside the source tree.

| Removed | Reason |
| --- | --- |
| `setup.py`, `setup.cfg`, `MANIFEST.in`, `.bumpversion.cfg` | superseded by `pyproject.toml` |
| `requirements.txt`, `requirements-dev.txt` | `compas==1.17.10`, `sphinx_compas_theme`, `compas_invocations`, and the pre-ruff lint stack |
| `tasks.py` | entirely `compas_invocations` |
| `docs/` Sphinx sources | built on the COMPAS theme |
| `.github/workflows/{build,docs,release}.yml` | `compas-dev/compas-actions.*` |
| `temp/`, `scripts/`, `data/` placeholders, `.editorconfig`, `src/jax_cem/__main__.py` | COMPAS cookiecutter residue |
| the COMPAS parts of `conftest.py` | the doctest namespace fixtures and the Rhino/Blender/GHPython collection guards |
| the `from_topology_diagram` constructors | they are the library's last COMPAS dependency |

The library is COMPAS-free at the end of this phase. `structures.py` imported
`compas.numerical.connectivity_matrix`, which COMPAS 2.x removed, so it gained a
local implementation and `itertools.pairwise` replaced `compas.utilities.pairwise`.
The two diagram constructors could not stay: they made `compas_cem` a hard import of
`jax_cem.parameters`, which left `jax_cem.equilibrium` unimportable without it.

The topology fixtures stay, because that is how they were authored, and the
converters they need now live in `tests/converters.py` rather than in the library.
Phase 6 replaces them with the native array constructor and deletes the module.

`compas_cem` is a dev dependency pinned to git. Its COMPAS 2.x and numpy 2 support
is on `main` but unreleased; the published 0.8.6 still requires `compas==1.17.10`,
which is the stack this phase leaves behind. It reverts to a version pin once 0.8.7
ships. Two COMPAS 2.x renames followed from the move: `connected_edges` became
`node_connected_edges`, and `edge_length` takes the edge rather than its two nodes.

`pytest-lazy-fixture` is replaced by the maintained `pytest-lazy-fixtures`, which
lifts the `pytest<8` ceiling the abandoned plugin imposed.

The package moves from `src/jax_cem/` to a top-level `jax_cem/`. Added:
`pyproject.toml` with `uv.lock` and `.python-version`, `requires-python`
`>=3.11,<3.14`, dependencies on `jax`, `numpy`, `equinox`, and `jaxtyping`, and
`[dependency-groups]` for `dev` and `typecheck`; ruff at line length 88 with
`select = ["COM812", "D202", "D300", "E", "F", "I", "W"]`, `force-single-line`
imports, the `F722` and `F821` jaxtyping exemptions, and the relaxed `E712`/`E501`
rules for tests; a `.pre-commit-config.yaml` whose ruff revision is pinned to the
same version as the dev group.

CI becomes `test.yml` and `publish.yml` on the `smax` model — a uv-based lint and
format job, a matrix on the floor and ceiling of the supported Python range, and
`pytest-split` shards — plus the XLA kernel cache from `jax_fdm`, whose reasoning
applies here for the same reason, and its `pyright` job. The changelog check in
`pr-checks.yml` stays; it is not COMPAS.

The README is rewritten. It currently documents the Grasshopper componentizer and
`compas_rhino.install`, neither of which this repository contains.

### Phase 2 — Native structure and trail search

The array structure above, the native trail search that populates it, and the
change from mask multiplication to block slicing in the kernel.

The trail search runs in `__init__`. It cannot run in `__check_init__`, which is
validation only: an equinox module is frozen by then, and assigning a field there
fails with `Field ... was not initialized`.

Two parts of `TopologyDiagram.build_trails` are authoring decisions rather than
derivations, so the search rejects a topology that needs them. `align_trails` applies
one of them: it starts every trail at the sequence of the node it deviates to and
returns a new structure, which is how a transform of an immutable module reads.

Auxiliary trails are deferred. They append a node, a support, and a trail edge per
node that deviation edges alone connect, so they change the parameter arrays as well
as the topology, and a structure that needs them cannot be built in the first place.
Until then, the caller supplies the augmented topology, and a structure whose nodes do
not all reach a support is rejected with a message naming them.

Whether a deviation edge is direct or indirect is settled by the sequences, so it is
derived and never supplied, as it is in COMPAS CEM. It is not stored either: the field
became the `is_edge_deviation_direct` property, which cannot fall out of step with a
trail that shifts. The edge order it would otherwise have keyed stays fixed, for the
reason recorded above.

The graph operations are gathers and scatters, not matrix products. Edge vectors are
`xyz[edges[:, 1]] - xyz[edges[:, 0]]`, and the deviation force at a node is a
`segment_sum` over the deviation edges, which removes `connectivity` and `incidence`
entirely. This diverges from `jax_fdm`, which keeps a dense `connectivity` and a
`BCOO` variant because the force density method assembles and solves a matrix. The CEM
steps through sequences and accumulates at nodes, so the matrix buys nothing and costs
`edges * nodes`.

The conversion landed behind a benchmark, since the gain is asymptotic and the
baselines are small. Jitted on synthetic trail grids: 3.9x at 100 nodes, 36x at 900,
and 88x at 3600, where the dense form spent 966 ms against 11 ms. The edge vector half
is exact, because the matrix product added a zero for every node an edge does not
touch. The accumulation half is exact on the six fixtures but reduces in a different
order than a dot product, so in general it agrees to rounding, which double precision
keeps far below the tolerance the baselines use.

`EquilibriumState` adopts the Formax field set: `residuals` rather than `reactions`
(a reaction is the negation of a residual, and storing one of the two removes the
chance of disagreement), an added `vectors`, and `forces` and `lengths` shaped
`"edges"` rather than `"edges 1"`.

Adopting the name is not adopting the quantity, and phase 4 found the field still
holding the old one. CEM propagates a force down a trail — what the load and the
deviation forces at a node leave for its outgoing trail edge to carry — and both the
CEM literature and COMPAS CEM call that a residual. Formax and `jax_fdm` mean the
node's out-of-balance force, which is zero at a free node in equilibrium. The two
coincide only at a support, and even there they differ in sign.

So `residuals` is now assembled from the edge forces and the loads, over the whole
edge set, and the trail quantity is `residuals_trail`. The trail quantity is not
stored on the state: the force in a trail edge and the vector of that edge already
hold it. Assembling rather than scattering is what lets a free node report that the
sweep left it out of equilibrium, which is the measure the iteration is driving down.

The sentinel that encodes a plane-driven trail edge as `length == 0.0` is deferred
rather than fixed. Whether a trail edge is driven by a length or by a plane is a fact
about the model and not a reserved value in the parameters, so the mask belongs on the
structure — but nothing derives it, which means a fifth constructor argument. That is
the same question as auxiliary trails and the shifts before them: an authoring decision
with nowhere to live. It waits for the constructor to settle.

The six topology fixtures are **not** ported to the array constructor here, which is a
change from an earlier draft of this phase. They stay COMPAS CEM diagrams,
`compas_cem` stays a dev dependency, and `tests/converters.py` stays with it, because
that is what lets the rewritten kernel be measured against the one it replaces. The
port moves to phase 6.

### Phase 3 — Typing

jaxtyping shapes replace the `# N x 3` comments throughout. Every array a function
takes or returns now states its shape, and the dimension names are shared across the
package: `nodes`, `edges`, `edges_trail`, `edges_deviation`, `nodes_fixed`, `trails`,
`sequences`, and `sequences_edges`. Every shape the kernel carries is one the structure
states, since the padded node array that phase carried, `nodes_padded`, went with the
sentinel that indexed it.

The shapes are verified rather than asserted. The root `conftest.py` installs
jaxtyping's import hook with `beartype`, which turns every annotation into a runtime
check for the whole suite. It found three that were wrong on its first run: the
constructor accepted a flat edge array where it declared pairs,
`sequences_edges_indices` was named after the flattened sequence grid when it holds
one entry per trail edge, and the sequence lengths were documented as a column where
they are a vector. The hook has to sit in a conftest rather than in the pytest
`addopts`, because its plugin parses the raw command line before the ini options
merge into it.

The structure fields become JAX arrays, which is the decision recorded above and the
part of it that had not landed. The trail search still computes with NumPy and the
conversion happens once, at the boundary between the search and the structure.

`pyright` was already clean, which phase 1 reached by correcting the annotations that
lied rather than by adding shapes: the model took the empty `Structure` base class
where it reads `Structure` attributes, two `vmap`-ed parameters were
declared `int` where they receive a traced scalar, and `node_index`, `edge_index`,
`sequences_edges` and `sequences_edges_indices` were declared `jax.Array` where they
hold dicts and NumPy arrays. This phase keeps it clean while adding the shapes.

Two functions could not be annotated honestly and went instead. `vector_length` took
a `keepdims` flag that changed its return shape and that nothing ever passed, and
`trail_length` was dead code carrying a comment that described a shape it never
returned.

### Phase 4 — Tests

The six expected-result dictionaries in `test_equilibrium.py` are already literal
Python and survive COMPAS removal untouched; they are the frozen baseline that
phases 1 and 2 are each measured against. The fixtures that build them are ported in
phase 6.

This phase adds what the suite has never had: `jit`, `vmap`, and gradient tests, a
trail-sequencing unit test independent of equilibrium, and the `compas_xcheck` tests
that solve each fixture with `compas_cem.equilibrium.static_equilibrium` and compare
every node and every edge against it.

A frozen baseline records one setting; a live solver also covers the path that reaches
it, so the two run at a matching iteration limit and convergence threshold. The
cross-check was verified to bite rather than to pass vacuously, by swapping the two
scatters of the deviation accumulation and confirming which comparisons fail.

Two invariants have no COMPAS CEM counterpart, and asking whether they held is what
exposed that `EquilibriumState.residuals` was still the trail quantity rather than the
Formax one. Both are stated on the corrected field: the residual vanishes at every free
node, and the residuals sum to the applied load.

They fail in different ways, which is why both are kept. The free-node check measures
convergence, because the residual is assembled from the edge forces rather than read
off the sweep; the sum holds to machine precision at any iteration count, because the
internal forces cancel in pairs whether or not the sweep has settled, so what it checks
is the bookkeeping between the sweep, the edge forces, and the loads.

They call no solver, so they belong with the kernel tests rather than the cross-checks,
and they run in both places: over structures built natively, which is where they stay,
and over the six fixtures, which are the richest ones available until phase 6 ports
them.

### Phase 5 — Examples

Ported to the native API: `02_braced_tower_2d` (it has a test baseline),
`03_bridge_2d` (indirect deviation edges, JSON input), and `05_tensegrity_wheel_2d`
(exercises planes). The rest, including the MEM paper artifacts, move to
`examples/legacy/`, excluded from ruff and CI, and are ported as the API firms up.
`optimization_basic.py` imports both `jaxopt` and `optax`; the port picks one, and
`optimistix` is where `smax` and `jax_fdm` both landed.

### Phase 6 — Fixtures, documentation, and metadata

The six topology fixtures are ported to the array constructor, which deletes
`tests/converters.py` and gives the constructor its first exercise outside the unit
tests. This is the earliest phase that can hold it. The converters are what the phase 5
examples are ported against, and the frozen baselines are the measure that phases 2 and
5 are each held to, so moving the port earlier would remove the evidence while it is
still being read.

`compas_cem` then leaves the dev group and returns as an optional extra, for the
examples and the `compas_xcheck` tests. It does not leave the repository: those tests
are the standing check that the rewrite computes what COMPAS CEM computes, and they
need a solver to call.

A real `Unreleased` entry in `CHANGELOG.md`, including the note that
`deviation_edges` has changed meaning from a float mask to an edge array.
`AUTHORS.md` and `CONTRIBUTING.md` cleared of COMPAS. A documentation site is
deferred: `formax` and `smax` ship none, and the `mkdocs` and `mike` setup in
`jax_fdm` is the template when one is wanted.


### Phase 7 — Forward-mode differentiation

An equilibrium differentiates in reverse and not forward. The sweep is not what blocks
it: `lax.scan` transforms both ways, and a model at `tmax=1`, which skips the iteration,
answers `jacfwd` and `jacrev` alike. The iteration is. A checkpointed while loop is a
`custom_vjp`, and JAX refuses to apply a JVP to one, so `jacfwd` on a model that
iterates raises instead of returning a tangent.

Equinox answers this in one word. `kind="bounded"` differentiates both ways, and it
lowers to a base-16 tree of `scan` whose levels are guarded by `cond` and rematerialized
by `jax.checkpoint`, so it keeps both the early exit and the flat memory in `tmax`.
It changes nothing about what a gradient means: the same exact derivative of the
truncated iterate, agreeing with the checkpointed loop to the bit. It pays for the
second direction in the first one, at 1.7 times the gradient time and 7 times its peak
memory on a hundred node grid, and 8 times the memory on a four hundred node one. Under
`vmap` its `cond` lowers to a select and the early exit goes with it, which is the one
thing the checkpointed loop does that no scan-shaped loop can.

So the phase is a settings question before it is a port. The loop kind can be a field of
the model, defaulting to the checkpointed loop and switched where a forward derivative
is wanted, since nothing else about the iteration differs between the two.

A custom JVP rule on the fixed point is the alternative, and it is only worth writing if
that memory is worth removing rather than paying. It has to be a JVP and not another VJP,
because reverse mode is the transpose of a linearized forward rule and so falls out of
the one rule where a `custom_vjp` only ever covers the one direction. Differentiating
`x = f(a, x)` at the fixed point gives `(I - df/dx) dx = (df/da) da`, and the tangent
solve has to be transposable for the reverse direction to survive it, which
`lax.custom_linear_solve` is and a bare `lax.while_loop` is not. What to weigh is that
such a rule changes what a gradient means. The checkpointed loop returns the exact
derivative of the iterate it computed, truncation and all; an implicit rule returns the
derivative of the converged fixed point. The two agree once the iteration has converged
and part company silently when `tmax` cuts it short, by half a percent on a grid stopped
at three of the ten steps it needed.


## Open questions

- **Version.** `0.1.0` was never released and the repository has no tags, so phase
  1 can adopt `dynamic = ["version"]` with nothing to migrate.
- **Structure arrays in the Formax contract.** `AbstractStructure` annotates
  `nodes`, `edges`, and `supports` as JAX arrays, which `jax_cem` follows. Whether
  the contract should also admit NumPy for static index data, as `jax_fdm` stores
  it, is a decision for `formax`.
