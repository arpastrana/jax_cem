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

`EquilibriumStructure` is constructed from arrays, not from a declarative layer of
`Node` and `Edge` objects.

```python
class EquilibriumStructure(AbstractStructure):
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

The concatenation is trail-major, and that order is canonical: `sequences_edges`
indexes it, and any source with its own edge order — JSON, a mesh, a COMPAS CEM
diagram — is permuted into it at the boundary.

Two consequences reach past the constructor. Trail and deviation edges occupy
contiguous blocks, so the masks the kernel currently multiplies through
(`deviation_edges`, `indirect_edges`) become slices, and `indirect_edges` shrinks
to a mask over the deviation block alone. Trail-edge forces are outputs, recovered
from the residuals, so `ParameterState.forces` covers only the deviation block;
the trail entries the old parameter array carried were overwritten on every call
and never read.

A declarative layer remains possible later as a constructor-side module that emits
these arrays. It is deferred rather than rejected: it adds nothing the kernel or
the Formax adapter needs, and building it first would mean merging topology and
parameters in one object only to separate them again one function later.

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

### License

Apache-2.0, following `formax` and `smax`. The patent grant is the stronger choice
for a library `formax` is built on. `jax_fdm` and `compas_cem` remain MIT.

### COMPAS

No COMPAS or COMPAS CEM code inside the `jax_cem` package. `compas_cem` stays as an
optional dependency used by the examples and by cross-check tests marked
`compas_xcheck`, on the pattern `jax_fdm` uses for deletable scaffolding.

`compas.numerical.connectivity_matrix` gains a local NumPy implementation. This is
forced rather than chosen: COMPAS 2.x removed `compas.numerical` entirely, which is
the only reason the package was pinned to `compas==1.17.10`. `compas_cem` 0.8.6
already requires `compas>=2.15,<3.0` on Python 3.10–3.13, so the optional extra
installs alongside modern dependencies.

The native trail search is the substantive piece of work in the refresh. `jax_cem`
does not own trail sequencing today; it reads `trails()`, `node_sequence()`, and
`is_indirect_deviation_edge()` off a COMPAS CEM topology diagram. Trail discovery,
sequence assignment, shifted-trail padding, auxiliary trails, and direct versus
indirect deviation edge classification all move into this repository.


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
Phase 2 replaces them with the native array constructor and deletes the module.

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
change from mask multiplication to block slicing in the kernel. Deletes the
`from_topology_diagram` constructors and the COMPAS imports in `structures.py` and
`parameters.py`.

The six topology fixtures are ported to the array constructor in the same phase,
which is what keeps the suite green across it and gives the constructor its first
real exercise. `compas_cem` drops out of the dev group here and reappears only as an
optional extra, for the examples and the `compas_xcheck` tests.

`EquilibriumState` adopts the Formax field set: `residuals` rather than `reactions`
(a reaction is the negation of a residual, and storing one of the two removes the
chance of disagreement), an added `vectors`, and `forces` and `lengths` shaped
`"edges"` rather than `"edges 1"`.

The sentinel that currently encodes a plane-driven trail edge as `length == 0.0`
becomes an explicit mask on the structure. Whether a trail edge is driven by a
length or by a plane is a fact about the topology, not a reserved value in the
parameters.

### Phase 3 — Typing

jaxtyping shapes replace the `# N x 3` comments throughout, including the fields
still annotated with a question mark.

`pyright` is already clean, which phase 1 reached by correcting the annotations that
lied rather than by adding shapes: the model took the empty `Structure` base class
where it reads `EquilibriumStructure` attributes, two `vmap`-ed parameters were
declared `int` where they receive a traced scalar, and `node_index`, `edge_index`,
`sequences_edges` and `sequences_edges_indices` were declared `jax.Array` where they
hold dicts and NumPy arrays. This phase keeps it clean while adding the shapes.

### Phase 4 — Tests

The six expected-result dictionaries in `test_equilibrium.py` are already literal
Python and survive COMPAS removal untouched; they are the frozen baseline that
phases 1 and 2 are each measured against. The fixtures that build them are ported in
phase 2.

This phase adds what the suite has never had: `jit`, `vmap`, and gradient tests, a
trail-sequencing unit test independent of equilibrium, and the `compas_xcheck` tests
that compare against `compas_cem.equilibrium.static_equilibrium`.

### Phase 5 — Examples

Ported to the native API: `02_braced_tower_2d` (it has a test baseline),
`03_bridge_2d` (indirect deviation edges, JSON input), and `05_tensegrity_wheel_2d`
(exercises planes). The rest, including the MEM paper artifacts, move to
`examples/legacy/`, excluded from ruff and CI, and are ported as the API firms up.
`optimization_basic.py` imports both `jaxopt` and `optax`; the port picks one, and
`optimistix` is where `smax` and `jax_fdm` both landed.

### Phase 6 — Documentation and metadata

A real `Unreleased` entry in `CHANGELOG.md`, including the note that
`deviation_edges` has changed meaning from a float mask to an edge array.
`AUTHORS.md` and `CONTRIBUTING.md` cleared of COMPAS. A documentation site is
deferred: `formax` and `smax` ship none, and the `mkdocs` and `mike` setup in
`jax_fdm` is the template when one is wanted.


## Open questions

- **Version.** `0.1.0` was never released and the repository has no tags, so phase
  1 can adopt `dynamic = ["version"]` with nothing to migrate.
- **Structure arrays in the Formax contract.** `AbstractStructure` annotates
  `nodes`, `edges`, and `supports` as JAX arrays, which `jax_cem` follows. Whether
  the contract should also admit NumPy for static index data, as `jax_fdm` stores
  it, is a decision for `formax`.
