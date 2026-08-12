# jax_cem

The combinatorial equilibrium modeling framework in JAX.

`jax_cem` computes the static equilibrium of pin-jointed bar structures with the
combinatorial equilibrium modeling (CEM) form-finding algorithm. The
implementation is differentiable, jittable, and vectorizable, so an equilibrium
state can be used directly as a term in a gradient-based design problem.

> **Status.** Under active refresh. The public API is being reshaped around an
> array-based structure that satisfies the Formax equilibrium contract; see
> [`docs/roadmap.md`](docs/roadmap.md).

## Installation

```bash
pip install jax_cem
```

Python 3.11 to 3.13.

## Development

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev          # create the environment
uv run pre-commit install    # install the ruff hook
uv run pytest                # run the tests
uv run ruff check .          # lint
uv run ruff format .         # format
```

Cross-checks against the reference COMPAS CEM implementation are marked
`compas_xcheck`. Run the suite without them with:

```bash
uv run pytest -m "not compas_xcheck"
```

## License

Apache-2.0. See [LICENSE](LICENSE).
