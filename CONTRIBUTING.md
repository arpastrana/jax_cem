# Contributing

Contributions are welcome and very much appreciated!

## Code contributions

We accept code contributions through pull requests.
In short, this is how that works.

1. Fork [the repository](https://github.com/arpastrana/jax_cem) and clone the fork.
2. Create the development environment with [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync --group dev
   uv run pre-commit install
   ```

3. Make sure all tests pass:

   ```bash
   uv run pytest
   ```

4. Start making your changes on a branch off **main**.
5. Make sure all tests still pass, and that the linter and the type checker are
   clean:

   ```bash
   uv run pytest
   uv run ruff check .
   uv run pyright
   ```

6. Add an entry under `## Unreleased` in `CHANGELOG.md`. A pull request without
   one fails its checks.
7. Add yourself to the *Contributors* section of `AUTHORS.md`.
8. Commit your changes and push your branch to GitHub.
9. Create a [pull request](https://help.github.com/articles/about-pull-requests/) through the GitHub website.

## Bug reports

When [reporting a bug](https://github.com/arpastrana/jax_cem/issues) please include:

* Operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

## Feature requests

When [proposing a new feature](https://github.com/arpastrana/jax_cem/issues) please include:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
