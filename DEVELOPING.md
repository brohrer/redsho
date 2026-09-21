# Developing Redsho

Documentation for anyone working on the code itself (probably just me).

## Setup

### `uv`

I recommend installing `uv` if you haven't already.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Repo

This page assumes a local copy of the repository.

```bash
git clone https://codeberg.org/brohrer/redsho.git
```

### Package installation

Install the package as editable.

```bash
uv pip install -e .
```

or if you're not using `uv`

```bash
python3 -m pip install -e redsho
```

Similarly, if you want to install `redsho` for another `uv` project
you're working in

```bash
uv pip install -e <path-to-local-redsho-repo>
```

## Testing

### Unit testing

Unit tests are built to run quickly and give a rapid-iteration check
for non-brokenness.
To run unit tests, at the top repo dir:

```bash
uv run pytest
```

to run them with prints enabled

```bash
uv run pytest -s
```

### Integration testing

Integration tests put more of the pieces together. These take longer
to run and so have been broken out into a separate test suite.

```bash
uv run tests/integration_test_optimizer.py -s
```

### End-to-end testing (examples)

These are intended to be even more comprehensive than integration tests,
taking plausible application examples and running them from start to
finish. They are also intended as reasonable starting points for users
to copy/paste implementations from.

The run times on these can be quite long, and they are set up to be
run deliberately and one at a time.

Examples are collected in the
[`redsho-examples` repo](https://codeberg.org/brohrer/redsho-examples)
and each comes with its own documentation.

### Writing tests

Adding tests is one of the best ways to keep future you from pulling your
hair out.

Change a function? Add a test to verify that it works the way you expect.
Even if it appears obviously correct to you.

Fix a bug? After you fix it add a test that would have caught it.

Making a big change? Add an integration test and an example that put it
through its paces.

## Workflow

I've found a good rhythm in checking large code changes file by file. 

1. Make some changes in `<filename>`
2. Write/modify tests to cover them in `<test_filename>`
3. `uvx ruff check <filename> <test_filename>`
4. `uvx ruff format <filename> <test_filename>`
5. `uv run mypy <filename> <test_filename>`
6. `uv run pytest -s <test_filename>`

Substantial errors at any stage usually means falling back to the beginning.
It seems tedious, but flows smoothly after a while.

## Pushing new commits

Before pushing a new commit (a serious one, not a work-in-progress save)
I find it helpful to do similar checks but across the repo. From
the top level of the repo.

1. `uvx ruff check`
2. `uvx ruff format`
3. `uv run mypy .`
4. `uv run pytest -s`
5. `uv run pytest -s tests/<integration_test>`

## Publishing to PyPI

And when it's time to publish a new version for the wide world to see,
after doing all the above

1. Increment the version number in `pyproject.toml` appropriately,
    as needed by semantic versioning: `<major-change>.<minor-change>.<bugfix>`
2. Carefully update the `README.md`
3. `uv build`
4. `uv publish --token <pypi_token>`
