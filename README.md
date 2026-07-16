# decisionpy

A Python library for modeling and solving **mixed-type limited-memory influence diagrams (LIMIDs)**, built on top of **NumPyro**.

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups             # install dev/lint/test dependencies
uv run pytest                    # run tests
pre-commit install
```

