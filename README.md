# highway

DeepL course project. Python 3.15 managed with [uv](https://docs.astral.sh/uv/).

## Layout

```
src/highway/      package code
notebooks/        Jupyter notebooks
tests/            pytest tests
```

## Setup

```bash
uv sync                      # create .venv and install everything (incl. dev tools)
make kernel                  # register the Jupyter kernel (already done in this checkout)
```

The venv is registered as the **Python 3.15 (highway)** kernel, project-locally inside
`.venv/share/jupyter/kernels/highway` (no global Jupyter state). Re-register it if you
recreate the venv:

```bash
uv run python -m ipykernel install --sys-prefix --name highway --display-name "Python 3.15 (highway)"
```

## Commands

Run `make` to list targets.

| Task               | make      | raw command                            |
| ------------------ | --------- | -------------------------------------- |
| Notebook UI        | `make notebook` | `uv run jupyter notebook notebooks` |
| JupyterLab UI      | `make lab`      | `uv run jupyter lab notebooks`      |
| Lint (incl. ipynb) | `make lint`     | `uv run ruff check .`               |
| Fix + format       | `make fmt`      | `uv run ruff check --fix . && uv run ruff format .` |
| Type check         | `make typecheck`| `uv run ty check`                   |
| Tests              | `make test`     | `uv run pytest`                     |
| Everything         | `make check`    | lint + typecheck + tests            |
| Run the script     | -               | `uv run highway`                    |
| Add a dependency   | -               | `uv add <pkg>` / `uv add --dev <pkg>` |

## Config

Tool settings live in `pyproject.toml`:

- `[tool.ruff]` - line length 100, rules `E,W,F,I,B,C4,SIM,PTH,RET,UP,RUF`; both lint and
  format apply to `.ipynb` cells as well as `.py` files
- `[tool.ty]` - checks `src/` and `tests/`, ignores notebooks
- `[tool.pytest.ini_options]` - tests live in `tests/`
