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
```

Jupyter is already registered as the **Python 3.15 (highway)** kernel. Re-register it
after recreating `.venv`:

```bash
uv run python -m ipykernel install --user --name highway --display-name "Python 3.15 (highway)"
```

## Commands

| Task              | Command                                     |
| ----------------- | ------------------------------------------- |
| Notebook UI       | `uv run jupyter notebook`                   |
| JupyterLab UI     | `uv run jupyter lab`                        |
| Lint (incl. ipynb)| `uv run ruff check .`                       |
| Auto-fix lint     | `uv run ruff check --fix .`                 |
| Format            | `uv run ruff format .`                      |
| Type check        | `uv run ty check`                           |
| Tests             | `uv run pytest`                             |
| Run the script    | `uv run highway`                            |
| Add a dependency  | `uv add <pkg>` / `uv add --dev <pkg>`       |

## Config

Tool settings live in `pyproject.toml`:

- `[tool.ruff]` - line length 100, rules `E,W,F,I,B,C4,SIM,PTH,RET,UP,RUF`
- `[tool.ty]` - checks `src/` and `tests/`, ignores notebooks
- `[tool.pytest.ini_options]` - tests live in `tests/`
