.DEFAULT_GOAL := help
PYTHON_VERSION := 3.15
KERNEL_NAME := highway
KERNEL_DISPLAY := Python $(PYTHON_VERSION) ($(KERNEL_NAME))

.PHONY: help sync kernel notebook lab lint fmt typecheck test check

help: ## show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

sync: ## create .venv and install all dependencies (default)
	uv sync

kernel: ## (re)register the venv as the "$(KERNEL_DISPLAY)" Jupyter kernel
	uv run python -m ipykernel install --sys-prefix --name $(KERNEL_NAME) --display-name "$(KERNEL_DISPLAY)"

notebook: ## open the Jupyter Notebook UI
	uv run jupyter notebook notebooks

lab: ## open the JupyterLab UI
	uv run jupyter lab notebooks

lint: ## check linting (includes notebooks)
	uv run ruff check .

fmt: ## auto-fix and format everything
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## run ty
	uv run ty check

test: ## run the test suite
	uv run pytest

check: lint typecheck test ## lint + typecheck + tests
