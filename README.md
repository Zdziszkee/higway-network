# Highway Networks on CIFAR-10

DeepL course project: a minimal, pedagogical PyTorch implementation of
[Highway Networks](https://arxiv.org/abs/1505.00387) (Srivastava, Greff, Schmidhuber, 2015),
trained on CIFAR-10. The whole project is one notebook:
[`highway_cifar10.ipynb`](highway_cifar10.ipynb).

## What's inside

- the paper's highway layer: `y = H(x) * T(x) + x * (1 - T(x))`
- a convolutional CIFAR-10 model at the paper's depth-20 convention: plain stem plus 19 highway blocks
- about 2.7M parameters
- negative transform-gate bias initialization, also from the paper
- paper-style training: SGD with momentum, He initialization, exponential learning-rate decay
- CIFAR-10 training and evaluation, loss/accuracy curves
- gradient flow at initialization and after training
- gate analysis: bias and activity per layer, evolution during training, selectivity for one image
- class-specific routing and layer-lesioning plots from the paper's analysis
- an information-flow plot and a t-SNE view of the learned features

## Run locally

```bash
uv sync                  # create .venv and install everything
uv run jupyter lab       # open the notebook
```

Open `highway_cifar10.ipynb` and select the **Python 3.15 (highway)** kernel. A GPU is
recommended but not required. The depth-20 convolutional run fits in 8 GB and takes roughly 20 to
30 minutes for the highway model on a modern GPU. CIFAR-10 is downloaded to `data/` on the first run
(git-ignored).

If you recreate the virtual environment, re-register the project-local kernel:

```bash
uv run python -m ipykernel install --sys-prefix --name highway --display-name "Python 3.15 (highway)"
```

## Run on Google Colab

Upload `highway_cifar10.ipynb`, then choose *Runtime > Change runtime type > GPU*. Colab has
`torch`, `torchvision`, and `matplotlib` preinstalled, so there is no setup cell to run.

## Tooling

| Task               | Command                |
| ------------------ | ---------------------- |
| Lint (incl. ipynb) | `uv run ruff check .`  |
| Fix + format       | `uv run ruff check --fix . && uv run ruff format .` |
| Type check         | `uv run ty check`      |
