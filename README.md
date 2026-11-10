# Highway Networks on CIFAR-10

DeepL course project: a minimal, pedagogical PyTorch implementation of
[Highway Networks](https://arxiv.org/abs/1505.00387) (Srivastava, Greff, Schmidhuber, 2015),
trained on CIFAR-10. The whole project is one notebook:
[`highway_cifar10.ipynb`](highway_cifar10.ipynb).

## What's inside

- the paper's highway layer: `y = H(x) * T(x) + x * (1 - T(x))`
- the paper's 19-layer convolutional CIFAR-10 architecture: 14 highway layers, 4 plain
  convolutions, dropout, and a 1x1 100-channel classifier head
- about 2.26M parameters
- negative transform-gate bias initialization, also from the paper
- paper-inspired training: Nesterov SGD with momentum, He initialization, dropout 0.2,
  exponential learning-rate decay, 100 epochs with decay 0.95
- CIFAR-10 training and evaluation, loss/accuracy curves
- gradient flow at initialization and after training
- gate analysis: bias and activity per layer, evolution during training, selectivity for one image
- class-specific routing and layer-lesioning plots from the paper's analysis
- an information-flow plot and a t-SNE view of the learned features
- paper Figure 1-style plain-vs-highway depth comparison on MNIST
- paper Figure 2-style per-channel gate maps and block-output heatmaps

## Run locally

```bash
uv sync                  # create .venv and install everything
uv run jupyter lab       # open the notebook
```

Open `highway_cifar10.ipynb` and select the **Python 3.15 (highway)** kernel. A GPU is
recommended but not required. The 19-layer convolutional run fits in 8 GB and takes roughly 45
minutes for the highway model on an RTX 3060 Ti. The MNIST depth comparison adds about 10 minutes.
CIFAR-10 and MNIST are downloaded to `data/` on the first run (git-ignored).

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
