
== Cell 0: The highway layer: theory reference

*Role of this cell.* This is the reference page for every formula used later. It is not executed. When someone asks why the code looks the way it does, the answer is usually one of the equations here.
*The layer equation.* A plain layer is `y = H(x, W_H)`. A highway layer adds two gates: `y = H(x, W_H) * T(x, W_T) + x * C(x, W_C)`. `T` is the transform gate and `C` is the carry gate. The paper sets `C = 1 - T`, so the layer becomes `y = H(x) * T(x) + x * (1 - T(x))`.
*Why two gates.* The gate decides, per unit, how much of the output comes from the nonlinear transform and how much is copied from the input. This is a learned interpolation, not a fixed residual connection.
*The sigmoid gate.* The paper defines `T(x) = sigmoid(W_T^T x + b_T)`. The sigmoid keeps every gate value strictly between 0 and 1, so the layer can move smoothly between copying and transforming.
*The two limiting cases.* If `T(x) = 0`, then `y = x`, so the layer is the identity. If `T(x) = 1`, then `y = H(x)`, so the layer is a plain layer. The code never reaches 0 or 1 exactly because a sigmoid is asymptotic, but it can get close.
*Why the Jacobian matters.* For one unit, `y_i = T_i(x) * H_i(x) + (1 - T_i(x)) * x_i`. With a completely closed gate, `T_i(x) = 0`, this becomes `y_i = x_i`. The derivative is `dy_i / dx_j = delta_ij`, so the Jacobian is the identity matrix `I`. A gradient arriving at the layer output is multiplied by `I` on the way back, so it reaches the previous layer unchanged. This is the information highway.
*The precise Jacobian caveat.* A real sigmoid is never exactly zero and depends on `x`. The exact Jacobian is `J = diag(T) * H'(x) + diag(1 - T) + diag(H(x) - x) * T'(x)`. The identity is the ideal, locally constant-gate case. Even when the last term is nonzero, the `diag(1 - T)` term keeps a direct gradient path.
*Shape requirement.* Equation (3) adds `H(x)` and `x`, so `x`, `y`, `H(x)`, and `T(x)` must have the same shape. This is why the network starts with a plain stem and uses plain layers when changing channels or resolution.
*Negative gate bias.* The paper starts the gate bias at a negative value such as -1, -2, or -3. The notebook uses -2, so `sigmoid(-2) ≈ 0.12`. The network starts biased toward carrying, behaving like a shallower network, and the gates open during training where transforming helps.
*How to use this cell in a presentation.* Point to the equation, then to the two limiting cases, then to the Jacobian argument. The rest of the notebook is evidence about whether the gates actually behave this way.

== Cell 1: Setup heading

*Role of this cell.* A section heading only. It separates the theory from the executable setup.

== Cell 2: Imports and device selection

*What this cell does.* Loads every library used later and chooses the GPU or CPU. It prints the PyTorch version and the GPU name so the run is reproducible and the hardware is known.

```
  1 | import random
  2 | import time
  3 | from pathlib import Path
  4 | 
  5 | import matplotlib.pyplot as plt
  6 | import torch
  7 | import torch.nn as nn
  8 | import torch.nn.functional as F
  9 | from sklearn.manifold import TSNE
 10 | from torch.utils.data import DataLoader
 11 | from torchvision import datasets, transforms
 12 | 
 13 | print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
 14 | device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 15 | if device.type == "cuda":
 16 |     print(torch.cuda.get_device_name(0))
```

*Line by line:*

- *Line 1:* `import random` loads Python's random-number generator, used by `set_seed` for reproducibility.
- *Line 2:* `import time` is used to measure how long training takes.
- *Line 3:* `from pathlib import Path` creates the `data` directory path in a platform-independent way.
- *Line 5:* `import matplotlib.pyplot as plt` is the plotting library used for every figure.
- *Line 6:* `import torch` loads PyTorch: tensors, autograd, and device handling.
- *Line 7:* `import torch.nn as nn` loads neural-network building blocks, including `Conv2d`, `Linear`, and `CrossEntropyLoss`.
- *Line 8:* `import torch.nn.functional as F` loads functional operations used directly, such as `relu`, `sigmoid`, `cosine_similarity`, and `cross_entropy`.
- *Line 9:* `from sklearn.manifold import TSNE` loads scikit-learn's t-SNE implementation for the feature-space plot.
- *Line 10:* `from torch.utils.data import DataLoader` loads the batching and shuffling utility.
- *Line 11:* `from torchvision import datasets, transforms` loads CIFAR-10 and the image preprocessing pipeline.
- *Line 13:* Prints the PyTorch version and whether CUDA is available. This confirms the environment before training.
- *Line 14:* Selects `cuda` if PyTorch can see a GPU, otherwise `cpu`. Every model and batch is moved to this device later.
- *Line 15:* Checks whether the selected device is CUDA.
- *Line 16:* If CUDA is selected, prints the GPU name, which confirms the RTX 3060 Ti is being used.

== Cell 3: Configuration and seeding

*What this cell does.* Defines the small set of values that control the run and defines `set_seed`. The configuration is deliberately short: architecture details live with the model, and optimizer details live inside `train`.

```
  1 | SEED = 0
  2 | DATA_ROOT = Path("data")
  3 | BATCH_SIZE = 128
  4 | EPOCHS = 25
  5 | LR = 0.1  # paper CIFAR rate; exposes the depth-20 optimization gap
  6 | GATE_BIAS = -2.0
  7 | NUM_WORKERS = 2  # set to 0 on Windows if the data loader hangs
  8 | 
  9 | 
 10 | def set_seed(seed=SEED):
 11 |     random.seed(seed)
 12 |     torch.manual_seed(seed)
 13 |     torch.cuda.manual_seed_all(seed)
```

*Line by line:*

- *Line 1:* `SEED = 0` fixes the random seed so initialization and data order are repeatable. Zero is a conventional, easy-to-quote choice.
- *Line 2:* `DATA_ROOT = Path("data")` puts the CIFAR-10 download in `./data`, which is git-ignored.
- *Line 3:* `BATCH_SIZE = 128` is the number of images per training step. The paper does not report a batch size; 128 is the value used by the public reproduction and is standard for SGD.
- *Line 4:* `EPOCHS = 25` means 25 full passes over the 50,000 training images. This was reduced from 150 to keep the run near 20 to 30 minutes.
- *Line 5:* `LR = 0.1` is the initial SGD learning rate. It is the rate used in the paper's CIFAR-10 convolutional runs and is high enough to expose optimization differences at depth. The comment mentions the depth-20 gap, but the notebook now trains only the highway model.
- *Line 6:* `GATE_BIAS = -2.0` is the initial transform-gate bias. `sigmoid(-2) ≈ 0.12`, so gates start mostly closed, as recommended in Section 2.2 of the paper.
- *Line 7:* `NUM_WORKERS = 2` uses two background processes for data loading. Set it to 0 on Windows if the loader hangs.
- *Lines 10-13:* `set_seed` seeds Python, PyTorch's CPU generator, and every CUDA generator. Calling it before model creation makes runs comparable.

== Cell 4: Data heading

*Role of this cell.* Explains the dataset and the augmentation choices. CIFAR-10 has 50,000 training and 10,000 test images of size 32×32 in 10 classes. Random crops and flips approximate the paper's random translations and mirroring. Both splits are normalized with the dataset mean and standard deviation.

== Cell 5: Dataset, transforms, and loaders

*What this cell does.* Builds the CIFAR-10 training and test pipelines, wraps them in `DataLoader` objects, and saves a fixed probe batch for later gate and gradient analysis.

```
  1 | MEAN = (0.4914, 0.4822, 0.4465)
  2 | STD = (0.2470, 0.2435, 0.2616)
  3 | 
  4 | train_tf = transforms.Compose(
  5 |     [
  6 |         transforms.RandomCrop(32, padding=4),
  7 |         transforms.RandomHorizontalFlip(),
  8 |         transforms.ToTensor(),
  9 |         transforms.Normalize(MEAN, STD),
 10 |     ]
 11 | )
 12 | test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
 13 | 
 14 | train_set = datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=train_tf)
 15 | test_set = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=test_tf)
 16 | classes = test_set.classes
 17 | 
 18 | train_loader = DataLoader(
 19 |     train_set,
 20 |     batch_size=BATCH_SIZE,
 21 |     shuffle=True,
 22 |     num_workers=NUM_WORKERS,
 23 |     pin_memory=device.type == "cuda",
 24 | )
 25 | test_loader = DataLoader(
 26 |     test_set,
 27 |     batch_size=512,
 28 |     shuffle=False,
 29 |     num_workers=NUM_WORKERS,
 30 |     pin_memory=device.type == "cuda",
 31 | )
 32 | 
 33 | # A fixed batch used later to inspect the gates.
 34 | probe_x, _ = next(iter(test_loader))
 35 | probe_x = probe_x[:256].to(device)
 36 | 
 37 | print(f"{len(train_set):,} train / {len(test_set):,} test images")
```

*Line by line:*

- *Line 1:* `MEAN = (0.4914, 0.4822, 0.4465)` is the per-channel mean of CIFAR-10. It is used to center the inputs.
- *Line 2:* `STD = (0.2470, 0.2435, 0.2616)` is the per-channel standard deviation. It scales the inputs to roughly unit variance.
- *Lines 4-11:* `train_tf` is the training transform pipeline. The order matters: augmentation first, then tensor conversion, then normalization.
- *Line 6:* `RandomCrop(32, padding=4)` pads the image to 40×40 and takes a random 32×32 crop. This is translation augmentation, matching the paper's use of small translations.
- *Line 7:* `RandomHorizontalFlip()` mirrors the image with probability 0.5, matching the paper's mirroring augmentation.
- *Line 8:* `ToTensor()` converts a PIL image to a float tensor, changes H×W×C to C×H×W, and scales values to the range 0 to 1.
- *Line 9:* `Normalize(MEAN, STD)` applies `(x - mean) / std` per channel.
- *Line 12:* `test_tf` contains only tensor conversion and normalization. Test data must not be augmented, so evaluation is deterministic.
- *Line 14:* Creates the CIFAR-10 training set, downloads it if missing, and applies `train_tf`.
- *Line 15:* Creates the test set with `test_tf`.
- *Line 16:* `classes = test_set.classes` stores the 10 class names for plot labels and prediction titles.
- *Lines 18-24:* The training `DataLoader` uses batch size 128, shuffles every epoch, uses two workers, and pins memory only when CUDA is available.
- *Line 20:* `batch_size=BATCH_SIZE` uses the configured 128.
- *Line 21:* `shuffle=True` reshuffles the training data every epoch, which is important for SGD.
- *Line 22:* `num_workers=NUM_WORKERS` loads batches in parallel.
- *Line 23:* `pin_memory=True` only on CUDA can speed up host-to-GPU copies.
- *Lines 25-31:* The test `DataLoader` uses batch size 512 because evaluation has no gradients and can use larger batches. `shuffle=False` keeps the order stable.
- *Lines 33-35:* `probe_x` is a fixed batch of 256 test images moved to the device. It is reused for gate statistics, gradient measurements, and layer-similarity plots so those analyses are comparable across calls.
- *Line 37:* Prints the dataset sizes with thousands separators.

== Cell 6: The highway layer in code heading

*Role of this cell.* Explains that the next code cell is the convolutional version of Equation (3). One 3×3 convolution computes `H`, another computes gate logits, and a sigmoid produces `T`. Padding 1 keeps the spatial size unchanged. The full gate tensor is stored for later inspection.

== Cell 7: The convolutional highway block

*What this cell does.* Defines one highway block. It is the central architectural unit of the whole notebook. Both the transform and the gate are 3×3 convolutions with padding 1, so the block preserves channels and spatial resolution and the carry connection is shape-valid.

```
  1 | class Highway(nn.Module):
  2 |     def __init__(self, channels: int, gate_bias: float = GATE_BIAS):
  3 |         super().__init__()
  4 |         self.H = nn.Conv2d(channels, channels, 3, padding=1)
  5 |         self.gate = nn.Conv2d(channels, channels, 3, padding=1)
  6 |         assert self.gate.bias is not None
  7 |         nn.init.constant_(self.gate.bias, gate_bias)  # start biased towards carry
  8 |         self.last_gate = torch.zeros(1)  # filled on every forward pass
  9 |         self.gate_off = False  # used later to switch this layer off
 10 | 
 11 |     def forward(self, x):
 12 |         h = F.relu(self.H(x))
 13 |         t = torch.sigmoid(self.gate(x))
 14 |         if self.gate_off:
 15 |             t = torch.zeros_like(t)
 16 |         self.last_gate = t.detach()
 17 |         return h * t + x * (1 - t)
```

*Line by line:*

- *Line 1:* `class Highway(nn.Module)` defines a PyTorch module, so its parameters are tracked and moved with the rest of the network.
- *Line 2:* `__init__(self, channels, gate_bias=GATE_BIAS)` takes the number of input and output channels. A highway block cannot change the channel count because it adds the input to the transformed output.
- *Line 3:* `super().__init__()` initializes the base `nn.Module`.
- *Line 4:* `self.H = nn.Conv2d(channels, channels, 3, padding=1)` is the transform branch. It is a 3×3 convolution with padding 1, so the output has the same height, width, and channels as the input.
- *Line 5:* `self.gate = nn.Conv2d(channels, channels, 3, padding=1)` is the gate branch. It produces one logit per channel and spatial position.
- *Line 6:* `assert self.gate.bias is not None` documents and narrows the type. `Conv2d` has a bias by default, and the next line needs it.
- *Line 7:* `nn.init.constant_(self.gate.bias, gate_bias)` sets every gate bias to -2. This is the paper's negative initialization, which starts the network biased toward carrying.
- *Line 8:* `self.last_gate = torch.zeros(1)` is a placeholder for the most recent gate tensor. It is overwritten on every forward pass and is used by the analysis cells.
- *Line 9:* `self.gate_off = False` is a switch used by the lesioning experiment. When true, the block is forced to copy its input.
- *Line 11:* `def forward(self, x)` defines the forward pass.
- *Line 12:* `h = F.relu(self.H(x))` computes the transformed state `H(x)` and applies ReLU. ReLU is the paper's CIFAR-10 activation.
- *Line 13:* `t = torch.sigmoid(self.gate(x))` turns the gate logits into values between 0 and 1.
- *Lines 14-15:* If `gate_off` is true, `t` is replaced by zeros. This is the lesioning mechanism: a closed gate forces `y = x`.
- *Line 16:* `self.last_gate = t.detach()` stores the gate tensor without autograd history, so the analysis cells can inspect it without keeping a computation graph alive.
- *Line 17:* `return h * t + x * (1 - t)` implements `y = H(x) * T(x) + x * (1 - T(x))` elementwise. `h * t` is the transformed part and `x * (1 - t)` is the carried part.

== Cell 8: The convolutional highway network heading

*Role of this cell.* Describes the full model: a plain 3×3 stem, two plain strided convolutions that halve the spatial size and change channels, nineteen highway blocks split 6/6/7, global average pooling, and a linear classifier.
*Depth convention.* With the stem, this is the paper's depth-20 convention: one plain layer followed by 19 hidden layers. The stage widths are 32, 64, and 128.
*Why the transitions are plain.* `H` and `T` both preserve channels and spatial size. Layers that change channels or resolution cannot be highway blocks because the carry addition would have mismatched shapes. Section 2.1 of the paper uses plain layers for this.
*Honesty note.* This is a compact version of the paper's convolutional setup, not an exact FitNet reproduction.
*Initialization.* All weights use He initialization. Gate biases are reset to -2 after initialization so the gates start mostly closed.

== Cell 9: The full convolutional network

*What this cell does.* Defines the complete highway network, the parameter counter, and the initialization routine. The model has 19 highway blocks, 41 physical convolution layers (1 stem + 38 block convolutions + 2 downsampling convolutions), one global average pooling step, and one linear classifier.

```
  1 | class Net(nn.Module):
  2 |     def __init__(
  3 |         self,
  4 |         channels: tuple[int, int, int] = (32, 64, 128),
  5 |         blocks_per_stage: tuple[int, int, int] = (6, 6, 7),
  6 |     ):
  7 |         super().__init__()
  8 |         c1, c2, c3 = channels
  9 |         self.stem = nn.Sequential(
 10 |             nn.Conv2d(3, c1, 3, padding=1),
 11 |             nn.ReLU(),
 12 |         )
 13 |         self.blocks = nn.ModuleList(
 14 |             Highway(ch) for ch, n in zip(channels, blocks_per_stage, strict=True) for _ in range(n)
 15 |         )
 16 |         self.break1 = blocks_per_stage[0]
 17 |         self.break2 = blocks_per_stage[0] + blocks_per_stage[1]
 18 |         self.down1 = nn.Sequential(
 19 |             nn.Conv2d(c1, c2, 3, stride=2, padding=1),
 20 |             nn.ReLU(),
 21 |         )
 22 |         self.down2 = nn.Sequential(
 23 |             nn.Conv2d(c2, c3, 3, stride=2, padding=1),
 24 |             nn.ReLU(),
 25 |         )
 26 |         self.head = nn.Linear(c3, 10)
 27 | 
 28 |     def forward(self, x):
 29 |         x = self.stem(x)
 30 |         for i, block in enumerate(self.blocks):
 31 |             if i == self.break1:
 32 |                 x = self.down1(x)
 33 |             if i == self.break2:
 34 |                 x = self.down2(x)
 35 |             x = block(x)
 36 |         x = x.mean(dim=(2, 3))  # global average pooling
 37 |         return self.head(x)
 38 | 
 39 | 
 40 | def n_params(model):
 41 |     return sum(p.numel() for p in model.parameters())
 42 | 
 43 | 
 44 | def init_weights(model):
 45 |     """He initialization for every conv and linear layer, then reset the gate biases."""
 46 |     for module in model.modules():
 47 |         if isinstance(module, (nn.Linear, nn.Conv2d)):
 48 |             nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
 49 |             if module.bias is not None:
 50 |                 nn.init.zeros_(module.bias)
 51 |     for module in model.modules():
 52 |         if isinstance(module, Highway):
 53 |             assert module.gate.bias is not None
 54 |             nn.init.constant_(module.gate.bias, GATE_BIAS)
```

*Line by line:*

- *Line 1:* `class Net(nn.Module)` defines the full model.
- *Lines 2-6:* The constructor takes `channels=(32, 64, 128)` and `blocks_per_stage=(6, 6, 7)`. The block counts sum to 19, matching the paper's depth-20 convention when the stem is included.
- *Line 7:* `super().__init__()` initializes the base module.
- *Line 8:* `c1, c2, c3 = channels` unpacks the three stage widths.
- *Lines 9-12:* `self.stem` is a plain 3×3 convolution from 3 RGB channels to 32 channels, followed by ReLU. It changes the input dimensionality before any highway block.
- *Lines 13-15:* `self.blocks` builds the 19 highway blocks in stage order. The nested comprehension pairs each channel width with its block count, and `strict=True` checks that the two tuples have the same length.
- *Line 16:* `self.break1 = 6` is the index where stage 1 ends. The first downsampling happens before block index 6.
- *Line 17:* `self.break2 = 12` is the index where stage 2 ends. It is `6 + 6`, so the second downsampling happens before block index 12.
- *Lines 18-21:* `self.down1` is a plain 3×3 convolution with stride 2 from 32 to 64 channels, followed by ReLU. It halves 32×32 to 16×16.
- *Lines 22-25:* `self.down2` is a plain 3×3 convolution with stride 2 from 64 to 128 channels, followed by ReLU. It halves 16×16 to 8×8.
- *Line 26:* `self.head = nn.Linear(c3, 10)` maps the 128 pooled features to 10 class logits.
- *Line 28:* `def forward(self, x)` defines the forward pass.
- *Line 29:* `x = self.stem(x)` applies the stem.
- *Line 30:* The loop visits all 19 highway blocks in order.
- *Lines 31-32:* Before block 6, apply `down1`. This moves from the 32-channel 32×32 stage to the 64-channel 16×16 stage.
- *Lines 33-34:* Before block 12, apply `down2`. This moves from the 64-channel 16×16 stage to the 128-channel 8×8 stage.
- *Line 35:* `x = block(x)` applies the current highway block.
- *Line 36:* `x = x.mean(dim=(2, 3))` is global average pooling. It turns the `(batch, 128, 8, 8)` feature map into `(batch, 128)`.
- *Line 37:* `return self.head(x)` returns 10 logits per image. No softmax is applied because `CrossEntropyLoss` expects logits.
- *Lines 40-41:* `n_params` sums the number of elements in every learnable parameter. The model has 2,714,826 parameters.
- *Lines 44-54:* `init_weights` applies He initialization to every convolution and linear layer, zeros their biases, and then restores the -2 gate bias for every highway block.
- *Line 48:* `nn.init.kaiming_normal_(module.weight, nonlinearity="relu")` is PyTorch's He normal initialization, appropriate for ReLU networks.
- *Line 49-50:* Biases are zeroed only if they exist. This keeps the function safe for layers without a bias.
- *Lines 51-54:* The second loop finds every `Highway` block and resets its gate bias to -2. This is necessary because the first loop zeros all biases.
