
= Reference tables and likely questions

== Tensor shape flow

#table(
  columns: (auto, auto, 1fr),
  inset: 5pt,
  align: (left, left, left),
  table.header([*Stage*], [*Tensor shape*], [*What happens*]),
  [Input], [`(B, 3, 32, 32)`], [CIFAR-10 batch.],  [Stem], [`(B, 32, 32, 32)`], [Plain 3×3 convolution, 3 to 32 channels, ReLU.],  [Stage 1], [`(B, 32, 32, 32)`], [Six highway blocks at 32 channels.],  [Down 1], [`(B, 64, 16, 16)`], [Plain 3×3 convolution, stride 2, 32 to 64 channels, ReLU.],  [Stage 2], [`(B, 64, 16, 16)`], [Six highway blocks at 64 channels.],  [Down 2], [`(B, 128, 8, 8)`], [Plain 3×3 convolution, stride 2, 64 to 128 channels, ReLU.],  [Stage 3], [`(B, 128, 8, 8)`], [Seven highway blocks at 128 channels.],  [Pool], [`(B, 128)`], [Global average pooling over the 8×8 spatial grid.],  [Head], [`(B, 10)`], [Linear classifier producing 10 logits.],)

=== Gate tensor shape

Inside a highway block, `T` has shape `(B, C, H, W)`. There is one gate value for every image, channel, and spatial position. The analysis reduces it in three ways: `mean(dim=(1,2,3))` gives one number per image; `mean()` gives one number per layer; `mean(dim=0)` on a single image gives a spatial map averaged over channels.

== Parameter breakdown

#table(
  columns: (auto, auto, 1fr),
  inset: 5pt,
  align: (left, right, left),
  table.header([*Part*], [*Parameters*], [*Why*]),
  [Stem 3 to 32], [896], [3×3×3×32 weights plus 32 biases.],
  [6 blocks at 32], [110,976], [Each block has two convs: 2 × (32×32×3×3 + 32) = 18,496.],
  [Down 1, 32 to 64], [18,496], [32×64×3×3 + 64.],
  [6 blocks at 64], [443,136], [Each block: 2 × (64×64×3×3 + 64) = 73,856.],
  [Down 2, 64 to 128], [73,856], [64×128×3×3 + 128.],
  [7 blocks at 128], [2,066,176], [Each block: 2 × (128×128×3×3 + 128) = 295,168.],
  [Head 128 to 10], [1,290], [128×10 weights plus 10 biases.],
  [*Total*], [*2,714,826*], [Printed by the notebook.],
)

== Hyperparameter rationale

#table(
  columns: (auto, auto, 1fr),
  inset: 5pt,
  align: (left, left, left),
  table.header([*Setting*], [*Value*], [*Why this value*]),
  [Seed], [`0`], [Reproducible initialization and data order.],
  [Batch size], [`128`], [Paper reproduction and standard SGD batch size.],
  [Epochs], [`25`], [Reduced from 150 to keep the run near 20 to 30 minutes.],
  [Initial LR], [`0.1`], [Paper CIFAR-10 convolutional rate.],
  [LR decay], [`0.89` per epoch], [Chosen so 25 epochs end near the same LR as 150 epochs with 0.98.],
  [Momentum], [`0.9`], [Standard SGD momentum; the paper uses SGD with momentum.],
  [Weight decay], [`5e-4`], [Practical CIFAR-10 regularization; not stated explicitly in the paper.],
  [Gradient clipping], [`10`], [Stability addition; not in the paper.],
  [Gate bias], [`-2`], [Paper's negative initialization. `sigmoid(-2) ≈ 0.12`.],
  [Blocks], [`19`, split `6/6/7`], [Depth-20 convention with one stem.],
  [Channels], [`32/64/128`], [Compact widths for an 8 GB GPU.],
  [Workers], [`2`], [Parallel data loading; use 0 on Windows if needed.],
)

== Depth counting

There are three different counts, and they are all correct in different contexts:

- *Paper-style depth:* 20. One plain stem plus 19 hidden highway blocks. This matches the paper's depth-20 convention.
- *Highway blocks:* 19. Each block contains a transform convolution `H` and a gate convolution `T`.
- *Physical convolution layers:* 41. One stem convolution, 38 block convolutions, and two downsampling convolutions. The classifier is a linear layer, not a convolution.

== Likely questions and short answers

=== Why use convolution instead of a fully connected layer?
The paper's CIFAR-10 experiments use convolutional highway layers. Convolutions share weights across spatial positions, preserve local structure, and need far fewer parameters than a dense layer on 32×32×3 inputs.

=== Why is the kernel 3×3 with padding 1?
A 3×3 convolution with padding 1 preserves the height and width. The highway equation adds the input to the transformed output, so both must have the same spatial size. Padding 1 makes that true.

=== Why can a highway block not change channels?
The carry term is `x * (1 - T)`. If `H(x)` had a different number of channels from `x`, the addition would be invalid. Channel-changing layers are therefore plain convolutions, as in Section 2.1 of the paper.

=== Why 19 blocks?
The paper's depth-20 network is one plain layer followed by 19 highway layers. Nineteen blocks give that depth while keeping the run around 20 to 30 minutes.

=== Why are there 41 convolutions if the depth is 20?
Each highway block has two convolutions, `H` and `T`. The paper counts a highway block as one layer. Nineteen blocks therefore give 38 convolutions, plus one stem and two downsampling convolutions, for 41 physical convolutions.

=== Why is the transform gate a sigmoid?
The gate must be between 0 and 1 so the layer is a convex combination of the transformed input and the original input. A sigmoid guarantees that range and is smooth and differentiable.

=== Why start the gate bias at -2?
`sigmoid(-2) ≈ 0.12`, so gates start mostly closed. The deep network initially behaves like a shallower, carry-heavy network and opens gates during training. This is the paper's Section 2.2 recommendation.

=== Why He initialization?
The transform branch uses ReLU. He normal initialization is designed to preserve signal variance in ReLU networks. The paper initializes the non-gate weights with the He et al. scheme.

=== Why reset the gate bias after initialization?
`init_weights` zeros every bias, including the gate biases. The second loop restores the -2 value for every highway block. Without that reset, the negative initialization would be lost.

=== Why use SGD with momentum instead of Adam?
The paper trains with SGD and momentum. Adam would be a different optimizer and would make the comparison to the paper less direct.

=== Why learning rate 0.1?
It is the paper's CIFAR-10 convolutional rate. The paper random-searched per-network rates, so 0.1 is a representative shared value rather than a universal optimum.

=== Why decay 0.89 instead of 0.98?
The schedule must fit 25 epochs, not 150. `0.1 * 0.89^25 ≈ 0.005`, which is close to `0.1 * 0.98^150 ≈ 0.005`. This keeps the final learning rate similar.

=== Why 25 epochs?
It keeps the run near 20 to 30 minutes on the RTX 3060 Ti. The tradeoff is lower accuracy and less gate convergence than a 150-epoch run.

=== Why weight decay 5e-4?
It is a practical CIFAR-10 regularization value that helps convolutional models generalize. The paper does not state it explicitly, so it is a documented addition.

=== Why gradient clipping at 10?
Deep ReLU networks with SGD can occasionally produce very large gradients. Clipping the total norm at 10 keeps training stable. The paper does not mention clipping, so it is an implementation addition.

=== Why is there no softmax before the loss?
`CrossEntropyLoss` expects raw logits and applies log-softmax internally. This is numerically more stable than applying softmax first.

=== Why global average pooling?
It turns the final `(B, 128, 8, 8)` feature map into `(B, 128)` without extra parameters. It also makes the classifier independent of the exact spatial position of features.

=== Why is `shuffle=True` only for training?
Shuffling prevents the optimizer from seeing a fixed batch order and improves SGD. Test order is kept fixed so results and probe batches are reproducible.

=== Why `pin_memory` only on CUDA?
Pinned memory speeds up host-to-GPU transfers. It has no benefit on CPU and can produce a warning there.

=== Why `set_to_none=True` in `zero_grad`?
It sets gradients to `None` instead of allocating zero tensors. This is slightly cheaper and is the recommended PyTorch default for many training loops.

=== Why `detach` when storing gates?
The stored gate tensor is for inspection only. Detaching it removes it from the autograd graph, so it does not keep graph memory alive across iterations.

=== Why `retain_grad` in the gradient experiment?
The gradient experiment needs the gradient of non-leaf tensors, the inputs to each block. PyTorch does not keep those by default, so `retain_grad()` is called on each one.

=== Why does the gradient helper repeat the downsampling logic?
The helper runs the model manually to capture intermediate tensors. It must apply `down1` and `down2` at the same block indices as `Net.forward`, or the channel counts will not match.

=== Why average the gate map over channels?
A 64-channel 16×16 gate tensor is too dense to read as one image. Averaging over channels gives a readable spatial map. The histogram still shows the full distribution.

=== Why does lesioning use accuracy instead of loss?
Accuracy is easy to interpret. The downside is noise: with 10,000 test images, differences below about 0.5 percentage points are near the noise floor. Training or test loss would be more sensitive.

=== Why is t-SNE qualitative?
t-SNE preserves local neighborhoods, not global distances. Cluster sizes and between-cluster distances are not meaningful, and the result changes with seed and perplexity.

=== Why is there only one model now?
The plain baseline was removed by request. The notebook now studies the highway model alone. It can no longer directly demonstrate that highway networks beat plain networks at depth.

== Known inconsistencies and caveats

- *Stale Results sentence.* Cell 14 still says "Two identical training setups, with and without the gates." The plain baseline has been removed, so this sentence is outdated.
- *Stale LR comment.* Cell 3 says the learning rate "exposes the depth-20 optimization gap", but there is no plain model in the notebook now.
- *No direct depth comparison.* Without a plain baseline, the notebook cannot prove the paper's depth claim. It can only show that the highway model trains and how its gates behave.
- *Short training.* Twenty-five epochs is much shorter than the paper's setup. Gates may not fully converge, and gate/lesioning plots may be noisy.
- *Compact architecture.* The model is not an exact FitNet reproduction. It uses a depth-20 convention and compact widths.
- *Lesioning noise.* Small accuracy drops should not be over-interpreted.
- *Single-image gate map.* Averaging over channels can hide per-channel selectivity. The histogram is the more reliable part.

== Paper map

#table(
  columns: (1fr, 1fr),
  inset: 5pt,
  align: (left, left),
  table.header([*Paper item*], [*Where it appears in the notebook*]),
  [Equation (3), the highway layer], [Cell 0 theory, cell 7 `Highway.forward`, line 17.],
  [Equations (4) and (5), identity and Jacobian cases], [Cell 0 derivation, `gate_off` lesioning in line 14-15.],
  [Section 2.1, convolutional highways and plain dimensionality changes], [Cell 8 description and cell 9 plain stem and downsampling convolutions.],
  [Section 2.2, negative gate bias], [Cell 3 `GATE_BIAS = -2` and cell 7 line 7.],
  [Section 3.1, MLP depth study], [Not reproduced. The notebook uses a CIFAR-10 convolutional model.],
  [Table 1, CIFAR-10 convolutional highways], [Depth-20 convolutional model with 2.71M parameters. This is a compact version, not an exact FitNet layout.],
  [Figure 2, gate biases and activity], [Cells 18-23.],
  [Figure 3, class-specific routing], [Cells 24-25.],
  [Figure 4, layer lesioning], [Cells 28-29.],
  [Longer paper 1507.06228], [Source for the convolutional highway construction and the gate/lesioning analyses.],
)

== One-paragraph summary to memorize

The notebook builds a depth-20 convolutional highway network for CIFAR-10. A plain stem changes the input to 32 channels. Three stages of highway blocks use 32, 64, and 128 channels, with plain strided convolutions between stages. Each highway block computes `y = H(x) * T(x) + x * (1 - T(x))`, where `H` and `T` are 3×3 convolutions with padding 1 and the gate bias starts at -2. The model has 19 highway blocks, 41 physical convolutions, and 2,714,826 parameters. It is trained with SGD momentum 0.9, learning rate 0.1, weight decay 5e-4, decay 0.89 per epoch, gradient clipping 10, batch size 128, for 25 epochs. The analysis plots gradient flow, gate biases and activity, gate evolution, single-image gate maps, class routing, input-output similarity, lesioning, and t-SNE features.
