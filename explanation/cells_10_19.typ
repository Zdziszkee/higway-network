
== Cell 10: Training setup heading

*Role of this cell.* Lists the paper-inspired training recipe: SGD with momentum 0.9, batch size 128, He initialization, weight decay 5e-4, exponential learning-rate decay by 0.89 per epoch, and 25 epochs.
*Why 25 epochs.* The authors' released Caffe config uses 200k iterations, about 400 epochs at batch 100. Twenty-five epochs keeps this pedagogical run near 13 to 15 minutes while still giving the gates time to move.
*Why decay 0.89.* The paper's released config uses step decay at 100k/150k/175k iterations, not a per-epoch exponential. For this short run, 0.89 brings the learning rate down to about 0.005.
*Why learning rate 0.1.* It is a high CIFAR-10 rate that stays stable here because the highway shortcuts preserve gradient flow. The authors' released config uses base LR 0.025 with Nesterov momentum.
*Why weight decay and clipping are mentioned as additions.* The paper does not explicitly state weight decay or gradient clipping. Weight decay 5e-4 is a practical CIFAR-10 choice, and clipping at 10 keeps SGD stable.
*Gate tracking.* The mean gate value of every block is recorded after every epoch for the gate-evolution heatmap.

== Cell 11: Accuracy, gate statistics, and training loop

*What this cell does.* Defines three functions: `accuracy` for evaluation, `gate_means_on_batch` for gate statistics, and `train` for the full SGD loop. The training loop is the only place where parameters are updated.

```
  1 | @torch.no_grad()
  2 | def accuracy(model, loader):
  3 |     model.eval()
  4 |     correct = total = 0
  5 |     for x, y in loader:
  6 |         x, y = x.to(device), y.to(device)
  7 |         correct += (model(x).argmax(dim=1) == y).sum().item()
  8 |         total += y.numel()
  9 |     return correct / total
 10 | 
 11 | 
 12 | @torch.no_grad()
 13 | def gate_means_on_batch(model, x):
 14 |     model.eval()
 15 |     model(x)
 16 |     return [block.last_gate.mean().item() for block in model.blocks]
 17 | 
 18 | 
 19 | def train(model, track_gates: bool = False, epochs: int = EPOCHS, lr: float = LR):
 20 |     model.to(device)
 21 |     opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
 22 |     sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.89)
 23 |     loss_fn = nn.CrossEntropyLoss()
 24 | 
 25 |     history = {"loss": [], "acc": [], "gates": []}
 26 |     start = time.time()
 27 |     for epoch in range(1, epochs + 1):
 28 |         model.train()
 29 |         running = 0.0
 30 |         for x, y in train_loader:
 31 |             x, y = x.to(device), y.to(device)
 32 |             opt.zero_grad(set_to_none=True)
 33 |             loss = loss_fn(model(x), y)
 34 |             loss.backward()
 35 |             torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
 36 |             opt.step()
 37 |             running += loss.item() * y.numel()
 38 |         history["loss"].append(running / len(train_set))
 39 |         history["acc"].append(accuracy(model, test_loader))
 40 |         if track_gates:
 41 |             history["gates"].append(gate_means_on_batch(model, probe_x))
 42 |         print(
 43 |             f"epoch {epoch:3d}/{epochs} | lr {opt.param_groups[0]['lr']:.4f} "
 44 |             f"| loss {history['loss'][-1]:.3f} | test acc {history['acc'][-1]:.1%}"
 45 |         )
 46 |         sched.step()
 47 | 
 48 |     print(f"trained in {time.time() - start:.1f}s")
 49 |     return history
```

*Line by line:*

- *Line 1:* `@torch.no_grad()` disables gradient tracking for `accuracy`, saving memory and time during evaluation.
- *Line 2:* `def accuracy(model, loader)` defines the evaluation function.
- *Line 3:* `model.eval()` switches to evaluation mode. This model has no dropout or batch normalization, but `eval()` is still correct practice.
- *Line 4:* `correct = total = 0` initializes the counters for correct predictions and examples.
- *Line 5:* Loops over the loader one batch at a time.
- *Line 6:* Moves the images and labels to the selected device.
- *Line 7:* `model(x).argmax(dim=1)` picks the class with the highest logit. Comparing with `y`, summing, and calling `.item()` counts correct predictions as a Python integer.
- *Line 8:* `total += y.numel()` adds the batch size.
- *Line 9:* Returns the fraction correct.
- *Line 12:* `@torch.no_grad()` disables gradients for gate inspection.
- *Line 13:* `def gate_means_on_batch(model, x)` defines the gate-statistics helper.
- *Line 14:* Switches to evaluation mode.
- *Line 15:* Runs one forward pass so every block stores its `last_gate` tensor.
- *Line 16:* Returns the mean gate value of each block. `mean()` averages over batch, channels, height, and width, producing one number per layer.
- *Line 19:* `def train(model, track_gates=False, epochs=EPOCHS, lr=LR)` defines the training function. `track_gates` controls whether gate means are recorded.
- *Line 20:* Moves the model to the device.
- *Line 21:* Creates SGD with momentum 0.9 and weight decay 5e-4. SGD with momentum is the paper's optimizer.
- *Line 22:* `ExponentialLR(gamma=0.89)` multiplies the learning rate by 0.89 after every epoch. The final rate is about `0.1 * 0.89^25 ≈ 0.0054`.
- *Line 23:* `CrossEntropyLoss` combines log-softmax and negative log-likelihood, so the model outputs raw logits.
- *Line 25:* `history` stores training loss, test accuracy, and optional gate means.
- *Line 26:* Records the start time for the runtime report.
- *Line 27:* Loops over epochs from 1 to `epochs`.
- *Line 28:* `model.train()` switches to training mode.
- *Line 29:* `running = 0.0` accumulates the total loss for the epoch.
- *Line 30:* Loops over training batches.
- *Line 31:* Moves the batch to the device.
- *Line 32:* `opt.zero_grad(set_to_none=True)` clears old gradients. Setting them to `None` is slightly cheaper than zeroing them.
- *Line 33:* Forward pass and cross-entropy loss.
- *Line 34:* Backpropagation computes all gradients.
- *Line 35:* `clip_grad_norm_(..., 10.0)` caps the total gradient norm at 10. This is not in the paper but prevents occasional SGD explosions.
- *Line 36:* `opt.step()` updates the parameters.
- *Line 37:* `running += loss.item() * y.numel()` converts the batch mean loss back to a sum. Dividing by the dataset size later gives the true epoch mean.
- *Line 38:* Appends the epoch's average training loss.
- *Line 39:* Evaluates test accuracy after the epoch.
- *Lines 40-41:* If `track_gates` is true, records the mean gate value of every block on the fixed probe batch.
- *Lines 42-45:* Prints epoch, current learning rate, loss, and test accuracy. `opt.param_groups[0]["lr"]` is the rate actually used during that epoch.
- *Line 46:* `sched.step()` decays the learning rate after the epoch.
- *Line 48:* Prints total training time.
- *Line 49:* Returns the history dictionary.

== Cell 12: Train the highway network heading

*Role of this cell.* A section heading before the model is created and trained.

== Cell 13: Create and train the highway model

*What this cell does.* Seeds the run, creates the 19-block convolutional highway network, applies He initialization and the gate-bias reset, prints the parameter count, and trains for 25 epochs with gate tracking.

```
  1 | set_seed()
  2 | highway = Net()
  3 | init_weights(highway)
  4 | print(f"{n_params(highway):,} parameters")
  5 | highway_hist = train(highway, track_gates=True)
```

*Line by line:*

- *Line 1:* `set_seed()` fixes the random seed before model creation.
- *Line 2:* `highway = Net()` creates the model with the default `(32, 64, 128)` channels and `(6, 6, 7)` blocks.
- *Line 3:* `init_weights(highway)` applies He initialization and resets every gate bias to -2.
- *Line 4:* Prints the parameter count. The model has 2,714,826 parameters.
- *Line 5:* Trains for 25 epochs and records gate means after every epoch.

== Cell 14: Results heading

*Role of this cell.* A heading for the training curves. The body now says that there is no plain baseline, so this is not the paper's depth comparison.

== Cell 15: Loss and accuracy curves

*What this cell does.* Plots the highway model's training loss and test accuracy over the 25 epochs.

```
  1 | fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 3.5))
  2 | 
  3 | ax_loss.plot(highway_hist["loss"], color="tab:orange")
  4 | ax_acc.plot([100 * a for a in highway_hist["acc"]], color="tab:orange")
  5 | 
  6 | ax_loss.set(xlabel="epoch", ylabel="train loss", title="Training loss")
  7 | ax_acc.set(xlabel="epoch", ylabel="test accuracy (%)", title="Test accuracy")
  8 | for ax in (ax_loss, ax_acc):
  9 |     ax.grid(alpha=0.3)
 10 | 
 11 | plt.tight_layout()
 12 | plt.show()
 13 | 
 14 | print(f"final test accuracy: {highway_hist['acc'][-1]:.1%}")
```

*Line by line:*

- *Line 1:* Creates one figure with two side-by-side axes.
- *Line 3:* Plots training loss against epoch in orange.
- *Line 4:* Plots test accuracy as a percentage. `100 * a` converts fractions to percent.
- *Line 6:* Labels the loss axes: x is epoch, y is training loss.
- *Line 7:* Labels the accuracy axes: x is epoch, y is test accuracy in percent.
- *Lines 8-9:* Adds a light grid to both panels.
- *Line 11:* `tight_layout` prevents label overlap.
- *Line 12:* Displays the figure.
- *Line 14:* Prints the final test accuracy. Because the run is only 25 epochs, this may be below the best value seen in a longer run.

== Cell 16: Gradient flow heading

*Role of this cell.* Explains the gradient-flow experiment. Backpropagation multiplies one Jacobian per layer. Without a shortcut, those products can shrink the gradient with depth. A highway layer's identity path lets gradients pass through more easily.
*What the next plot shows.* The gradient norm arriving at each block input, measured once at initialization and once after training.

== Cell 17: Measure gradient norm at each block input

*What this cell does.* Builds a fresh untrained model, runs one batch through the model manually, keeps the gradient of every block input, and plots the norms at initialization versus after training.

```
  1 | def layer_input_grad_norms(model, x, y):
  2 |     model.train()
  3 |     model.zero_grad(set_to_none=True)
  4 | 
  5 |     h = model.stem(x)
  6 |     hs = []
  7 |     for i, block in enumerate(model.blocks):
  8 |         if i == model.break1:
  9 |             h = model.down1(h)
 10 |         if i == model.break2:
 11 |             h = model.down2(h)
 12 |         h.retain_grad()
 13 |         hs.append(h)
 14 |         h = block(h)
 15 | 
 16 |     loss = F.cross_entropy(model.head(h.mean(dim=(2, 3))), y)
 17 |     loss.backward()
 18 | 
 19 |     norms = [max(t.grad.norm().item(), 1e-12) for t in hs]
 20 |     model.zero_grad(set_to_none=True)
 21 |     return norms
 22 | 
 23 | 
 24 | set_seed()
 25 | fresh_highway = Net().to(device)
 26 | init_weights(fresh_highway)  # match the initialization used for the trained model
 27 | 
 28 | x_grad, y_grad = next(iter(train_loader))
 29 | x_grad, y_grad = x_grad[:256].to(device), y_grad[:256].to(device)
 30 | 
 31 | fig, ax = plt.subplots(figsize=(8, 3.5))
 32 | layers = range(1, len(highway.blocks) + 1)
 33 | ax.semilogy(
 34 |     layers,
 35 |     layer_input_grad_norms(fresh_highway, x_grad, y_grad),
 36 |     "o-",
 37 |     label="at initialization",
 38 | )
 39 | ax.semilogy(
 40 |     layers,
 41 |     layer_input_grad_norms(highway, x_grad, y_grad),
 42 |     "o-",
 43 |     label="after training",
 44 | )
 45 | highway.eval()
 46 | ax.set(
 47 |     xlabel="layer",
 48 |     ylabel="gradient norm at block input (log scale)",
 49 |     title="Gradient reaching each layer",
 50 | )
 51 | ax.grid(alpha=0.3, which="both")
 52 | ax.legend()
 53 | plt.tight_layout()
 54 | plt.show()
```

*Line by line:*

- *Line 1:* `def layer_input_grad_norms(model, x, y)` defines the measurement function.
- *Line 2:* `model.train()` puts the model in training mode. There is no dropout or batch normalization, so this mainly matches training conditions.
- *Line 3:* Clears old gradients.
- *Line 5:* Runs the stem to get the input to the first block.
- *Line 6:* `hs = []` will store the tensor that enters each block.
- *Line 7:* Loops over all 19 blocks by index.
- *Lines 8-9:* Before block 6, apply `down1`, matching the model's forward pass.
- *Lines 10-11:* Before block 12, apply `down2`.
- *Line 12:* `h.retain_grad()` keeps the gradient of this non-leaf tensor so it can be read after backpropagation.
- *Line 13:* Appends the block input to `hs`.
- *Line 14:* Applies the block.
- *Line 16:* Applies global average pooling, the classifier, and cross-entropy. Pooling must be included because `Net.forward` does it before the head.
- *Line 17:* Backpropagates through the whole network.
- *Line 19:* For each stored block input, reads the gradient norm. `max(..., 1e-12)` avoids zero values on a log scale.
- *Line 20:* Clears gradients after measuring.
- *Line 21:* Returns one norm per block.
- *Line 24:* Seeds the fresh model.
- *Line 25:* `fresh_highway = Net().to(device)` creates an untrained copy for the initialization measurement.
- *Line 26:* `init_weights(fresh_highway)` applies the same He initialization and gate-bias reset as the trained model. Without it, the "at initialization" curve would use PyTorch's default initialization and would not be comparable.
- *Lines 28-29:* Takes one training batch of 256 images for the gradient estimate.
- *Line 31:* Creates one figure and one axis.
- *Line 32:* `layers` is the 1-based layer index for the x-axis.
- *Lines 33-38:* Plots gradient norms at initialization with circle markers and a log y-axis.
- *Lines 39-44:* Plots gradient norms after training on the same axes.
- *Line 45:* `highway.eval()` restores evaluation mode after the measurement.
- *Lines 46-50:* Sets axis labels and title.
- *Line 51:* Adds a grid for both major and minor ticks.
- *Line 52:* Adds the legend.
- *Lines 53-54:* Lays out and displays the figure.

== Cell 18: Gate bias and gate activity heading

*Role of this cell.* Explains the paper's Figure 2 idea for its 50-layer networks: inspect the learned transform-gate biases and the mean gate activity per layer. A negative bias does not necessarily mean a dead layer. It often makes gates selective: some units stay closed while others open for particular inputs.
*What the bars show.* The average gate value of each layer over the whole test set. In this run the biases move toward zero and the gates open on average, which differs from the paper's deeper networks.

== Cell 19: Gate bias and mean activity per layer

*What this cell does.* Computes the mean learned gate bias per layer and the mean gate activity per layer over the test set, then plots both.

```
  1 | def gate_biases(model):
  2 |     return [block.gate.bias.mean().item() for block in model.blocks]
  3 | 
  4 | 
  5 | @torch.no_grad()
  6 | def gate_activity(model, loader):
  7 |     model.eval()
  8 |     sums = torch.zeros(len(classes), len(model.blocks))
  9 |     counts = torch.zeros(len(classes))
 10 |     for x, y in loader:
 11 |         model(x.to(device))
 12 |         batch_means = torch.stack(
 13 |             [block.last_gate.mean(dim=(1, 2, 3)).cpu() for block in model.blocks],
 14 |             dim=1,
 15 |         )
 16 |         for c in range(len(classes)):
 17 |             mask = y == c
 18 |             sums[c] += batch_means[mask].sum(dim=0)
 19 |             counts[c] += mask.sum()
 20 |     return sums.sum(dim=0) / counts.sum(), sums / counts[:, None]
 21 | 
 22 | 
 23 | overall_activity, class_activity = gate_activity(highway, test_loader)
 24 | biases = gate_biases(highway)
 25 | 
 26 | fig, (ax_bias, ax_act) = plt.subplots(1, 2, figsize=(11, 3.5))
 27 | ax_bias.plot(range(1, len(highway.blocks) + 1), biases, "o-", label="learned")
 28 | ax_bias.axhline(GATE_BIAS, color="gray", linestyle="--", label="initial")
 29 | ax_bias.set(xlabel="layer", ylabel="gate bias", title="Transform-gate bias per layer")
 30 | ax_bias.legend()
 31 | ax_bias.grid(alpha=0.3)
 32 | 
 33 | ax_act.bar(range(1, len(highway.blocks) + 1), overall_activity, color="tab:orange")
 34 | ax_act.set(
 35 |     xlabel="layer",
 36 |     ylabel="mean T",
 37 |     title="Mean gate activity on the test set",
 38 |     ylim=(0, 1),
 39 | )
 40 | ax_act.grid(alpha=0.3, axis="y")
 41 | 
 42 | plt.tight_layout()
 43 | plt.show()
```

*Line by line:*

- *Line 1:* `def gate_biases(model)` defines the bias helper.
- *Line 2:* For each block, averages the gate-bias vector into one number and returns one value per layer.
- *Line 5:* `@torch.no_grad()` disables gradients for gate statistics.
- *Line 6:* `def gate_activity(model, loader)` defines the activity helper.
- *Line 7:* Switches to evaluation mode.
- *Line 8:* `sums` has shape `(10, 19)`: one total per class and layer.
- *Line 9:* `counts` stores how many examples of each class were seen.
- *Line 10:* Loops over the test loader.
- *Line 11:* Runs the batch so every block stores `last_gate`.
- *Lines 12-15:* For each block, averages gate values over channels and spatial positions, giving one value per sample. Stacks into `(batch, layers)`.
- *Line 16:* Loops over the 10 classes.
- *Line 17:* `mask = y == c` selects the samples of class `c`.
- *Line 18:* Adds those samples' gate activity to the class totals.
- *Line 19:* Adds the sample count.
- *Line 20:* Returns the overall mean per layer and the per-class means with shape `(10, 19)`. `counts[:, None]` makes counts a column for broadcasting.
- *Line 23:* Computes both statistics for the trained highway model.
- *Line 24:* Gets the learned gate biases.
- *Line 26:* Creates two side-by-side panels.
- *Line 27:* Plots learned bias per layer.
- *Line 28:* Draws the initial -2 bias as a dashed reference.
- *Line 29:* Labels the bias axes.
- *Line 30:* Adds the legend.
- *Line 31:* Adds a grid.
- *Line 33:* Bar chart of mean gate activity per layer.
- *Lines 34-39:* Labels the axes, titles the panel, and fixes the y-range to 0 to 1.
- *Line 40:* Adds a horizontal grid at the y-axis ticks.
- *Lines 42-43:* Lays out and displays the figure.
