
== Cell 20: Gate evolution heading

*Role of this cell.* Explains the gate-evolution heatmap: rows are highway layers, columns are epochs. Bright cells mean the layer transforms its input, dark cells mean it mostly carries. At initialization the gates are near `sigmoid(-2) ≈ 0.12`; the first recorded column is already after one epoch, and in this compact run the gates open on average.

== Cell 21: Gate-evolution heatmap

*What this cell does.* Turns the recorded per-epoch gate means into a layer-by-epoch heatmap.

```
  1 | gate_history = torch.tensor(highway_hist["gates"]).T  # layers x epochs
  2 | 
  3 | plt.figure(figsize=(8, 4))
  4 | plt.imshow(gate_history, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
  5 | plt.colorbar(label="mean transform gate T")
  6 | plt.xlabel("epoch")
  7 | plt.ylabel("highway layer")
  8 | plt.title("Mean gate value of every layer during training")
  9 | plt.tight_layout()
 10 | plt.show()
```

*Line by line:*

- *Line 1:* `highway_hist["gates"]` has shape `(epochs, layers)`. Transposing gives `(layers, epochs)`, which displays naturally as rows equal layers and columns equal epochs.
- *Line 3:* Creates the figure.
- *Line 4:* `imshow` displays the matrix. `aspect="auto"` stretches cells to fit, `origin="lower"` puts layer 1 at the bottom, and `vmin=0, vmax=1` fixes the color scale to the valid gate range.
- *Line 5:* Adds a color bar labeled with the mean transform gate.
- *Line 6:* Labels the x-axis as epoch.
- *Line 7:* Labels the y-axis as highway layer.
- *Line 8:* Adds the title.
- *Lines 9-10:* Lays out and displays the figure.

== Cell 22: Single-image gate map heading

*Role of this cell.* Explains the single-image gate inspection. In a convolutional highway layer, the gate is a tensor with one value per channel and spatial position. The left panel shows one layer's gate map averaged over channels; the right panel shows the distribution of all gate values in that layer.
*Why average over channels.* A 64-channel 16×16 gate tensor is too much to display directly. Averaging over channels gives a readable spatial map. The histogram preserves the full distribution.

== Cell 23: Gate map and histogram for one image

*What this cell does.* Runs one test image through the trained model, selects the last block of stage two, and plots that block's gate map and gate histogram.

```
  1 | image, label = test_set[0]
  2 | with torch.no_grad():
  3 |     highway(image.unsqueeze(0).to(device))
  4 | 
  5 | layer = highway.break2 - 1  # last block of stage two, 16x16 gates
  6 | block = highway.blocks[layer]
  7 | assert isinstance(block, Highway)
  8 | gate_map = block.last_gate[0].mean(dim=0).cpu()  # average over channels
  9 | gates = block.last_gate.cpu()
 10 | 
 11 | fig, (ax_map, ax_hist) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 1]})
 12 | im = ax_map.imshow(gate_map, cmap="viridis", vmin=0, vmax=1)
 13 | ax_map.set(
 14 |     xlabel="width",
 15 |     ylabel="height",
 16 |     title=f"Layer {layer + 1} gate map for one image ({classes[label]})",
 17 | )
 18 | fig.colorbar(im, ax=ax_map, label="mean T over channels")
 19 | 
 20 | ax_hist.hist(gates.flatten(), bins=50, range=(0, 1), color="tab:orange")
 21 | ax_hist.set(xlabel="T", ylabel="values", title="Gate distribution")
 22 | plt.tight_layout()
 23 | plt.show()
```

*Line by line:*

- *Line 1:* `image, label = test_set[0]` gets the first test image, already normalized, and its integer label.
- *Lines 2-3:* Runs the image through the model with gradients disabled so every block stores `last_gate`.
- *Line 5:* `layer = highway.break2 - 1` selects the last block of stage two. `break2` is 12, so this is block index 11, the 12th block. It has 16×16 spatial gates.
- *Line 6:* Gets that block.
- *Line 7:* `assert isinstance(block, Highway)` documents and narrows the type.
- *Line 8:* `gate_map = block.last_gate[0].mean(dim=0)` selects the single image and averages over channels, giving a 16×16 map.
- *Line 9:* `gates = block.last_gate.cpu()` keeps the full gate tensor for the histogram.
- *Line 11:* Creates two panels, with the map three times wider than the histogram.
- *Line 12:* Displays the gate map with a fixed 0 to 1 scale.
- *Lines 13-17:* Labels the axes and titles the panel with the true class.
- *Line 18:* Adds the color bar to the map panel with the label `mean T over channels`.
- *Line 20:* Plots a histogram of all gate values in that block.
- *Line 21:* Labels the histogram axes.
- *Lines 22-23:* Lays out and displays the figure.

== Cell 24: Class-specific routing heading

*Role of this cell.* Explains the paper's Figure 3 idea for its 50-layer networks: compare each class's mean gate activity with the overall average. Red cells mean a class transforms more than usual in that layer; blue cells mean it carries more than usual.
*Why this matters.* It tests whether the gates are input-dependent or just a fixed layer-wise pattern.

== Cell 25: Class-routing heatmap

*What this cell does.* Subtracts the overall per-layer gate activity from each class's per-layer activity and displays the result as a diverging heatmap.

```
  1 | diff = (class_activity - overall_activity).numpy()
  2 | limit = max(abs(diff).max(), 0.01)
  3 | 
  4 | plt.figure(figsize=(9, 4))
  5 | plt.imshow(diff, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
  6 | plt.colorbar(label="class mean T - overall mean T")
  7 | plt.yticks(range(len(classes)), classes)
  8 | plt.xlabel("layer")
  9 | plt.title("Which layers each class uses more or less than average")
 10 | plt.tight_layout()
 11 | plt.show()
```

*Line by line:*

- *Line 1:* `diff = (class_activity - overall_activity).numpy()` computes the difference. `class_activity` has shape `(10, 19)` and `overall_activity` has shape `(19,)`, so broadcasting subtracts the overall value from each class.
- *Line 2:* `limit = max(abs(diff).max(), 0.01)` finds the largest absolute difference and uses at least 0.01 so the color scale never has zero width.
- *Line 4:* Creates the figure.
- *Line 5:* `imshow` displays the difference. `coolwarm` is a diverging colormap, and symmetric limits keep zero at the center color.
- *Line 6:* Adds a color bar labeled with the difference in mean gate activity.
- *Line 7:* Puts the 10 class names on the y-axis.
- *Line 8:* Labels the x-axis as layer.
- *Line 9:* Adds the title.
- *Lines 10-11:* Lays out and displays the figure.

== Cell 26: Information highway heading

*Role of this cell.* Explains the cosine-similarity plot. A highway layer can pass its input through almost unchanged. Values close to 1 mean the block copied its input; values near 0 mean it changed the representation substantially.

== Cell 27: Layer-to-layer input/output similarity

*What this cell does.* Runs the fixed probe batch through the model manually, computes the cosine similarity between each block's output and its input, and plots the average similarity per block.

```
  1 | @torch.no_grad()
  2 | def layer_similarity(model, x):
  3 |     model.eval()
  4 |     h = model.stem(x)
  5 |     sims = []
  6 |     for i, block in enumerate(model.blocks):
  7 |         if i == model.break1:
  8 |             h = model.down1(h)
  9 |         if i == model.break2:
 10 |             h = model.down2(h)
 11 |         h_new = block(h)
 12 |         sims.append(F.cosine_similarity(h_new, h, dim=1).mean().item())
 13 |         h = h_new
 14 |     return sims
 15 | 
 16 | 
 17 | plt.figure(figsize=(8, 3.5))
 18 | plt.plot(range(1, len(highway.blocks) + 1), layer_similarity(highway, probe_x), "o-")
 19 | plt.axhline(0, color="gray", linewidth=1)
 20 | plt.ylim(-0.2, 1)
 21 | plt.xlabel("layer")
 22 | plt.ylabel("cosine similarity(output, input)")
 23 | plt.title("How much each layer changes its input")
 24 | plt.grid(alpha=0.3)
 25 | plt.tight_layout()
 26 | plt.show()
```

*Line by line:*

- *Line 1:* `@torch.no_grad()` disables gradients.
- *Line 2:* `def layer_similarity(model, x)` defines the helper.
- *Line 3:* Switches to evaluation mode.
- *Line 4:* Runs the stem to get the first block input.
- *Line 5:* `sims = []` stores one similarity per block.
- *Line 6:* Loops over blocks by index.
- *Lines 7-8:* Before block 6, apply `down1`, matching the model's forward pass.
- *Lines 9-10:* Before block 12, apply `down2`.
- *Line 11:* Applies the block.
- *Line 12:* `F.cosine_similarity(h_new, h, dim=1)` computes similarity along the channel dimension, giving a map per spatial position. `.mean().item()` averages over batch and space and stores a Python float.
- *Line 13:* Updates the input for the next block.
- *Line 14:* Returns one similarity per block.
- *Line 17:* Creates the figure.
- *Line 18:* Plots highway similarity per layer with circle markers.
- *Line 19:* Draws a zero reference line.
- *Line 20:* Fixes the y-range to -0.2 to 1.
- *Line 21:* Labels the x-axis as layer.
- *Line 22:* Labels the y-axis as cosine similarity between output and input.
- *Line 23:* Adds the title.
- *Line 24:* Adds a grid.
- *Lines 25-26:* Lays out and displays the figure.

== Cell 28: Lesioning heading

*Role of this cell.* Explains the paper's Figure 4 lesioning experiment for its 50-layer networks. Close all transform gates of one layer at a time, forcing that layer to copy its input, and measure the test-accuracy drop. A drop near zero means the layer is barely used; a large drop means it is important.
*Caveat.* With 10,000 test images, differences below about 0.5 percentage points are close to noise, so small bars should not be over-interpreted.

== Cell 29: Layer-lesion experiment

*What this cell does.* Measures baseline test accuracy, closes each highway block's gates one at a time, records the accuracy drop, restores the block, and plots the drops.

```
  1 | baseline_acc = accuracy(highway, test_loader)
  2 | drops = []
  3 | for block in highway.blocks:
  4 |     assert isinstance(block, Highway)
  5 |     block.gate_off = True
  6 |     drops.append(100 * (baseline_acc - accuracy(highway, test_loader)))
  7 |     block.gate_off = False
  8 | 
  9 | plt.figure(figsize=(8, 3.5))
 10 | plt.bar(range(1, len(highway.blocks) + 1), drops, color="tab:red")
 11 | plt.axhline(0, color="gray", linewidth=1)
 12 | plt.xlabel("layer with closed gates")
 13 | plt.ylabel("accuracy drop (percentage points)")
 14 | plt.title("Contribution of each highway layer")
 15 | plt.grid(alpha=0.3, axis="y")
 16 | plt.tight_layout()
 17 | plt.show()
```

*Line by line:*

- *Line 1:* `baseline_acc = accuracy(highway, test_loader)` measures accuracy with every gate active.
- *Line 2:* `drops = []` will store one drop per layer.
- *Line 3:* Loops over all highway blocks.
- *Line 4:* `assert isinstance(block, Highway)` documents and narrows the type.
- *Line 5:* `block.gate_off = True` forces that block's gate to zero, so the block returns its input.
- *Line 6:* Measures accuracy with the block lesioned and stores the drop in percentage points. `100 * (baseline - lesioned)` is positive when accuracy falls.
- *Line 7:* Restores the block before moving to the next one.
- *Line 9:* Creates the figure.
- *Line 10:* Bar chart of accuracy drop per lesioned layer.
- *Line 11:* Draws a zero reference line.
- *Line 12:* Labels the x-axis as the layer whose gates were closed.
- *Line 13:* Labels the y-axis as accuracy drop in percentage points.
- *Line 14:* Adds the title.
- *Line 15:* Adds a horizontal grid.
- *Lines 16-17:* Lays out and displays the figure.

== Cell 30: Feature space heading

*Role of this cell.* Explains the t-SNE plot. The global average pooled feature before the classifier is the network's learned representation. t-SNE compresses that 128-dimensional space to two dimensions. Well-separated clusters mean the representation is easier to classify.
*Caveat.* t-SNE is qualitative. Distances between clusters are not meaningful, and the plot can change with seed and perplexity.

== Cell 31: t-SNE of pooled features

*What this cell does.* Collects the 128-dimensional pooled feature for 2,000 test images, runs t-SNE, and scatters the 2D embedding colored by class.

```
  1 | @torch.no_grad()
  2 | def penultimate_features(model, loader, limit=2000):
  3 |     model.eval()
  4 |     feats, labels = [], []
  5 |     seen = 0
  6 |     for x, y in loader:
  7 |         h = model.stem(x.to(device))
  8 |         for i, block in enumerate(model.blocks):
  9 |             if i == model.break1:
 10 |                 h = model.down1(h)
 11 |             if i == model.break2:
 12 |                 h = model.down2(h)
 13 |             h = block(h)
 14 |         h = h.mean(dim=(2, 3))  # global average pooling
 15 |         feats.append(h.cpu())
 16 |         labels.append(y)
 17 |         seen += y.numel()
 18 |         if seen >= limit:
 19 |             break
 20 |     return torch.cat(feats)[:limit].numpy(), torch.cat(labels)[:limit].numpy()
 21 | 
 22 | 
 23 | def tsne_2d(feats):
 24 |     return TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(feats)
 25 | 
 26 | 
 27 | feats_highway, labels_highway = penultimate_features(highway, test_loader)
 28 | emb_highway = tsne_2d(feats_highway)
 29 | 
 30 | fig, ax = plt.subplots(figsize=(8, 5))
 31 | for c, name in enumerate(classes):
 32 |     mask = labels_highway == c
 33 |     ax.scatter(emb_highway[mask, 0], emb_highway[mask, 1], s=5, alpha=0.7, label=name)
 34 | ax.set(xticks=[], yticks=[])
 35 | 
 36 | handles, names = ax.get_legend_handles_labels()
 37 | fig.legend(handles, names, loc="lower center", ncol=5, fontsize=7, markerscale=2)
 38 | plt.suptitle("t-SNE of pooled features (2,000 test images)")
 39 | plt.tight_layout(rect=(0, 0.08, 1, 1))
 40 | plt.show()
```

*Line by line:*

- *Line 1:* `@torch.no_grad()` disables gradients.
- *Line 2:* `def penultimate_features(model, loader, limit=2000)` defines the feature collector.
- *Line 3:* Switches to evaluation mode.
- *Line 4:* Initializes feature and label lists.
- *Line 5:* `seen = 0` counts how many samples have been collected.
- *Line 6:* Loops over the test loader.
- *Line 7:* Runs the stem.
- *Line 8:* Loops over blocks by index.
- *Lines 9-10:* Before block 6, apply `down1`.
- *Lines 11-12:* Before block 12, apply `down2`.
- *Line 13:* Applies the block.
- *Line 14:* `h = h.mean(dim=(2, 3))` applies global average pooling, giving `(batch, 128)`.
- *Line 15:* Stores the features on CPU.
- *Line 16:* Stores the labels.
- *Line 17:* Adds the batch size to `seen`.
- *Lines 18-19:* Stops once at least 2,000 samples are collected.
- *Line 20:* Concatenates batches, trims to exactly 2,000, and converts to NumPy for scikit-learn.
- *Line 23:* `def tsne_2d(feats)` defines the t-SNE wrapper.
- *Line 24:* Runs t-SNE with two components, perplexity 30, PCA initialization, and the global seed. Perplexity controls the effective neighborhood size; 30 is standard for a few thousand points.
- *Line 27:* Collects highway features from the test set.
- *Line 28:* Computes the 2D embedding.
- *Line 30:* Creates one figure.
- *Line 31:* Loops over the 10 classes.
- *Line 32:* `mask = labels_highway == c` selects the points of class `c`.
- *Line 33:* Scatters those points with small, semi-transparent markers.
- *Line 34:* Removes axis ticks because t-SNE coordinates have no meaning.
- *Line 36:* Collects legend handles and class names from the axis.
- *Line 37:* Places a shared legend below the plot.
- *Line 38:* Adds the title.
- *Line 39:* Leaves space at the bottom for the legend.
- *Line 40:* Displays the figure.

== Cell 32: Predictions heading

*Role of this cell.* A heading for the prediction grid. It is a qualitative sanity check, not a quantitative evaluation.

== Cell 33: Prediction grid

*What this cell does.* Takes one test batch, keeps eight images, runs the highway model, and shows each image with its true label, predicted label, and confidence.

```
  1 | sample_iter = iter(test_loader)  # keep the iterator alive for a clean worker shutdown
  2 | images, labels = next(sample_iter)
  3 | images, labels = images[:8], labels[:8]  # one batch holds 512, show 8
  4 | with torch.no_grad():
  5 |     probs = highway(images.to(device)).softmax(dim=1).cpu()
  6 | labels = labels.tolist()
  7 | preds = probs.argmax(dim=1).tolist()
  8 | confs = probs.max(dim=1).values.tolist()
  9 | 
 10 | fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))
 11 | for ax, img, label, pred, conf in zip(axes.flat, images, labels, preds, confs, strict=True):
 12 |     rgb = (img * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]).clamp(0, 1)
 13 |     ax.imshow(rgb.permute(1, 2, 0))
 14 |     ax.set_title(f"true: {classes[label]}\npred: {classes[pred]} ({conf:.0%})", fontsize=9)
 15 |     ax.axis("off")
 16 | 
 17 | plt.tight_layout()
 18 | plt.show()
```

*Line by line:*

- *Line 1:* Creates a test iterator and keeps it in `sample_iter` so its workers shut down cleanly.
- *Line 2:* Gets the first test batch. The test loader does not shuffle, so this is always the same batch.
- *Line 3:* Keeps only the first eight images and labels.
- *Lines 4-5:* Runs the model without gradients, applies softmax to turn logits into probabilities, and moves the result to CPU.
- *Line 6:* Converts labels to Python integers.
- *Line 7:* `probs.argmax(dim=1).tolist()` gives the predicted class indices.
- *Line 8:* `probs.max(dim=1).values.tolist()` gives the confidence of each prediction.
- *Line 10:* Creates a 2×4 grid of axes.
- *Line 11:* Iterates over images, labels, predictions, and confidences. `strict=True` checks that all sequences have the same length.
- *Line 12:* Reverses normalization with `img * STD + MEAN`, broadcasting per channel, and clamps to 0 to 1 for display.
- *Line 13:* Converts from channels-first to channels-last for matplotlib and displays the image.
- *Line 14:* Adds a title with the true class, predicted class, and confidence as a percentage.
- *Line 15:* Hides the axes.
- *Lines 17-18:* Lays out and displays the grid.
