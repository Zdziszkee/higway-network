#set document(
  title: "Highway Network Notebook: Line-by-Line Explanation",
  author: "Jcode",
)
#set page(
  paper: "a4",
  margin: (x: 1.55cm, top: 1.7cm, bottom: 1.7cm),
  numbering: "1",
  header: align(right)[#text(size: 8pt, fill: luma(110))[Highway notebook explanation]],
)
#set text(size: 9.6pt, fill: luma(20))
#set par(justify: true, leading: 0.62em)
#show raw: set text(size: 7.1pt)
#show heading.where(level: 1): set text(size: 18pt)
#show heading.where(level: 2): set text(size: 13.5pt)
#show heading.where(level: 3): set text(size: 11pt)
#show heading: set block(above: 1.1em, below: 0.55em)
#set list(indent: 0.55em, body-indent: 0.45em, spacing: 0.42em)
#set table(inset: 4pt)

#let callout(body) = block(
  width: 100%,
  fill: rgb("#f4f7fb"),
  stroke: 0.6pt + rgb("#cfd8e3"),
  radius: 4pt,
  inset: 9pt,
  body,
)

#align(center)[
  #v(2.2cm)
  #text(size: 25pt, weight: "bold")[Highway Network Notebook]
  #v(0.35cm)
  #text(size: 15pt)[Line-by-Line Explanation]
  #v(0.8cm)
  #text(size: 10.5pt, fill: luma(80))[CIFAR-10 convolutional highway network]
  #v(0.25cm)
  #text(size: 10.5pt, fill: luma(80))[Depth-20 convention, 19 highway blocks, 2,714,826 parameters]
  #v(1.6cm)
  #block(
    width: 92%,
    fill: rgb("#f4f7fb"),
    stroke: 0.6pt + rgb("#cfd8e3"),
    radius: 4pt,
    inset: 12pt,
  )[
    *Purpose.* This document explains the notebook cell by cell and, for code cells, line by line.
    It answers two questions for every line: what the line does, and why it is written that way.
    It is meant as preparation for questions about the implementation, not as a replacement for the
    paper or the notebook.
  ]
  #v(1.2cm)
  #table(
    columns: (auto, 1fr),
    inset: 5pt,
    align: (left, left),
    [*Notebook*], [`highway_cifar10.ipynb`],
    [*Model*], [Convolutional highway network, 19 highway blocks (depth-20 hidden stack)],
    [*Hidden blocks*], [19, split `6/6/7` over three stages],
    [*Physical convolutions*], [41, including stem and downsampling],
    [*Parameters*], [2,714,826],
    [*Training*], [SGD, 25 epochs, batch 128, LR 0.1, decay 0.89],
    [*Runtime*], [About 13 to 15 minutes on an RTX 3060 Ti],
  )
]

#pagebreak()
#outline(title: [Contents], indent: auto)
#pagebreak()

= How to read this document

- The document follows the notebook in cell order. Each section starts with a heading that names the cell and its role.
- Markdown cells are explained paragraph by paragraph. Code cells show the full numbered source and then a line-by-line explanation.
- In the line-by-line lists, `*Line 12:*` refers to the line number printed in the code block, not to a line in this document.
- A few multi-line expressions are explained as one logical line, for example `*Lines 9-12:*`. In those cases the range names every physical line in the group.
- The phrase *why this value* marks a rationale for a hyperparameter or a design choice.
- At the end there are reference tables, likely questions, known inconsistencies, and a map back to the paper.

= Executive summary

== What the notebook does

The notebook implements a convolutional highway network for CIFAR-10 and studies its internal gates. It trains one model, not a plain baseline. The model uses a depth-20 hidden stack: one plain convolutional stem followed by 19 highway blocks. The paper's released 19-layer CIFAR-10 network instead has 14 highway layers plus plain convolutions. Each block computes

```
y = H(x) * T(x) + x * (1 - T(x))
```

where `H` and `T` are 3×3 convolutions with padding 1. The gate `T` is a sigmoid, and its bias starts at -2 so the gates begin mostly closed. A plain strided convolution changes channels and resolution between stages. Global average pooling and a linear layer produce the 10 class logits.

== What it is not

- It is not the paper's exact FitNet reproduction. The widths, block counts, and training length are compact choices for an 8 GB GPU.
- It does not include a plain baseline. The notebook can show that the highway model trains and how its gates behave, but it cannot by itself prove that highway networks beat plain networks at depth.
- It is not a paper-scale run. The authors' released config uses about 400 epochs; this uses 25, so accuracy and gate convergence are lower.

== Architecture at a glance

#table(
  columns: (auto, auto, 1fr),
  inset: 4pt,
  align: (left, left, left),
  table.header([*Part*], [*Shape*], [*Detail*]),
  [Stem], [`(B, 32, 32, 32)`], [Plain 3×3 convolution, 3 to 32 channels, ReLU.],
  [Stage 1], [`(B, 32, 32, 32)`], [6 highway blocks at 32 channels.],
  [Down 1], [`(B, 64, 16, 16)`], [Plain 3×3 convolution, stride 2, 32 to 64 channels.],
  [Stage 2], [`(B, 64, 16, 16)`], [6 highway blocks at 64 channels.],
  [Down 2], [`(B, 128, 8, 8)`], [Plain 3×3 convolution, stride 2, 64 to 128 channels.],
  [Stage 3], [`(B, 128, 8, 8)`], [7 highway blocks at 128 channels.],
  [Pool], [`(B, 128)`], [Global average pooling.],
  [Head], [`(B, 10)`], [Linear classifier, raw logits.],
)

#callout[
  *The single most important sentence.* A highway block learns a per-unit, per-input interpolation
  between a transformed input `H(x)` and the original input `x`. When the gate is closed, the block
  is the identity and gradients can pass through unchanged.
]

#callout[
  *The single most important caveat.* Twenty-five epochs is a short run. Gate plots may be noisier
  and less converged than in the paper's ~400-epoch experiments.
]

#include "cells_0_9.typ"
#include "cells_10_19.typ"
#include "cells_20_33.typ"
#include "appendix.typ"
