# Requirements

Extracted from the actual uploaded `Pasted text.txt`, recovered from the linked chat on 22 October 2026. Private recruitment correspondence is excluded.

## Constraints

Python 3.10+ and NumPy for core computation; Matplotlib for plots. No PyTorch, TensorFlow, JAX, Keras, HuggingFace, external autodiff, neural-network, optimizer or pruning libraries. Scikit-learn is allowed only for loading data; this project plans synthetic spirals and no scikit-learn.

## Acceptance matrix

| ID | Required behavior | Evidence |
|---|---|---|
| A01 | Tensor with values, parents and gradients | source and graph tests |
| A02 | Addition, subtraction, multiplication, division, matmul | finite differences for both operands |
| A03 | Sum, mean, ReLU and tanh | numerical and shape tests |
| A04 | Stable fused softmax cross-entropy | extreme-logit and gradient tests |
| A05 | Reverse topology, accumulation and broadcasting | shared-parent and broadcast tests |
| A06 | Zero masked forward contributions and correct gradients | fixed-mask numerical check |
| A07 | Safe pruned-then-revived Adam state | lifecycle regression tests |
| A08 | Composable MLP and handwritten Adam/minibatching | source, optimizer tests and training |
| A09 | Stable learning and justified initialization | finite losses and learning curves |
| A10 | Derived importance beyond magnitude | Taylor derivation and saliency code |
| A11 | Gradual schedule and hard sparsity budget | exact-count tests and achieved sparsity |
| A12 | Truly zero removed weights | weight/mask/state invariants |
| A13 | 0/50/75/90/95 percent sweep | raw records and Pareto plot |
| A14 | Real sparse-aware cost evidence | parity, MAC counts and measured timings |
| A15 | Matched magnitude baseline and variance | paired three-seed results |
| A16 | Quantitative falsifiable claim | RESULTS.md supported by raw numbers |
| A17 | One-command tests/training/pruning/sweep | verified README instructions |
| A18 | Bottlenecks and multi-tenant serving | DESIGN.md |
| A19 | Explainable decisions and live extensions | DEFENSE.md and source/test navigation |

## Delivery and scope

Organize source under engine/, nn/, prune/, train/ and tests/. Commit plots, raw numbers, README and DESIGN. Preserve genuine incremental Git history; never manufacture retrospective history. Deliver a downloadable GitHub-ready archive. Publishing and recruiter submission are separate external actions.

Automatic regrowth is bonus scope. Explicit safe revival and its tests are planned without claiming automatic regrowth. Accuracy superiority and wall-clock speedup are measured outcomes, not promises.

Done means all required evidence exists, tests pass, experiments complete, claims agree with data, and archive contents are verified.
