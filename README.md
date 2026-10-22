# AQUA Self-Pruning Network - staged rebuild

This is a staged rebuild of sdm0p/tredence-aqua at source commit 6999a8605b85cbb6434936d291cf24f4465a2e59. Source is reused and verified in dependency order. These commits record the rebuild, not the original development history or an independent first implementation. See REBUILD.md.

Python/NumPy reverse-mode autodiff, MLP, Adam, gradual pruning and sparse-aware inference.

## Install and run

From the repository root with Python 3.10+:

```sh
python -m pip install -r requirements.txt
python -m pytest -q
python -m train.run
python -c "from train.run import train; m,h=train(sparsity=.9); print(h[-1])"
python evaluate_submission.py
```

The final command runs the complete 30-run sweep, independent validation/test evaluation, fair inference benchmarks, raw CSV generation, plots and RESULTS.md generation. It takes approximately a minute on the development machine. Timing varies.

## Implementation

- engine/: Tensor, reverse topology, broadcast VJPs, 2D matmul, reductions, ReLU, tanh and log-sum-exp cross-entropy.
- nn/: composable Linear/MLP, He initialization and mask-aware Adam.
- prune/: exact global ranking and sparse export.
- train/: seeded spiral training, EMA sensitivity, cubic ramp and recovery.
- tests/: operation finite differences, extreme loss, graph accumulation, real mask lifecycle, hard budget and sparse parity.
- results/: sweep.csv, history.csv, timings.csv, environment.json and plots.

Read [RESULTS.md](RESULTS.md) for actual measurements and [DESIGN.md](DESIGN.md) for derivations. Scikit-learn is not used; no neural-network or autodiff frameworks are used.

## Evidence and limits

Test accuracy uses separate data (seed 9), validation uses seed 8, training uses seed 7. Three paired seeds control initialization/order. Test data selects no checkpoints or hyperparameters. Weight sparsity excludes 67 dense biases. Integer budgets remove floor(target * 1184) connections.

Training is dense masked NumPy math. Exported inference visits active edges only; the dense benchmark also avoids autograd. Arithmetic savings do not guarantee latency savings. Synthetic data and three seeds do not establish production quality or statistical superiority. Automatic regrowth is not implemented; explicit revival is tested.

The original repository retains its history. This new repository records the staged migration, additional tests and a direct-leaf backward fix found during verification. AI assistance is disclosed in AI_USAGE.md.
