# Rebuild verification

Run `python -m pytest -q` from the root. The engine, MLP/optimizer and pruning checks currently total 33 passing tests. Checks include both operands of numerical operations, CE finite differences, extreme logits, shared/repeated backward, direct leaf accumulation, reduction axes, multiaxis broadcasting, mask transitions, invalid masks, stable ties, no implicit regrowth, zero/full sparsity, sparse parity and learning integration. A masked-weight central-difference test checks every active and inactive coordinate. Fractional budgets are rounded up and tested to enforce at least the requested sparsity.

Run `python evaluate_submission.py` to regenerate all 30 paired held-out runs, raw histories/timings, plots and RESULTS.md. Timing is hardware-dependent. The rebuild adds verification to reused source; see REBUILD.md for provenance.
