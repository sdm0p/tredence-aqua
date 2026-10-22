# Technical defense guide

AI assistance is permitted, but the candidate must inspect, run and understand the code. Be candid about assistance.

## Derivations

1. Y[n,d]=X[n,d]+b[d] implies db[d]=sum_n dY[n,d]. Why is another average wrong?
2. L=x*x+x gives 2*x+1. Why accumulate before parent propagation?
3. Derive cross-entropy gradient (p-one_hot(y))/N and invariance to a common logit shift.
4. Derive U=M*W; distinguish dL/dW from dL/dU.
5. Derive removal sensitivity |w*g| and explain interaction/first-order limitations.

## Optimizer trap

Trace one coordinate through active updates, prune, inactive steps and revival. Track weight, gradient, moments and coordinate age. Explain stale momentum and why fresh moments with an old bias-correction clock do not match fresh Adam. Demonstrate the regression test.

## Questions requiring actual results

What are per-seed accuracies at 90% for both methods? What is the paired mean difference and variability? Does evidence support superiority? How many MACs were saved, and did latency improve? What biases remain? Could global pruning disconnect a layer? Why warm-up/ramp/recovery? What does a synthetic task prevent you claiming?

## Live exercises

Implement sigmoid and a numerical-gradient test. Change score aggregation and explain its meaning. Add structured pruning and correct MAC accounting. Design a separate regrowth probe without silently using a straight-through estimator.

## Serving

Explain immutable artifacts, export, batching, tenant authorization/isolation, backpressure, p99 latency, canary evaluation and rollback. Local microbenchmarks do not establish thousands of requests per second.

Implementation map: engine/tensor.py implements VJPs and stable loss; nn/mlp.py implements mask transitions and coordinate ages; train/run.py records pre-update EMA scores and the cubic ramp; prune/export.py executes frozen active edges. tests/test_repairs.py checks the reproduced failures and true revival lifecycle. RESULTS.md provides measured answers backed by results/sweep.csv and results/timings.csv.
