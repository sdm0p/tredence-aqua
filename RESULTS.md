# Measured results

Reproduced with `python evaluate_submission.py`. Accuracy below is on independent test data (seed 9). Three paired initialization/order seeds; error bars are sample standard deviation.

| Target | Achieved | Magnitude mean ± SD | Saliency mean ± SD |
|---|---|---|---|
| 0% | 0.000% | 100.00 ± 0.00% | 100.00 ± 0.00% |
| 50% | 50.000% | 100.00 ± 0.00% | 100.00 ± 0.00% |
| 75% | 75.000% | 99.78 ± 0.00% | 99.85 ± 0.06% |
| 90% | 90.034% | 99.07 ± 0.06% | 99.33 ± 0.22% |
| 95% | 95.017% | 84.22 ± 8.39% | 97.96 ± 0.17% |

## Falsifiable claim

At target 90%, saliency averages 99.33% held-out accuracy versus 99.07% for magnitude across three paired seeds; paired difference 0.26 percentage points. This is a small synthetic-task observation, not statistical proof of general superiority.

## Inference cost

dense: median 0.1218 ms; 1065600 weight MACs per batch of 900. Bias and activation work remains dense.
sparse: median 0.5736 ms; 106200 weight MACs per batch of 900. Bias and activation work remains dense.

Maximum sparse/dense absolute error: 7.11e-15. Export and graph construction excluded from both timed inference paths. Raw repetitions: results/timings.csv. Timings are machine-dependent.

Training remains dense masked computation. No sparse training speedup or production capacity claim is made. Data noise varies independently across splits, but all splits share the same synthetic task. No automatic regrowth is evaluated.