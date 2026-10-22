# Reproducibility protocol

Canonical command: `python evaluate_submission.py`.

Three classes, 900 training samples (seed 7), 450 validation (seed 8), 900 independent test samples (seed 9). Splits share the same spiral task but independent noise. Architecture 2-32-32-3, float64, 80 epochs, ten minibatches, Adam learning rate .002. No checkpoint selection. Model/order seeds 0,1,2; each method gets matching initialization/order.

Targets 0/.5/.75/.9/.95, magnitude and .9-EMA Taylor sensitivity. Dense warm-up 20%, cubic ramp to 70%, recovery thereafter. Store achieved sparsity and layer counts. Test accuracy replaces the prototype's training accuracy; old results are superseded.

Results files contain raw run metrics, epoch mean losses, individual latency repetitions and environment metadata. Pareto x-axis is achieved sparsity, error bars sample SD with individual seeds. At 90%, benchmark both inference-only paths after five warm-ups with 20 repetitions each, excluding export. Bias/activation work is not counted as weight MACs.

No hyperparameters were selected using test metrics. Three seeds and one synthetic task support limited observations only. Timing varies by host and BLAS. Full production benchmarks and automatic regrowth are outside scope.
