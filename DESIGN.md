# Design and derivations

## Autodiff

Each node stores parents and a vector-Jacobian product. Reverse topology aggregates contributions before propagating to parents. Intermediate gradients reset for repeated backward; leaf gradients accumulate. Graph inputs and masks must not mutate between forward and backward. Broadcasting gradients sum extra leading dimensions and singleton operand dimensions. Matmul supports two-dimensional inputs; vector/batched matmul is outside its contract.

For C=A@B, dA=dC@B.T and dB=A.T@dC. ReLU uses zero derivative at zero; tanh derivative is 1-tanh(x)^2. Reductions expand removed axes before broadcasting derivatives.

## Stable loss

For shifted logits q=z-max(z), cross-entropy is log(sum(exp(q)))-q_y, averaged over samples. Gradient is (softmax(z)-one_hot(y))/N. We never take log of a tiny class probability. Extreme-logit regression tests cover the underflow failure in the prototype.

## Masks and Adam

Effective weight U=M*W implies dL/dW=M*dL/dU. Inactive latent weights get zero derivatives, although dL/dU can be nonzero. A regrowth probe would be a separate mechanism, not the true latent gradient.

Adam.set_mask clears weights, gradients, first/second moments and coordinate ages for removed coordinates. Revived coordinates receive an explicit value and fresh state. Untouched coordinates retain their ages/moments. Active coordinates advance their own age; bias correction uses that age. Resetting an entire global optimizer clock would disturb other coordinates and is not the implementation.

Masks change after all backward/optimizer work for an epoch. Fresh graphs are built for later batches. The main sweep is monotone; automatic regrowth is not evaluated.

## Initialization and importance

Weights use He normal variance 2/fan_in for ReLU; biases start at zero and stay dense. Removing w gives delta_w=-w; first-order delta_loss approximately -w*g. Absolute |w*g| estimates sensitivity, not guaranteed positive loss increase. A .9-decay EMA of absolute minibatch sensitivities reduces cancellation/noise. Scores are recorded before Adam changes weights. Interactions among removed edges and second-order effects are omitted.

Magnitude uses |w| with the same architecture, seeds and training schedule. Saliency need not outperform it.

## Budget and schedule

20% dense warm-up, cubic ramp until 70%, then fixed-mask recovery. For progress clipped to [0,1], s=target*(1-(1-progress)^3). Global stable ranking removes exactly floor(s*total_weights), never implicitly reviving masked edges. Integer achieved sparsity can be slightly below the requested fraction. Biases are excluded. Layer disconnection is allowed and disclosed.

## Measurement and scaling

Training is dense. SparseMLP exports fixed column indices/values once, then traverses only active connections. Dense reference uses plain NumPy. Timing excludes export and graph creation, includes five warm-ups and 20 samples per run. Weight MAC count is batch_size*active_weights; bias/activation cost remains dense. Environment and raw samples are committed.

Graph allocation, Python VJP dispatch and dense products are engine bottlenecks; sparse Python dispatch/gathers can dominate inference. Profile before tape compaction, buffer reuse or fused kernels. Structured pruning may yield smaller dense products; unstructured pruning needs suitable compiled kernels and hardware.

At scale, freeze/version architecture, preprocessing, masks and weights together. Validate parity/quality; batch compatible requests with bounded waiting, quotas and backpressure. Enforce tenant authorization/artifact isolation; never mutate shared inference models. Canary/rollback artifacts and measure queue depth, errors, p50/p95/p99 latency and cost/request. Local MAC measurements do not establish thousands of requests per second. Online topology changes belong in a separate evaluated training pipeline.
