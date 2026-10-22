"""Monotone global pruning with deterministic ties and a hard sparsity floor."""
import numpy as np
from .export import SparseMLP


def prune(model, target, saliency=True, grad_snapshots=None, optimizer=None):
    if not 0 <= target <= 1:
        raise ValueError("invalid target")
    weights = model.weights()
    total = sum(weight.data.size for weight in weights)
    # Round removed connections up: achieved sparsity must be at least target.
    remove = int(np.ceil(target * total))
    keep = total - remove
    scores = []
    for i, weight in enumerate(weights):
        gradient = grad_snapshots[i] if grad_snapshots is not None else weight.grad
        if saliency and gradient is None:
            raise ValueError("saliency needs gradients or precomputed EMA scores")
        # Training supplies EMA scores; direct callers may supply current gradients.
        if saliency:
            score = gradient if grad_snapshots is not None else np.abs(weight.data * gradient)
        else:
            score = np.abs(weight.data)
        scores.append(np.where(model.layers[i].mask, score, -np.inf).ravel())

    flat = np.concatenate(scores)
    active = min(keep, sum(layer.mask.sum() for layer in model.layers))
    indices = np.argsort(-flat, kind="stable")[:active]
    selected = np.zeros(total, bool)
    selected[indices] = True
    position = 0
    for layer, weight in zip(model.layers, weights):
        size = weight.data.size
        layer.mask = selected[position:position + size].reshape(weight.data.shape)
        position += size
        weight.data *= layer.mask
        if weight.grad is not None:
            weight.grad *= layer.mask
        if optimizer is not None:
            optimizer.set_mask(weight, layer.mask)
    return 1 - sum(layer.mask.sum() for layer in model.layers) / total


def sparse_forward(model, x):
    """Convenience path; use a persistent SparseMLP export for benchmarking."""
    sparse = SparseMLP(model)
    return sparse(x), sparse.active * len(x)
