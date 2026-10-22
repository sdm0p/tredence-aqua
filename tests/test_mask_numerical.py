"""Finite differences verify latent masked derivatives, not surrogate gradients."""
import numpy as np
from engine import Tensor
from nn import MLP, Adam
from prune import prune


def test_fixed_mask_finite_differences_every_coordinate():
    values = np.array([[.4, -.7], [1.2, .5]])
    mask = np.array([[1., 0.], [0., 1.]])
    features = np.array([[1., .3], [-.2, .8]])
    weights = Tensor(values.copy(), True)
    output = (Tensor(features) @ (weights * mask)).tanh().sum()
    output.backward()

    def objective():
        return np.tanh(features @ (values * mask)).sum()

    for index in np.ndindex(values.shape):
        old = values[index]
        values[index] = old + 1e-6
        plus = objective()
        values[index] = old - 1e-6
        minus = objective()
        values[index] = old
        numerical = (plus - minus) / 2e-6
        assert np.allclose(weights.grad[index], numerical, atol=1e-6)
        if mask[index] == 0:
            assert numerical == weights.grad[index] == 0


def test_budget_is_at_least_target_for_fractional_connection():
    model = MLP(np.random.default_rng(0))
    optimizer = Adam(model.params())
    prune(model, .9, saliency=False, optimizer=optimizer)
    total = sum(layer.mask.size for layer in model.layers)
    removed = sum(np.count_nonzero(~layer.mask) for layer in model.layers)
    assert removed == 1066
    assert removed / total >= .9
    assert removed / total - .9 < 1 / total
