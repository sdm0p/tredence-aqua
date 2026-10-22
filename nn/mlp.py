"""Composable ReLU MLP and Adam with coordinate-local mask state."""
import numpy as np
from engine import Tensor


class Linear:
    def __init__(self, inp, out, rng):
        self.w = Tensor(rng.normal(0, np.sqrt(2 / inp), (inp, out)), True)
        self.b = Tensor(np.zeros(out), True)
        self.mask = np.ones((inp, out), bool)

    def __call__(self, x):
        return x @ (self.w * self.mask) + self.b

    def params(self):
        return [self.w, self.b]


class MLP:
    def __init__(self, rng, d=2, h=32, k=3):
        self.layers = [Linear(d, h, rng), Linear(h, h, rng), Linear(h, k, rng)]

    def __call__(self, x):
        z = Tensor(x)
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i < len(self.layers) - 1:
                z = z.relu()
        return z

    def params(self):
        return [param for layer in self.layers for param in layer.params()]

    def weights(self):
        return [layer.w for layer in self.layers]

    def predict(self, x):
        return np.argmax(self(x).data, 1)

    def dense_forward(self, x):
        """Inference without constructing an autodiff graph."""
        z = np.asarray(x)
        for i, layer in enumerate(self.layers):
            z = z @ (layer.w.data * layer.mask) + layer.b.data
            if i < len(self.layers) - 1:
                z = np.maximum(z, 0)
        return z


class Adam:
    def __init__(self, params, lr=2e-3):
        self.p = list(params)
        self.lr = lr
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.p]
        self.v = [np.zeros_like(p.data) for p in self.p]
        self.age = [np.zeros_like(p.data, dtype=int) for p in self.p]
        self.masks = [np.ones_like(p.data, dtype=bool) for p in self.p]

    def set_mask(self, param, mask, revival_value=0.):
        """Reset only changed/inactive coordinates; keep other state intact."""
        i = self.p.index(param)
        mask = np.asarray(mask)
        if mask.shape != param.data.shape or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("invalid mask")
        mask = mask.astype(bool)
        changed = mask != self.masks[i]
        param.data[changed & mask] = revival_value
        param.data[~mask] = 0
        reset = changed | ~mask
        for state in (self.m[i], self.v[i], self.age[i]):
            state[reset] = 0
        if param.grad is not None:
            param.grad[reset] = 0
        self.masks[i] = mask.copy()

    def step(self):
        self.t += 1  # Diagnostic batch count; bias correction uses coordinate age.
        for i, (param, first, second) in enumerate(zip(self.p, self.m, self.v)):
            if param.grad is None:
                continue
            mask = self.masks[i]
            gradient = np.where(mask, param.grad, 0)
            self.age[i][mask] += 1
            first[:] = np.where(mask, .9 * first + .1 * gradient, 0)
            second[:] = np.where(mask, .999 * second + .001 * gradient**2, 0)
            age = np.maximum(self.age[i], 1)
            corrected_first = first / (1 - .9**age)
            corrected_second = second / (1 - .999**age)
            param.data -= self.lr * corrected_first / (np.sqrt(corrected_second) + 1e-8)
            param.data[~mask] = 0
            param.grad = None
