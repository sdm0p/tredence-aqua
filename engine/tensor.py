"""Float64 reverse-mode autodiff with broadcast-safe vector-Jacobian products."""
import numpy as np


def _unbroadcast(gradient, shape):
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op

    def __add__(self, other):
        return self._bin(other, lambda a,b: a+b, lambda g,a,b: (g,g))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self._bin(other, lambda a,b: a-b, lambda g,a,b: (g,-g))

    def __rsub__(self, other):
        return Tensor(other) - self

    def __mul__(self, other):
        return self._bin(other, lambda a,b: a*b, lambda g,a,b: (g*b,g*a))

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self._bin(other, lambda a,b: a/b, lambda g,a,b: (g/b,-g*a/(b*b)))

    def __neg__(self):
        return self * -1

    def _bin(self, other, forward, derivatives):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(forward(self.data, other.data),
                     self.requires_grad or other.requires_grad, (self, other))

        def backward():
            left, right = derivatives(out.grad, self.data, other.data)
            if self.requires_grad:
                self._add_grad(_unbroadcast(left, self.data.shape))
            if other.requires_grad:
                other._add_grad(_unbroadcast(right, other.data.shape))
        out._backward = backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError("matmul requires 2D operands")
        out = Tensor(self.data @ other.data,
                     self.requires_grad or other.requires_grad, (self, other))

        def backward():
            if self.requires_grad:
                self._add_grad(out.grad @ other.data.T)
            if other.requires_grad:
                other._add_grad(self.data.T @ out.grad)
        out._backward = backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     self.requires_grad, (self,))

        def backward():
            gradient = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                normalized = sorted(a % self.data.ndim for a in axes)
                for reduction_axis in normalized:
                    gradient = np.expand_dims(gradient, reduction_axis)
            if self.requires_grad:
                self._add_grad(np.ones_like(self.data) * gradient)
        out._backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            count = np.prod([self.data.shape[a] for a in axes])
        return self.sum(axis, keepdims) / count

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), self.requires_grad, (self,))
        def backward():
            if self.requires_grad:
                self._add_grad(out.grad * (self.data > 0))
        out._backward = backward
        return out

    def tanh(self):
        out = Tensor(np.tanh(self.data), self.requires_grad, (self,))
        def backward():
            if self.requires_grad:
                self._add_grad(out.grad * (1 - out.data**2))
        out._backward = backward
        return out

    def _add_grad(self, gradient):
        self.grad = gradient if self.grad is None else self.grad + gradient

    def backward(self, grad=None):
        seed = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=np.float64)
        if seed.shape != self.data.shape:
            raise ValueError("seed shape mismatch")
        topology, visited = [], set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for parent in node._prev:
                visit(parent)
            topology.append(node)
        visit(self)

        if not self._prev:
            self._add_grad(seed)
            return
        # Preserve accumulated leaf gradients, but discard old intermediate values.
        for node in topology:
            if node._prev:
                node.grad = None
        self.grad = seed
        for node in reversed(topology):
            node._backward()


def cross_entropy(logits, labels):
    values, labels = logits.data, np.asarray(labels)
    if (values.ndim != 2 or len(values) == 0 or labels.shape != (len(values),)
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0) or np.any(labels >= values.shape[1])):
        raise ValueError("invalid logits or labels")
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    rows = np.arange(len(labels))
    loss = (np.log(exponentials.sum(axis=1)) - shifted[rows, labels]).mean()
    out = Tensor(loss, logits.requires_grad, (logits,))

    def backward():
        if logits.requires_grad:
            gradient = probabilities.copy()
            gradient[rows, labels] -= 1
            logits._add_grad(gradient / len(labels) * out.grad)
    out._backward = backward
    return out
