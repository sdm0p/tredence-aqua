"""Freeze active indices and values once, then execute without graph building."""
import numpy as np


class SparseMLP:
    def __init__(self, model):
        self.layers = []
        self.active = 0
        for layer in model.layers:
            columns = []
            for output_index in range(len(layer.b.data)):
                indices = np.flatnonzero(layer.mask[:, output_index])
                values = layer.w.data[indices, output_index].copy()
                columns.append((indices, values))
                self.active += len(indices)
            self.layers.append((columns, layer.b.data.copy()))

    def __call__(self, x):
        features = np.asarray(x)
        for i, (columns, bias) in enumerate(self.layers):
            output = np.empty((len(features), len(bias)))
            for j, (indices, values) in enumerate(columns):
                output[:, j] = features[:, indices] @ values + bias[j]
            if i < len(self.layers) - 1:
                features = np.maximum(output, 0)
            else:
                features = output
        return features
