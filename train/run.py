"""Seeded training with pre-update EMA sensitivity and cubic gradual pruning."""
import numpy as np
from nn import MLP, Adam
from engine import cross_entropy
from prune import prune


def spiral(n=900, seed=0):
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for class_id in range(3):
        count = n // 3
        angle = np.linspace(class_id * 4, (class_id + 1) * 4, count)
        angle += rng.normal(0, .2, count)
        radius = np.linspace(.2, 1, count)
        features.extend(np.c_[radius * np.sin(angle), radius * np.cos(angle)])
        labels.extend([class_id] * count)
    return np.asarray(features), np.asarray(labels)


def train(seed=0, epochs=80, sparsity=0, method="saliency"):
    if epochs < 1 or method not in ("saliency", "magnitude") or not 0 <= sparsity <= 1:
        raise ValueError("invalid training configuration")
    features, labels = spiral(seed=7)
    rng = np.random.default_rng(seed)
    model = MLP(rng)
    optimizer = Adam(model.params())
    order = np.arange(len(labels))
    history = []
    scores = [np.zeros_like(weight.data) for weight in model.weights()]

    for epoch in range(epochs):
        rng.shuffle(order)
        epoch_loss = 0.
        for indices in np.array_split(order, 10):
            loss = cross_entropy(model(features[indices]), labels[indices])
            loss.backward()
            epoch_loss += float(loss.data) * len(indices)
            for i, weight in enumerate(model.weights()):
                sensitivity = np.abs(weight.data * weight.grad)
                scores[i] = .9 * scores[i] + .1 * sensitivity
            optimizer.step()

        progress = np.clip(((epoch + 1) - epochs * .2) / (epochs * .5), 0, 1)
        target = sparsity * (1 - (1 - progress)**3)
        prune(model, target, method == "saliency", scores, optimizer=optimizer)
        prediction = model.predict(features)
        history.append((epoch_loss / len(labels), float((prediction == labels).mean())))
    return model, history


if __name__ == "__main__":
    model, history = train()
    print(history[-1])
