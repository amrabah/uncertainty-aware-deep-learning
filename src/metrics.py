import numpy as np

def accuracy(probs, y):
    return float((np.argmax(probs, axis=1) == y).mean())

def brier_score(probs, y):
    targets = np.eye(probs.shape[1])[y]
    return float(np.mean(np.sum((probs - targets) ** 2, axis=1)))

def expected_calibration_error(probs, y, bins: int = 10):
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)

def selective_accuracy(probs, y, uncertainty, coverages=(1.0, .9, .8, .7, .5)):
    order = np.argsort(uncertainty)
    out = {}
    n = len(y)
    for coverage in coverages:
        keep = max(1, int(round(n * coverage)))
        idx = order[:keep]
        out[str(coverage)] = accuracy(probs[idx], y[idx])
    return out
