import numpy as np
from src.metrics import accuracy, brier_score, expected_calibration_error, selective_accuracy

def test_perfect_predictions():
    p = np.array([[0.9, 0.1], [0.1, 0.9]])
    y = np.array([0, 1])
    assert accuracy(p, y) == 1.0
    assert brier_score(p, y) < 0.03

def test_ece_is_nonnegative():
    p = np.array([[0.6, 0.4], [0.2, 0.8]])
    y = np.array([0, 0])
    assert expected_calibration_error(p, y) >= 0.0

def test_selective_accuracy_returns_requested_coverages():
    p = np.array([[.9,.1],[.4,.6],[.2,.8],[.7,.3]])
    y = np.array([0,1,1,1])
    u = np.array([.1,.2,.3,.9])
    out = selective_accuracy(p, y, u, coverages=(1.0,.5))
    assert set(out) == {"1.0", "0.5"}
