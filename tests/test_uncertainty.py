import numpy as np
import pytest
from src.uncertainty import predictive_entropy

def test_entropy_is_low_for_confident_prediction():
    confident = predictive_entropy(np.array([[0.999, 0.001]]))[0]
    uncertain = predictive_entropy(np.array([[0.5, 0.5]]))[0]
    assert confident < uncertain

def test_entropy_shape():
    p = np.array([[.5,.5],[.8,.2]])
    assert predictive_entropy(p).shape == (2,)
