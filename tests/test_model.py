import torch
from src.model import DropoutMLP

def test_model_output_shape():
    model = DropoutMLP(input_dim=30)
    x = torch.randn(8, 30)
    assert model(x).shape == (8, 2)
