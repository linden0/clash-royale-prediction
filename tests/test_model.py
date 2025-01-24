import torch
import torch.nn as nn
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model.model import CNN

def test_model_initialization():
    num_classes = 10
    model = CNN(num_classes=num_classes, in_channels=1)
    assert isinstance(model, nn.Module), "Model is not an instance of nn.Module"

def test_forward_pass():
    num_classes = 10
    model = CNN(num_classes=num_classes, in_channels=1)

    # Create dummy input: batch_size=4, time=32, n_mfcc=32
    dummy_input = torch.randn(4, 32, 32)  # (batch_size, time, n_mfcc)

    # Forward pass
    output = model(dummy_input)

    # Check output shape
    assert output.shape == (4, num_classes), f"Expected output shape (4, {num_classes}), got {output.shape}"

def test_edge_cases():
    num_classes = 10
    model = CNN(num_classes=num_classes, in_channels=1)

    # Small batch size
    small_input = torch.randn(1, 16, 16)  # (batch_size=1, time=16, n_mfcc=16)
    output = model(small_input)
    assert output.shape == (1, num_classes), f"Expected output shape (1, {num_classes}), got {output.shape}"

    # Large batch size
    large_input = torch.randn(128, 32, 32)  # (batch_size=128, time=32, n_mfcc=32)
    output = model(large_input)
    assert output.shape == (128, num_classes), f"Expected output shape (128, {num_classes}), got {output.shape}"
