import torch
import numpy as np


def rotation(alpha: float, tensor=False):
    alpha = (np.pi * alpha)/180

    matrix = torch.Tensor([[1, 0, 0],
                           [0, np.cos(alpha), -np.sin(alpha)],
                           [0, np.sin(alpha), np.cos(alpha)]])

    if not tensor:
        return matrix.numpy()

    return matrix
