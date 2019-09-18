import numpy as np
import torch


def normalize_mesh(vertices):
    '''
    normalize the vertices to reside inside a unit cube
    '''
    vertices = vertices-vertices.mean(0)
    if isinstance(vertices, np.ndarray):
        if np.max(np.abs(vertices)) <= 1:
            return vertices
        factor = np.sqrt((vertices@vertices.T).diagonal().max())
    else:
        if vertices.abs().max() <= 1:
            return vertices
        factor = torch.sqrt(vertices.mm(vertices.T).diagonal().max())
    return vertices/factor
