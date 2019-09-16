import numpy as np
import torch


def normalize_mesh(vertices):
    '''
    normalize the vertices to reside inside a unit cube
    '''
    vertices = vertices-vertices.mean(0)
    if isinstance(vertices, np.ndarray):
        factor = np.sqrt((vertices@vertices.T).diagonal().max())
    else:
        factor = torch.sqrt(vertices.mm(vertices.T).diagonal().max())
    return vertices/factor
