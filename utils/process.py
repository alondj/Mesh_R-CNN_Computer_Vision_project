import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import adaptive_max_pool3d, interpolate


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


def resample_voxels(voxels: Tensor, N: int):
    """
    up/downsample a BxVxVxV voxel grid to a BxNxNxN grid
    """
    assert voxels.ndim == 4, "expects batched input of shape BxVxVxV"
    dtype = voxels.dtype
    M = voxels.shape[1]
    assert voxels.shape[1:] == torch.Size([M, M, M])

    if M > N:
        # downsample
        return adaptive_max_pool3d(voxels.to(torch.float32), N).to(dtype)
    elif M < N:
        # upsample
        return interpolate(voxels.to(torch.float32).unsqueeze(1), size=N).squeeze(1).to(dtype)

    return voxels
