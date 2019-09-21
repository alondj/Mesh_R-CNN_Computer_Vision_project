from itertools import repeat
from typing import List, Tuple
import torch
from torch import Tensor


# ------------------------------------------------------------------------------------------------------
def conv_output(h: int, w: int, kernel: int = 3, padding: int = 0, dilation: int = 1, stride: int = 1):
    '''calculates the feature map height and width given the convolution parameters
    '''
    kh, kw = _tuple(kernel)
    ph, pw = _tuple(padding)
    dh, dw = _tuple(dilation)
    sh, sw = _tuple(stride)
    return _dim(h, k=kh, p=ph, s=sh, d=dh), _dim(w, k=kw, p=pw, s=sw, d=dw)


def _dim(h: int, k: int, p: int, s: int, d: int):
    return int((h + 2 * p - d * (k - 1) - 1) / s) + 1


# ------------------------------------------------------------------------------------------------------
def convT_output(h: int, w: int, kernel: int = 3, padding: int = 0, dilation: int = 1, stride: int = 1,
                 output_padding: int = 0):
    '''calculates the feature map height and width given the transposed convolution parameters
    '''
    kh, kw = _tuple(kernel)
    ph, pw = _tuple(padding)
    dh, dw = _tuple(dilation)
    sh, sw = _tuple(stride)
    pouth, poutw = _tuple(output_padding)
    return _dimT(h, k=kh, p=ph, s=sh, d=dh, pout=pouth), _dimT(w, k=kw, p=pw, s=sw, d=dw, pout=poutw)


def _dimT(h: int, k: int, p: int, s: int, d: int, pout: int):
    return (h - 1) * s - 2 * p + d * (k - 1) + pout + 1


def _tuple(n):
    if isinstance(n, tuple):
        assert len(n) == 2
        return n
    return n, n


# ------------------------------------------------------------------------------------------------------
# utils to multiply sparse adjacency matrices with dense matrices
# note that we do not need the sparse values as they are binary
# based on torch-scatter

def aggregate_neighbours(index: Tensor, matrix: Tensor) -> Tensor:
    row, col = index
    m, n = matrix.shape
    out = matrix[col]
    src, out, index, dim = gen_scatter_params(out, row, dim=0, dim_size=m)
    return out.scatter_add_(dim, index, src)


def maybe_dim_size(index, dim_size=None):
    if dim_size is not None:
        return dim_size
    return index.max().item() + 1 if index.numel() > 0 else 0


def gen_scatter_params(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    dim = range(src.dim())[dim]  # Get real dim value.

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = list(repeat(1, src.dim()))
        index_size[dim] = src.size(dim)
        if index.numel() > 0:
            index = index.view(index_size).expand_as(src)
        else:  # PyTorch has a bug when view is used on zero-element tensors.
            index = src.new_empty(index_size, dtype=torch.long)

    # Broadcasting capabilties: Expand dimensions to match.
    if src.dim() != index.dim():
        raise ValueError(
            ('Number of dimensions of src and index tensor do not match, '
             'got {} and {}').format(src.dim(), index.dim()))

    expand_size = []
    for s, i in zip(src.size(), index.size()):
        expand_size += [-1 if s == i and s != 1 and i != 1 else max(i, s)]
    src = src.expand(expand_size)
    index = index.expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        out_size = list(src.size())
        dim_size = maybe_dim_size(index, dim_size)
        out_size[dim] = dim_size
        out = src.new_full(out_size, fill_value, device=src.device)

    return src, out, index, dim


# ------------------------------------------------------------------------------------------------------

# return a dummy deterministic tensor of given shape
def dummy(*dims):
    s = 1
    for d in dims:
        s *= d

    return torch.arange(s).float().reshape(*dims)


# ------------------------------------------------------------------------------------------------------
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def filter_pix3d_input(targets, proposals, pix3d_input):
    filtered_output = []
    for img in range(len(proposals)):
        the_box = targets[img]["boxes"]
        scores = jaccard(the_box, proposals[img])
        _, indices = torch.max(scores, 0)
        filtered_output.append(pix3d_input[indices[0]])
    filtered_output = torch.stack(filtered_output,dim=0)
    return filtered_output
