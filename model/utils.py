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
def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    if torch.cuda.is_available():
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape).cuda()) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
    else:
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape)) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def filter_pix3d_input(detections, proposals, pix3d_input):
    max_prop_idx = [-1 for _ in range(len(proposals))]
    for img in range(len(proposals)):
        max_score = 0
        the_box = detections[img]["boxes"][0]
        for i in range(proposals[img].shape[0]):
            area = bbox_iou(the_box, proposals[img][i])
            if area > max_score:
                max_score = area
                max_prop_idx[img] = i
    filtered_output = []
    for img in range(len(proposals)):
        filtered_output.append(pix3d_input[img][max_prop_idx[img]])
    return filtered_output
