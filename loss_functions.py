from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

# TODO maybe later we will use custom kernels for all of these ops


# losses of of mesh prediction network

def voxel_loss(voxel_prediction: Tensor, voxel_gts: Tensor):
    # minimize binary cross entropy between predicted voxel occupancy probabilities and true voxel occupancies.
    return nn.functional.binary_cross_entropy(voxel_prediction, voxel_gts, reduction='mean')

# ------------------------------------------------------------------------------------------------------


def mesh_sampling(vertex_positions: Tensor, mesh_faces: Tensor, num_points: int) -> Tensor:
    # described originally here https://arxiv.org/pdf/1901.11461.pdf
    # given a mesh sample a point cloud to be used in the loss functions

    # first find probabilities of choosing a face denoted by area of the face / total surface area
    probas = face_probas(vertex_positions, mesh_faces)

    device = vertex_positions.device

    face_indices = torch.multinomial(probas, num_points, replacement=True)

    # num_points x 3 x 3
    chosen_faces = vertex_positions[mesh_faces[face_indices]]

    XI2s = torch.randn(num_points, device=device)
    XI1sqrt = torch.randn(num_points, device=device).sqrt()

    # the weights for the face sampling
    w0 = 1.0 - XI1sqrt
    w1 = torch.mul(1-XI2s, XI1sqrt)
    w2 = torch.mul(XI2s, XI1sqrt)

    # num_points x 1 x 3
    ws = torch.stack([w0, w1, w2], dim=1).unsqueeze(1)

    # broadcat ws and multiply
    # the result is p=∑wi*vi where each wi multiplyes a different row in chosen_faces
    # we then sum accross the vertice dimention to recieve the sampled vertices
    point_cloud = torch.mul(chosen_faces, ws.transpose(1, 2)).sum(1)

    return point_cloud


def face_probas(vertex_positions: Tensor, mesh_faces: Tensor) -> Tensor:
    # given triangle ABC the area is |AB x AC|/2
    mesh_points = vertex_positions[mesh_faces]

    ABs = mesh_points[:, 1]-mesh_points[:0]
    ACs = mesh_points[:, 2]-mesh_points[:0]

    norm_vecs = ABs.cross(ACs, dim=1)

    # we can use the rectangle surface area (|AB x AC|) as the 2 factor will cancel out
    surface_areas = norm_vecs.mm(norm_vecs.t()).diagonal().sqrt()

    probas = surface_areas / surface_areas.sum()

    return probas


# ------------------------------------------------------------------------------------------------------

def chamfer_distance(pt0: Tensor, pt1: Tensor) -> Tuple[Tensor, Tensor]:
    ''' compute the chamfer distance between 2 point clouds\n
        which is defined by the summed distance of each point in cloud A to it's nearest neighbour in cload B\n
        and vice versa
    '''
    # batched point to point distance computation
    xx = torch.bmm(pt0, pt0.transpose(2, 1))
    yy = torch.bmm(pt1, pt1.transpose(2, 1))
    xy = torch.bmm(pt0, pt1.transpose(2, 1))
    rx = torch.diagonal(xx, dim1=-2, dim2=-1)
    rx = rx.unsqueeze(1).expand_as(xy.transpose(2, 1))
    ry = torch.diagonal(yy, dim1=-2, dim2=-1).unsqueeze(1).expand_as(xy)
    p2p_dist = (rx.transpose(2, 1) + ry - 2*xy)

    mins, _ = torch.min(p2p_dist, 1)
    loss_1 = torch.sum(mins)
    mins, _ = torch.min(p2p_dist, 2)
    loss_2 = torch.sum(mins)

    return loss_1, loss_2


def normal_distance_between_point_clouds():
    pass


def edge_loss(vertex_positions: Tensor, vertex_adjacency: Tensor) -> Tensor:
    ''' compute the edge loss as denoted by L(V,E) =1/|E| * ∑(v,v′)∈E ‖v−v′‖^2\n
        vertex_positions can be many vertices stacked along the vertix dimention\n
        vertex_adjacency can be many adjacency matrices stacked together in block diagonal format
    '''
    # xx[i,j] = pi dot pj
    xx = vertex_positions.mm(vertex_positions.t())

    rx = torch.diagonal(xx).expand_as(xx.t())

    dist_matrix = rx.t() + rx - 2*xx

    # we mask only (v,v′)∈E
    masked_dist_matrix = torch.mul(vertex_adjacency, dist_matrix)

    # normalize by the number of edges
    normalize_factor = torch.nonzero(masked_dist_matrix).shape[0]

    # we count each edge twice so when we normalize it cancels out 2*s /2|E|
    return masked_dist_matrix.sum() / normalize_factor


# losses of the backbone network


def mask_loss():
    pass


def box_loss():
    pass


if __name__ == "__main__":

    preds = torch.ones(10, 100, 3)
    gts = torch.ones(10, 100, 3)*2

    print(chamfer_distance(preds.cuda(), gts.cuda()))
