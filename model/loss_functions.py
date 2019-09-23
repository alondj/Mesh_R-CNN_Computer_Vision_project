from typing import Tuple, Optional, List

import torch
import torch.nn as nn
from torch import Tensor
from data import Batch
from utils import sample
from utils.time_decorator import time


# compute combined losses of the bacbone and GCN
def total_loss(ws: dict, model_output: dict, voxel_gts: Tensor, batch: Batch,
               train_backbone: bool,
               backbone_type: str) -> Tensor:
    v_loss = voxel_loss(model_output['voxels'], voxel_gts)

    if 'edge_index' in model_output:
        chamfer_loss, normal_loss, edge_loss = batched_mesh_loss(
            model_output['vertex_positions'],
            model_output['faces'],
            model_output['edge_index'],
            model_output['vertice_index'],
            model_output['face_index'],
            batch
        )
    else:
        chamfer_loss, normal_loss, edge_loss = 0, 0, 0

    # if we train the backbone just add the losses
    backbone_loss = 0
    if train_backbone:
        if backbone_type == 'ShapeNet':
            backbone_loss = model_output['backbone']
        else:
            backbone_loss = sum(model_output['backbone'].values())

    return backbone_loss*ws['b']+chamfer_loss*ws['c']+normal_loss*ws['n']+edge_loss*ws['e']+v_loss*ws['v']


def voxel_loss(voxel_prediction: Tensor, voxel_gts: Tensor) -> Tensor:
    # minimize binary cross entropy between predicted voxel occupancy probabilities and true voxel occupancies.
    loss = nn.functional.binary_cross_entropy(voxel_prediction, voxel_gts.float(),
                                              reduction='mean')
    return loss


def batched_mesh_loss(vertex_positions_pred: Tensor, mesh_faces_pred: Tensor, pred_adjacency: Tensor,
                      vertices_per_sample_pred: List[int], faces_per_sample_pred: List[int],
                      batch: Batch,
                      point_cloud_size: float = 1e3,
                      num_neighbours_for_normal_loss: int = 10) -> Tuple[Tensor, Tensor, Tensor]:

    chamfer, normal, edge = mesh_loss(vertex_positions_pred[0], mesh_faces_pred, pred_adjacency,
                                      vertices_per_sample_pred, faces_per_sample_pred,
                                      batch, point_cloud_size,
                                      num_neighbours_for_normal_loss)

    for idx, pos in enumerate(vertex_positions_pred[1:]):
        c, n, e = mesh_loss(pos, mesh_faces_pred, pred_adjacency,
                            vertices_per_sample_pred, faces_per_sample_pred,
                            batch, point_cloud_size,
                            num_neighbours_for_normal_loss)
        chamfer += c
        normal += n
        edge += e

    return chamfer, normal, edge


def mesh_loss(vertex_positions_pred: Tensor, mesh_faces_pred: Tensor, pred_adjacency: Tensor,
              vertices_per_sample_pred: List[int], faces_per_sample_pred: List[int],
              batch: Batch,
              point_cloud_size: float = 1e3,
              num_neighbours_for_normal_loss: int = 10) -> Tuple[Tensor, Tensor, Tensor]:

    # edge loss
    p2p_dist = batched_point2point_distance(vertex_positions_pred).squeeze(0)
    edge_loss = total_edge_length(p2p_dist, pred_adjacency)

    # sample normalized point clouds from the predictions and gts
    point_cloud_pred = batched_mesh_sampling(vertex_positions_pred, mesh_faces_pred,
                                             vertices_per_sample_pred, faces_per_sample_pred,
                                             point_cloud_size)
    pos_gts, faces_gt = batch.meshes
    vertice_index, face_index = batch.vertice_index, batch.face_index

    point_cloud_gt = batched_mesh_sampling(pos_gts, faces_gt,
                                           vertice_index, face_index,
                                           point_cloud_size)

    # find distance between the points
    p2p_dist = batched_point2point_distance(point_cloud_pred, point_cloud_gt)

    # chamfer distance
    chamfer_p, idx_p, chamfer_gt, idx_gt = batched_chamfer_distance(p2p_dist)
    chamfer_loss = (chamfer_p+chamfer_gt)/point_cloud_size

    # normal_distance
    normal_dist_p, normal_dist_gt = batched_normal_distance(point_cloud_pred, point_cloud_gt,
                                                            p2p_dist, idx_p, idx_gt,
                                                            k=num_neighbours_for_normal_loss)
    normal_loss = -(normal_dist_p + normal_dist_gt) / point_cloud_size

    return chamfer_loss, normal_loss, edge_loss

    # ------------------------------------------------------------------------------------------------------


#  can be vectorized if necessary
def batched_mesh_sampling(vertex_positions: Tensor, mesh_faces: Tensor,
                          vertices_per_sample: List[int], faces_per_sample: List[int],
                          num_points: float = 1e3) -> Tensor:
    ''' given vertex positions and mesh faces sample point clouds to be used in the loss functions
        vertices_per sample and faces_per sample specify how to split the input along dimention 0
    '''
    point_clouds = [sample(*mesh, num_points=num_points)for mesh in
                    zip(vertex_positions.split(vertices_per_sample), mesh_faces.split(faces_per_sample))]
    clouds = torch.stack(point_clouds)
    return clouds
# ------------------------------------------------------------------------------------------------------


def batched_chamfer_distance(p2p_distance: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    ''' compute the chamfer distance between 2 point clouds\n
        which is defined by the summed distance of each point in cloud A to it's nearest neighbour in cload B\n
        and vice versa
    '''
    mins, idx1 = torch.min(p2p_distance, 2)
    loss_1 = torch.sum(mins)
    mins, idx2 = torch.min(p2p_distance, 1)
    loss_2 = torch.sum(mins)
    return loss_1, idx1, loss_2, idx2


# ------------------------------------------------------------------------------------------------------

def batched_normal_distance(p: Tensor, pgt: Tensor, p2p_distance: Tensor, idx_p: Tensor, idx_gt: Tensor, k=4) -> Tuple[Tensor, Tensor]:
    ''' calculate the normal distance between point clouds p and pgt\n
        p2p_distance is a matrix where p2p_distance[i,j]=|pi-pj|^2\n
        k is the number of neighbours used in order to estimate the normal to each point
    '''
    b_size = p.shape[0]
    # batch x size_p x 3 , batch x size_pgt x 3
    p_normals = compute_normals(p, p2p_distance, k=k)
    pgt_normals = compute_normals(pgt, p2p_distance.transpose(2, 1), k=k)

    # batch x size_p x 3
    # expand the batch_idx and broadcast with idx_p of shape batch x size_p
    nn_normals = pgt_normals[torch.arange(b_size).view(-1, 1), idx_p]
    loss_0 = torch.mul(p_normals, nn_normals).sum(2).abs().sum()

    # batch x size_pgt x 3
    # expand the batch_idx and broadcast with idx_gt of shape batch x size_pgt
    nn_normals = p_normals[torch.arange(b_size).view(-1, 1), idx_gt]
    loss_1 = torch.mul(pgt_normals, nn_normals).sum(2).abs().sum()
    return loss_0, loss_1


def compute_normals(pt: Tensor, p2p_distance: Tensor, k: int = 10) -> Tensor:
    # for each point find closest k neighbourhood
    # for each neighbourhood find center of mass M
    # for each neighbourhood compute scatter matrix Si= Yi.t() where Yi is xi-M
    # use pca to find the vector with least correlativity aka corresponds to smallest eigen value
    # as it's the best approximation of the normal to the plane approximated by the neighbourhood

    b, p, d = pt.shape
    # pt batch x num_points x 3
    # p2p_distance batch x num_points x num_points

    # batch x num_points x K
    nn_idxs = p2p_distance.topk(k, dim=2, largest=False, sorted=False).indices
    # pts batch x num_points x 3
    # idx batch x num_points x K

    # batch x num_points x k x 3
    neighbourhoods = pt[torch.arange(b).view(-1, 1, 1), nn_idxs]

    # batch x num_points x 3
    Ms = neighbourhoods.mean(2)

    # broadcast Ms and compute scatter matrix
    # Y is batch x points x k 3
    Y = torch.sub(neighbourhoods, Ms.unsqueeze(2))
    # batch x points x 3 x 3
    S = torch.matmul(Y.transpose(-2, -1), Y)

    # find the normal as eigen vector with smallest eigen value
    # aka the eigen vector which most different than the approximated plane
    # it's faster to compute this part on the cpu (including data transfer)
    # batch x points x 3 , batch x points x 3 x 3
    eigen_values, eigen_vectors = torch.symeig(S.cpu(), eigenvectors=True)
    # b x points
    smallest_eigen_values = eigen_values.argmin(2)

    # batch x points x 3
    normals = eigen_vectors[torch.arange(b).view(b, 1),
                            torch.arange(p).view(1, p).expand(b, p),
                            smallest_eigen_values]

    return normals.to(S.device)
# ------------------------------------------------------------------------------------------------------


#  normalizes as a whole and not per sample
def total_edge_length(p2p_distance: Tensor, vertex_adjacency: Tensor,) -> Tensor:
    ''' compute the edge loss as denoted by L(V,E) =1/|E| * ∑(v,v′)∈E ‖v−v′‖^2\n
        vertex_adjacency can be many adjacency matrices stacked together in block diagonal format
    '''
    # we mask only (v,v′)∈E
    masked_p2p_distance = p2p_distance[vertex_adjacency[0],
                                       vertex_adjacency[1]]

    # normalize by the number of edges
    normalize_factor = masked_p2p_distance.shape[0]

    # we count each edge twice so when we normalize it cancels out 2*s /2|E|
    loss = masked_p2p_distance.sum() / normalize_factor

    return loss


def batched_point2point_distance(pt0: Tensor, pt1: Optional[Tensor] = None) -> Tensor:
    ''' calculate the |pi - qj|^2 between the 2 given point clouds\n
        if pt1 is None calculate the point to point distance inside pt0\n
        the operation is batched and if a batch dim will be added if given 2d input
        returns matrix M such that M[i][j][k] = |pt0[i,j] - pt1[i,k]|^2
    '''
    if pt0.ndim == 2:
        pt0 = pt0.unsqueeze(0)

    if pt1 is None:
        xx = torch.bmm(pt0, pt0.transpose(2, 1))
        rx = xx.diagonal(dim1=1, dim2=2).unsqueeze(1).expand(xx.shape)
        p2p = rx.transpose(2, 1) + rx - 2*xx
        return p2p

    if pt1.ndim == 2:
        pt1 = pt1.unsqueeze(0)

    # X @ Y [i,j] = <Xi,Yj>

    xx = torch.bmm(pt0, pt0.transpose(2, 1))
    yy = torch.bmm(pt1, pt1.transpose(2, 1))
    zz = torch.bmm(pt0, pt1.transpose(2, 1))

    rx = xx.diagonal(dim1=1, dim2=2).unsqueeze(1).expand_as(zz.transpose(2, 1))

    ry = yy.diagonal(dim1=1, dim2=2).unsqueeze(1).expand_as(zz)
    P = (rx.transpose(2, 1) + ry - 2*zz)
    return P
