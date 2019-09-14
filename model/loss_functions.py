from typing import Tuple, Optional, List

import torch
import torch.nn as nn
from torch import Tensor


# compute combined losses of the bacbone and GCN
def total_loss(ws: dict, model_output: dict, voxel_gts: Tensor, pt_gts: Tensor,
               train_backbone: bool,
               backbone_type: str) -> Tensor:
    v_loss = voxel_loss(model_output['voxels'], voxel_gts)

    chamfer_loss, normal_loss, edge_loss = batched_mesh_loss(
        model_output['vertex_postions'],
        model_output['faces'],
        model_output['edge_index'],
        model_output['vertice_index'],
        model_output['face_index'],
        pt_gts
    )

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
    return nn.functional.binary_cross_entropy(voxel_prediction, voxel_gts, reduction='mean')


def batched_mesh_loss(vertex_positions_pred: Tensor, mesh_faces_pred: Tensor, pred_adjacency: Tensor,
                      vertices_per_sample_pred: List[int], faces_per_sample_pred: List[int],
                      pt_gts: Tensor,
                      point_cloud_size: float = 10e3,
                      num_neighbours_for_normal_loss: int = 10) -> Tuple[Tensor, Tensor, Tensor]:

    # edge loss
    p2p_dist = batched_point2point_distance(vertex_positions_pred).squeeze(0)
    edge_loss = total_edge_length(p2p_dist, pred_adjacency)

    # sample point clouds from the predictions
    point_cloud_pred = batched_mesh_sampling(vertex_positions_pred, mesh_faces_pred,
                                             vertices_per_sample_pred, faces_per_sample_pred,
                                             point_cloud_size)
    point_cloud_gt = pt_gts

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
                          num_points: float = 10e3) -> Tensor:
    ''' given vertex positions and mesh faces sample point clouds to be used in the loss functions
        vertices_per sample and faces_per sample specify how to split the input along dimention 0
    '''
    point_clouds = [mesh_sampling(*sample, num_points=num_points)for sample in
                    zip(vertex_positions.split(vertices_per_sample), mesh_faces.split(faces_per_sample))]

    return torch.stack(point_clouds)


def mesh_sampling(vertex_positions: Tensor, mesh_faces: Tensor, num_points: float = 10e3) -> Tensor:
    # described originally here https://arxiv.org/pdf/1901.11461.pdf
    # given a mesh sample a point cloud to be used in the loss functions
    f, d = mesh_faces.shape
    v, p = vertex_positions.shape
    num_points = int(num_points)
    areas = surface_areas(vertex_positions, mesh_faces)
    probas = areas/areas.sum()
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


def surface_areas(vertex_positions: Tensor, mesh_faces: Tensor) -> Tensor:
    # given triangle ABC the area is |AB x AC|/2
    p, d = vertex_positions.shape
    f, nv = mesh_faces.shape

    # f x nv x d
    mesh_points = vertex_positions[mesh_faces]

    # f x d
    ABs = mesh_points[:, 1]-mesh_points[:, 0]
    ACs = mesh_points[:, 2]-mesh_points[:, 0]

    # f x d
    # AB x AC
    norm_vecs = ABs.cross(ACs, dim=1)

    # |AB x AC|/2
    areas = norm_vecs.norm(p=2, dim=1) / 2

    return areas


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
def batched_normal_distance(p: Tensor, pgt: Tensor, p2p_distance: Tensor, idx_p: Tensor, idx_gt: Tensor, k=10) -> Tuple[Tensor, Tensor]:
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
    # batch x points x 3 , batch x points x 3 x 3
    eigen_values, eigen_vectors = torch.symeig(S, eigenvectors=True)
    # b x points
    smallest_eigen_values = eigen_values.argmin(2)

    # batch x points x 3
    normals = eigen_vectors[torch.arange(b).view(b, 1),
                            torch.arange(p).view(1, p).expand(b, p),
                            smallest_eigen_values]

    return normals


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
    return masked_p2p_distance.sum() / normalize_factor


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
        return rx.transpose(2, 1) + rx - 2*xx

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