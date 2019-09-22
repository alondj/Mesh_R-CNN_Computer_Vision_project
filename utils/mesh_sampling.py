import torch
from torch import Tensor
from .process import normalize_mesh


def sample(vertex_positions: Tensor, mesh_faces: Tensor, num_points: float = 10e3) -> Tensor:
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

    XI2s = torch.rand(num_points, device=device)
    XI1sqrt = torch.rand(num_points, device=device).sqrt()
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
    # return normalized cloud
    point_cloud = normalize_mesh(point_cloud)

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
