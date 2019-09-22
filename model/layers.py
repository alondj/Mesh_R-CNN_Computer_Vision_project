import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple, Optional, List
from datetime import datetime
import math
from .utils import aggregate_neighbours
import numpy as np
# data representation for graphs:
# adjacency matrix: we just save occupied indices in coo format
# vertex features matrix: we concatenate over the vertex dim resulting in total_vertices x Num_features
# this representation allows us to batch graph operations
# mesh_faces: a matrix of relevant vertex indices
Point = Tuple[float, float, float]
Face = Tuple[Point, Point, Point]


class GraphConv(nn.Module):
    '''GraphConv(D1→D2) computes new features based on a linear projection of current features
       and a linear projection of adjacent vertices features as denoted by:
        f′i = ReLU(W0xfi +∑ j∈N(i) W1xfj)
    '''

    def __init__(self, in_features: int, out_features: int):
        super(GraphConv, self).__init__()
        self.w0 = torch.nn.Parameter(data=torch.empty(in_features, out_features),
                                     requires_grad=True)
        self.w1 = torch.nn.Parameter(data=torch.empty(in_features, out_features),
                                     requires_grad=True)

        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0/math.sqrt(self.w0.size(0))
        self.w0.data.uniform_(-bound, bound)
        self.w1.data.uniform_(-bound, bound)

    def forward(self, vertex_features: Tensor, vertex_adjacency: Tensor) -> Tensor:
        # note that vertex_features is the concatination of all feature matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_adjacency is a block diagonal matrix containing all adjacency matrices of the batch
        # as block on the diagonal and all off diagonal blocks are zero blocks

        # ∑VxIn @ InxOut => ∑VxOut
        w0_features = torch.mm(vertex_features, self.w0)

        # ∑VxIn @ InxOut => ∑VxOut
        w1_features = torch.mm(vertex_features, self.w1)

        # batch ∑j∈N(i)W1xfj
        # use the adjacency matrix as a mask
        # note a vertex is not connected to itself
        # ∑Vx∑V @ ∑VxOut => ∑VxOut
        neighbours = aggregate_neighbours(vertex_adjacency, w1_features)

        # aggregate features of neighbours
        new_features = w0_features + neighbours

        return self.relu(new_features)


class ResGraphConv(nn.Module):
    '''ResGraphConv(D1→D2) layers consists of two graph convolution, layers each preceeded by ReLU,
       and an additive skip connection with linear projection if input dimension D1 differs from output dimension D2
    '''

    def __init__(self, in_features: int, out_features: int):
        super(ResGraphConv, self).__init__()
        self.conv0 = GraphConv(in_features, out_features)
        self.conv1 = GraphConv(out_features, out_features)

        if in_features != out_features:
            projection = nn.Linear(in_features, out_features, bias=False)
        else:
            projection = nn.Identity()

        self.projection: nn.Module = projection

    def forward(self, vertex_features: Tensor, vertex_adjacency: Tensor) -> Tensor:
         # note that vertex_features is the concatination of all feature matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_adjacency is a block diagonal matrix containing all adjacency matrices of the batch
        # as block on the diagonal and all off diagonal blocks are zero blocks

        # ∑VxIn @ InxOut => ∑VxOut
        skip = self.projection(vertex_features)

        # ∑VxIn => ∑VxOut
        out = self.conv0(vertex_features, vertex_adjacency)
        # ∑VxOut=> ∑VxOut
        out = self.conv1(out, vertex_adjacency)

        return skip+out


class ResVertixRefineShapenet(nn.Module):
    ''' VertixRefine are cells which given an image feature maps and a 3D mesh\n
        outputs an updated 3D mesh and vertex features
    '''

    def __init__(self, use_input_features: bool = True,
                 num_features: int = 128, alignment_size: int = 3840,
                 ndims: int = 3):
        super(ResVertixRefineShapenet, self).__init__()

        self.vertAlign = VertexAlign()

        self.linear = nn.Linear(alignment_size, num_features, bias=False)

        in_channels = num_features + ndims
        if use_input_features:
            in_channels += num_features

        self.resGraphConv0 = ResGraphConv(in_channels, num_features)
        self.use_input_features = use_input_features

        self.resGraphConv1 = ResGraphConv(num_features, num_features)
        self.resGraphConv2 = ResGraphConv(num_features, num_features)
        self.graphConv = GraphConv(num_features, ndims)

        self.tanh = nn.Tanh()

    def forward(self, vertice_index: List[int], img_feature_maps: List[Tensor],
                vertex_adjacency: Tensor, vertex_positions: Tensor,
                image_sizes: List, vertex_features: Optional[Tensor] = None,
                meshes_index: List[int] = None) -> Tuple[Tensor, Tensor]:

        # note that vertex_features is the concatination of all feature matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_positions is the concatination of all position matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_adjacency is a block diagonal matrix containing all adjacency matrices of the batch
        # as block on the diagonal and all off diagonal blocks are zero blocks

        # note that the conv_feature are batched NxCxHxW

        # unless specified otherwise only one mesh per image
        if meshes_index is None:
            meshes_index = [1 for _ in image_sizes]

        # project the 3D mesh to the 2D feature planes and pool new features
        # ∑Vx3840
        aligned_vertices = self.vertAlign(img_feature_maps, vertex_positions,
                                          vertice_index, image_sizes, meshes_index)

        # ∑Vx128
        projected = self.linear(aligned_vertices)

        # ∑Vx259 if there are initial vertex_features
        # and ∑Vx131 otherwise
        to_concat = [vertex_positions, projected]
        if not (vertex_features is None):
            assert self.use_input_features
            to_concat = [vertex_features]+to_concat
        else:
            assert not self.use_input_features
        vertex_features = torch.cat(to_concat, dim=1)

        # transforms input features to 128 features
        new_features = self.resGraphConv0(vertex_features, vertex_adjacency)
        # ∑Vx128
        new_features = self.resGraphConv1(new_features, vertex_adjacency)
        new_features = self.resGraphConv2(new_features, vertex_adjacency)

        # new_positions is of shape ∑Vx3
        new_positions = self.graphConv(new_features, vertex_adjacency)
        new_positions = self.tanh(new_positions)
        new_positions = vertex_positions+new_positions

        return new_positions, new_features


class VertixRefineShapeNet(nn.Module):
    ''' VertixRefine are cells which given an image feature maps and a 3D mesh\n
        outputs an updated 3D mesh and vertex features
    '''

    def __init__(self, use_input_features: bool = True,
                 num_features: int = 128, alignment_size: int = 3840,
                 ndims: int = 3):
        super(VertixRefineShapeNet, self).__init__()
        self.vertAlign = VertexAlign()

        self.linear0 = nn.Linear(alignment_size, num_features, bias=False)

        in_channels = num_features + ndims
        if use_input_features:
            in_channels += num_features
        self.graphConv0 = GraphConv(in_channels, num_features)
        self.use_input_features = use_input_features

        self.graphConv1 = GraphConv(num_features+ndims,
                                    num_features)
        self.graphConv2 = GraphConv(num_features+ndims,
                                    num_features)
        self.linear1 = nn.Linear(num_features, ndims, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, vertice_index: List[int], img_feature_maps: List[Tensor],
                vertex_adjacency: Tensor, vertex_positions: Tensor,
                image_sizes: List, meshes_index: List[int] = None,
                vertex_features: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        # note that vertex_features is the concatination of all feature matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_positions is the concatination of all position matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_adjacency is a block diagonal matrix containing all adjacency matrices of the batch
        # as block on the diagonal and all off diagonal blocks are zero blocks

        # note that the conv_feature are batched NxCxHxW

        if meshes_index is None:
            meshes_index = [1 for _ in image_sizes]

        # project the 3D mesh to the 2D feature planes and pool new features
        # ∑Vx3840
        aligned_vertices = self.vertAlign(img_feature_maps, vertex_positions,
                                          vertice_index, image_sizes, meshes_index)
        # ∑Vx128
        projected = self.linear0(aligned_vertices)

        # ∑Vx259 if there are initial vertex_features
        # and ∑Vx131 otherwise
        to_concat = [vertex_positions, projected]
        if not (vertex_features is None):
            assert self.use_input_features
            to_concat = [vertex_features]+to_concat
        else:
            assert not self.use_input_features
        vertex_features = torch.cat(to_concat, dim=1)

        # transforms input features to 128 features
        new_features = self.graphConv0(vertex_features, vertex_adjacency)

        # ∑Vx131
        new_features = torch.cat([vertex_positions, new_features], dim=1)
        # ∑Vx128
        new_features = self.graphConv1(new_features, vertex_adjacency)
        # ∑Vx131
        new_features = torch.cat([vertex_positions, new_features], dim=1)
        # ∑Vx128
        new_features = self.graphConv2(new_features, vertex_adjacency)

        # ∑Vx3
        new_positions = self.linear1(new_features)
        new_positions = self.tanh(new_positions)
        new_positions = vertex_positions+new_positions

        return new_positions, new_features


class VertixRefinePix3D(nn.Module):
    ''' VertixRefine are cells which given an image feature maps and a 3D mesh\n
        outputs an updated 3D mesh and vertex features
    '''

    def __init__(self, use_input_features: bool = True,
                 num_features: int = 128, alignment_size: int = 256,
                 ndims: int = 3):

        super(VertixRefinePix3D, self).__init__()
        self.vertAlign = VertexAlign()

        in_channels = alignment_size + ndims
        if use_input_features:
            in_channels += num_features
        self.graphConv0 = GraphConv(in_channels, num_features)

        self.use_input_features = use_input_features

        self.graphConv1 = GraphConv(num_features+ndims,
                                    num_features)
        self.graphConv2 = GraphConv(num_features+ndims,
                                    num_features)
        self.linear = nn.Linear(num_features+ndims,
                                ndims, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, vertice_index: List[int], back_bone_features: Tensor,
                vertex_adjacency: Tensor, vertex_positions: Tensor,
                image_sizes: List, mesh_index: List[int] = None,
                vertex_features: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        # note that vertex_features is the concatination of all feature matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_positions is the concatination of all position matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_adjacency is a block diagonal matrix containing all adjacency matrices of the batch
        # as block on the diagonal and all off diagonal blocks are zero blocks

        # note that the back_bone_features are batched NxCxHxW
        if mesh_index is None:
            mesh_index = [1 for _ in image_sizes]

        # project the 3D mesh to the 2D feature planes and pool new features
        algined = self.vertAlign([back_bone_features], vertex_positions,
                                 vertice_index, image_sizes, mesh_index)

        # ∑Vx387 if there are initial vertex_features
        # and ∑Vx259 otherwise
        to_concat = [vertex_positions, algined]
        if not (vertex_features is None):
            assert self.use_input_features
            to_concat = [vertex_features]+to_concat
        else:
            assert not self.use_input_features
        vertex_features = torch.cat(to_concat, dim=1)

        # tramsform to input features to 128 features
        new_featues = self.graphConv0(vertex_features, vertex_adjacency)
        # ∑Vx131
        new_featues = torch.cat([vertex_positions, new_featues], dim=1)

        # ∑Vx128
        new_featues = self.graphConv1(new_featues, vertex_adjacency)
        # ∑Vx131
        new_featues = torch.cat([vertex_positions, new_featues], dim=1)
        # ∑Vx128
        new_featues = self.graphConv2(new_featues, vertex_adjacency)

        # ∑Vx131
        new_positions = torch.cat([vertex_positions, new_featues], dim=1)
        # ∑Vx3
        new_positions = self.linear(new_positions)
        new_positions = self.tanh(new_positions)
        new_positions = vertex_positions+new_positions

        return new_positions, new_featues


# there is also the marching cube algorithm https://github.com/pmneila/PyMCubes
# explained https://medium.com/zeg-ai/voxel-to-mesh-conversion-marching-cube-algorithm-43dbb0801359
# we might want to compare between them


class Cubify(nn.Module):
    '''
    Cubify is the process which takes a voxel occupancy probabilities grid and a threshold for binarizing occupancy.
    and outputing a list of 3D meshes.\n each occupied voxel is replaced with a cuboid triangle mesh with 8 vertices, 18 edges, and 12 faces.
    '''

    def __init__(self, threshold: float = 0.5):
        super(Cubify, self).__init__()
        self.threshold = threshold

        # neighbours kernel
        w = torch.zeros(6, 3, 3, 3, requires_grad=False).float()
        w[:, 1, 1, 1] = 1
        # surrounding pixels
        # F  z  y  x
        w[0, 0, 1, 1] = -1  # center-back
        w[1, 2, 1, 1] = -1  # center-front
        w[2, 1, 2, 1] = -1  # center-top
        w[3, 1, 0, 1] = -1  # center-bottom
        w[4, 1, 1, 0] = -1  # center-left
        w[5, 1, 1, 2] = -1  # center-right

        w = w.unsqueeze(1)
        self.register_buffer("kernel", w)

        # matrix that converts voxel coord to vertices coord
        # 6 4 5
        #    B F z y x
        deltas = torch.Tensor([
            [[0, 0, -0.5, -0.5, -0.5],  # back
             [0, 0, -0.5, -0.5, +0.5],
             [0, 0, -0.5, +0.5, -0.5],
             [0, 0, -0.5, +0.5, +0.5]],

            [[0, 0, +0.5, -0.5, -0.5],  # front
             [0, 0, +0.5, -0.5, +0.5],
             [0, 0, +0.5, +0.5, -0.5],
             [0, 0, +0.5, +0.5, +0.5]],

            [[0, 0, +0.5, -0.5, -0.5],  # top
             [0, 0, +0.5, -0.5, +0.5],
             [0, 0, -0.5, -0.5, -0.5],
             [0, 0, -0.5, -0.5, +0.5]],

            [[0, 0, -0.5, +0.5, -0.5],  # bottom
             [0, 0, -0.5, +0.5, +0.5],
             [0, 0, +0.5, +0.5, -0.5],
             [0, 0, +0.5, +0.5, +0.5]],

            [[0, 0, +0.5, -0.5, -0.5],  # left
             [0, 0, -0.5, -0.5, -0.5],
             [0, 0, +0.5, +0.5, -0.5],
             [0, 0, -0.5, +0.5, -0.5]],

            [[0, 0, -0.5, -0.5, +0.5],  # right
             [0, 0, +0.5, -0.5, +0.5],
             [0, 0, -0.5, +0.5, +0.5],
             [0, 0, +0.5, +0.5, +0.5]],
        ])
        self.register_buffer("deltas", deltas.requires_grad_(False))

    def forward(self, t: Tensor):
        B, Z, Y, X = t.shape
        t = (t > self.threshold).float()

        t = t.unsqueeze(1)

        # find for each voxel which faces we need to add
        # Bx6xVxVxV
        t = torch.nn.functional.conv3d(t, self.kernel, padding=1)

        # fetch indices that correspond to added cube faces
        # Nx1x5 B,F,z,y,x
        t = (t == 1).nonzero().float().unsqueeze(1)

        # for each face add vertices
        vs = []
        faces = []
        for i, d in enumerate(self.deltas):
            # 4xfx5
            pos = t[t[:, :, 1] == i] + d.unsqueeze(1)
            # order by coords
            pos = pos.permute(1, 0, 2).contiguous()
            # vx4
            pos = pos.view(-1, 5)[:, [0, 2, 3, 4]]
            vs.append(pos)

        # remove duplicate vertices and create inefficient faces
        # Vx4
        vs = torch.cat(vs)

        # order by batch
        vs = vs[vs[:, 0].argsort()]

        # fx3x4
        faces = torch.cat([torch.stack([vs[0::4], vs[1::4], vs[2::4]], dim=1),
                           torch.stack([vs[0::4], vs[2::4], vs[3::4]], dim=1)],
                          dim=1).view(-1, 4)

        f_index = (faces[:, 0].long().bincount()//3).tolist()
        # vx4
        vs = vs.unique(dim=0, sorted=False)
        v_index = vs[:, 0].long().bincount().tolist()
        f_class = torch.cuda.LongTensor if vs.is_cuda else torch.LongTensor

        # create an effiecient face to vertice mapping
        # using a projection to a 1d dimention where each voxel grid position
        # is mapped to a unique idx
        # uses double precision because the index can be very big
        projection = torch.Tensor(
            [8*Z*Y*X, 4*Y*X, 2*X, 1]).to(torch.float64).to(vs.device)

        h_table = {p: i for i, p in enumerate(
            (2*vs+1).to(torch.float64).mv(projection).long().flatten().tolist())}

        faces = f_class([h_table[k] for k in (2*faces+1).to(torch.float64).mv(projection).long().flatten().tolist()],
                        device=vs.device).view(-1, 3)

        # discard batch idx
        vs = vs[:, 1:]

        # create adj_matrix
        faces_t = faces.t()
        # get all directed edges
        idx_i, idx_j = torch.cat(
            [faces_t[:2], faces_t[1:], faces_t[::2]], dim=1)

        # duplicate to get undirected edges
        idx_i, idx_j = torch.cat([idx_i, idx_j], dim=0), torch.cat(
            [idx_j, idx_i], dim=0)

        adj_index = torch.stack([idx_i, idx_j], dim=0).unique(dim=1)

        # negate offsets
        offsets = np.cumsum(v_index)-v_index
        faces = torch.cat(
            [f-off for f, off in zip(faces.split(f_index), offsets)])

        return vs, v_index, faces, f_index, adj_index


class VoxelBranch(nn.Sequential):
    ''' the VoxelBranch predicts a grid of voxel occupancy probabilities by applying a fully convolutional network
        to the input feature map
    '''

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 256):
        super(VoxelBranch, self).__init__(
            # N x in_channels x V/2 x V/2
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            # N x 256 x V/2 x V/2
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=3, padding=1),
            # N x 256 x V/2 x V/2
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=2, stride=2),
            # N x 256 x V x V
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
            # N x out_channels x V x V
        )


class VertexAlign(nn.Module):
    # explained in the article http://openaccess.thecvf.com/content_ECCV_2018/papers/Nanyang_Wang_Pixel2Mesh_Generating_3D_ECCV_2018_paper.pdf
    # as perceptual feature pooling https://github.com/Tong-ZHAO/Pixel2Mesh-Pytorch
    # http://bigvid.fudan.edu.cn/pixel2mesh/eccv2018/Pixel2Mesh-supp.pdf
    # https://github.com/nywang16/Pixel2Mesh original source code
    # https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e

    """VertexAlign layer, which pool 2D features to mesh
    The layer projects a vertex of the mesh to the 2D image and use
    bilinear interpolation to get the corresponding feature.
    """

    def forward(self, img_features: List[Tensor], vertex_positions: Tensor,
                vertices_per_mesh: List[int], image_sizes: List[Tuple[int, int]],
                mesh_index: List[int]) -> Tensor:
        # right now it's a possibly ugly hack we iterate over individual meshes
        # and compute the projection on the respective feature maps
        # img_features is a list of batched features map
        # so for eg. the first mesh will be projected into img_features[0][0],...img_features[len(img_features)-1][0]
        assert len(mesh_index) == img_features[0].shape[0]
        assert sum(mesh_index) == len(vertices_per_mesh)
        vertices = vertex_positions.split(vertices_per_mesh)

        i = 0
        feats = []
        for idx, (num_meshes, size) in enumerate(zip(mesh_index, image_sizes)):
            for positions in vertices[i:i+num_meshes]:
                sample_maps = [f_map[idx] for f_map in img_features]
                feats.append(self.single_projection(sample_maps,
                                                    positions, size))
            i += num_meshes

        # ∑V x ∑ image channels
        return torch.cat(feats, dim=0)

    def single_projection(self, img_features: List[Tensor], vertex_positions: Tensor, size) -> Tensor:
        # perform a projection of vertex_positions accross all given feature maps

        # dimentions are addresed in order X,Y,Z
        # Y/ Z
        # X/ -Z
        # TODO magic numbers for camera intrinsics
        # http://bigvid.fudan.edu.cn/pixel2mesh/eccv2018/Pixel2Mesh-supp.pdf
        # ∑V
        h = 248 * (vertex_positions[:, 1] / vertex_positions[:, 2]) + 111.5
        w = 248 * (vertex_positions[:, 0] / -vertex_positions[:, 2]) + 111.5
        H, W = size
        # scale upto original image size
        h = torch.clamp(h, min=0, max=H-1)
        w = torch.clamp(w, min=0, max=W-1)

        feats = [self.project(img_feat, h, w, size)
                 for img_feat in img_features]

        # ∑V x ∑image channels
        output = torch.cat(feats, 1)

        return output

    def project(self, img_feat: Tensor, h: Tensor, w: Tensor, size) -> Tensor:
        size_y, size_x = img_feat.shape[-2:]
        H, W = size
        # scale to current feature map size
        # ∑V
        x = w / (float(W) / size_x)
        y = h / (float(H) / size_y)

        x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
        y1, y2 = torch.floor(y).long(), torch.ceil(y).long()

        x2 = torch.clamp(x2, max=size_x - 1)
        y2 = torch.clamp(y2, max=size_y - 1)

        # C x ∑V x  ∑V
        Q11 = img_feat[:, x1, y1].clone()
        Q12 = img_feat[:, x1, y2].clone()
        Q21 = img_feat[:, x2, y1].clone()
        Q22 = img_feat[:, x2, y2].clone()

        x, y = x.long(), y.long()

        weights = torch.mul(x2 - x, y2 - y)
        Q11 = torch.mul(weights.float().view(-1, 1),
                        torch.transpose(Q11, 0, 1))

        weights = torch.mul(x2 - x, y - y1)
        Q12 = torch.mul(weights.float().view(-1, 1),
                        torch.transpose(Q12, 0, 1))

        weights = torch.mul(x - x1, y2 - y)
        Q21 = torch.mul(weights.float().view(-1, 1),
                        torch.transpose(Q21, 0, 1))

        weights = torch.mul(x - x1, y - y1)
        Q22 = torch.mul(weights.float().view(-1, 1),
                        torch.transpose(Q22, 0, 1))

        # ∑V x C
        output = Q11 + Q21 + Q12 + Q22

        return output
