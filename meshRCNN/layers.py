import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign, boxes as box_ops
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.roi_heads import fastrcnn_loss, maskrcnn_inference, \
    keypointrcnn_inference, keypointrcnn_loss, maskrcnn_loss
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
from typing import Tuple, Optional, List
import math
from .utils import aggregate_neighbours
import numpy as np
from utils.rotation import rotation
# data representation for graphs:
# adjacency matrix: we just save occupied indices in coo format
# vertex features matrix: we concatenate over the vertex dim resulting in total_vertices x Num_features
# this representation allows us to batch graph operations
# mesh_faces: a matrix of relevant vertex indices of size total_faces x 3
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

        # note that vertex_adjacency is in coo format of occupied indices in the adj matrix

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
                mesh_index: List[int] = None) -> Tuple[Tensor, Tensor]:

        # note that vertex_features is the concatination of all feature matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_positions is the concatination of all position matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_adjacency is in sparse COO format of occupied indices in the adj matrix

        # note that the conv_feature are batched NxCxHxW

        # unless specified otherwise only one mesh per image
        if mesh_index is None:
            mesh_index = [1 for _ in image_sizes]

        # project the 3D mesh to the 2D feature planes and pool new features
        # ∑Vx3840
        aligned_vertices = self.vertAlign(img_feature_maps, vertex_positions,
                                          vertice_index, image_sizes, mesh_index)

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
                image_sizes: List, mesh_index: List[int] = None,
                vertex_features: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        # note that vertex_features is the concatination of all feature matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_positions is the concatination of all position matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_adjacency is in sparse COO format of occupied indices in the adj matrix

        # note that the conv_feature are batched NxCxHxW

        if mesh_index is None:
            mesh_index = [1 for _ in image_sizes]

        # project the 3D mesh to the 2D feature planes and pool new features
        # ∑Vx3840
        aligned_vertices = self.vertAlign(img_feature_maps, vertex_positions,
                                          vertice_index, image_sizes, mesh_index)
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

        # note that vertex_adjacency is in sparse COO format of occupied indices in the adj matrix

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

        # if we have no vertices then error ugly hack
        if len(vs) == 0:
            raise ValueError("empty grid")

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
        vs = vs.mm(rotation(90, tensor=True).to(device=vs.device,
                                                dtype=vs.dtype))
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
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            # N x out_channels x V x V
            nn.Sigmoid()
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
        if self.training:
            assert len(vertices_per_mesh) == len(image_sizes)
            assert mesh_index == [1 for _ in image_sizes]

        assert len(mesh_index) == len(image_sizes)

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


class ModifiedRoIHead(RoIHeads):
    '''ROI_Heads implementation that returns ROI features of detections
       during training and inference
    '''

    def postprocess_detections(self, box_features, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores and featyres per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_features = box_features.split(boxes_per_image, 0)
        all_boxes = []
        all_scores = []
        all_labels = []
        all_features = []
        for features, boxes, scores, image_shape in zip(pred_features, pred_boxes, pred_scores, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()

            # at this stage every feature correspond to nClasses-1 boxes
            # so the idea is to keep track of which original box indices are kept
            # and from them compute feature_indices = box_idx // nClasses-1
            box_keep_idxs = torch.arange(boxes.shape[0])

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            if boxes[inds].shape[0] > 0:
                boxes, scores, labels, box_keep_idxs = boxes[
                    inds], scores[inds], labels[inds], box_keep_idxs[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            if boxes[keep].shape[0] > 0:
                boxes, scores, labels, box_keep_idxs = boxes[
                    keep], scores[keep], labels[keep], box_keep_idxs[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            if boxes[keep].shape[0] > 0:
                boxes, scores, labels, box_keep_idxs = boxes[
                    keep], scores[keep], labels[keep], box_keep_idxs[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

            feature_indices = box_keep_idxs / (num_classes - 1)
            all_features.append(features[feature_indices])
        return all_boxes, all_scores, all_labels, all_features

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                assert t["boxes"].dtype.is_floating_point, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint:
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(
                proposals, targets)

        # this is where we changed the code so that boxes will be returned in training too
        box_features_return = self.box_roi_pool(features, proposals,
                                                image_shapes)
        box_features = self.box_head(box_features_return)
        class_logits, box_regression = self.box_predictor(box_features)
        result, losses = [], {}

        if self.training:
            boxes, scores, new_labels, GCN_features = self.postprocess_detections(box_features_return, class_logits,
                                                                                  box_regression, proposals,
                                                                                  image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=new_labels[i],
                        scores=scores[i],
                    )
                )

            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = dict(loss_classifier=loss_classifier,
                          loss_box_reg=loss_box_reg)
        else:
            boxes, scores, labels, GCN_features = self.postprocess_detections(box_features_return, class_logits,
                                                                              box_regression, proposals,
                                                                              image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                    )
                )

        if self.has_mask:
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])

            mask_features = self.mask_roi_pool(
                features, mask_proposals, image_shapes)
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)

            loss_mask = {}
            if self.training:
                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals,
                    gt_masks, gt_labels, pos_matched_idxs)
                loss_mask = dict(loss_mask=loss_mask)
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        if self.has_keypoint:
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])

            keypoint_features = self.keypoint_roi_pool(
                features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                gt_keypoints = [t["keypoints"] for t in targets]
                loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals,
                    gt_keypoints, pos_matched_idxs)
                loss_keypoint = dict(loss_keypoint=loss_keypoint)
            else:
                keypoints_probs, kp_scores = keypointrcnn_inference(
                    keypoint_logits, keypoint_proposals)
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses, GCN_features


def build_RoI_head(out_channels, num_classes=None, box_roi_pool=None, box_head=None, box_predictor=None,
                   box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                   box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                   box_batch_size_per_image=512, box_positive_fraction=0.25,
                   bbox_reg_weights=None, mask_predictor=None, mask_roi_pool=None, mask_head=None):
    if box_roi_pool is None:
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=7,
            sampling_ratio=2)

    if box_head is None:
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)

    if box_predictor is None:
        representation_size = 1024
        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes)

    if mask_roi_pool is None:
        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=[0, 1, 2, 3],
            output_size=14,
            sampling_ratio=2)

    if mask_head is None:
        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

    roi_heads = ModifiedRoIHead(
        # Box
        box_roi_pool, box_head, box_predictor,
        box_fg_iou_thresh, box_bg_iou_thresh,
        box_batch_size_per_image, box_positive_fraction,
        bbox_reg_weights,
        box_score_thresh, box_nms_thresh, box_detections_per_img, mask_predictor=mask_predictor, mask_head=mask_head,
        mask_roi_pool=mask_roi_pool)
    return roi_heads