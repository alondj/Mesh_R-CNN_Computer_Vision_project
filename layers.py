import torch
import torch.Tensor as Tensor
import torch.nn as nn

from typing import Tuple, Optional

# TODO use sparse Tensor for adjacency matrix
# the problem is that bmm does not support sparse dense batch multipication and a for loop is slow


# TODO fast efficient cubify now it envolves 4 loops over all dimentions

# TODO maybe add more modularity for the refinement and voxel branches

class GraphConv(nn.Module):
    # f′i = ReLU(W0xfi +∑j∈N(i)W1xfj)
    def __init__(self, in_channels: int, out_channels: int):
        super(GraphConv, self).__init__()
        self.w0 = torch.nn.Parameter(data=torch.empty(in_channels, out_channels),
                                     requires_grad=True)
        self.w1 = torch.nn.Parameter(data=torch.empty(in_channels, out_channels),
                                     requires_grad=True)

        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0/torch.sqrt(self.w0.shape[0])
        self.w0.data.uniform_(-bound, bound)
        self.w1.data.uniform_(-bound, bound)

    def forward(self, vertix_features: Tensor, vertex_adjacency: Tensor) -> Tensor:
        # sanity checks for dimentions
        assert vertix_features.dim() == vertex_adjacency.dim()
        assert vertix_features.shape[0] == vertex_adjacency.shape[0]
        assert vertix_features[1] == vertex_adjacency.shape[1]
        assert vertex_adjacency.shape[2] == vertex_adjacency.shape[1]

        # NxVxIn @ InxOut => NxVxOut
        w0_features = torch.matmul(vertix_features, self.w0)

        # NxVxIn @ InxOut => NxVxOut
        w1_features = torch.matmul(vertix_features, self.w1)

        # batch ∑j∈N(i)W1xfj
        # use the adjacency matrix as a mask
        # note a vertex is not connected to itself
        # NxVxV @ NxVxOut
        # resulting in output shape of NxVxOut
        neighbours = torch.bmm(vertex_adjacency, w1_features)

        # aggregate features of neighbours
        new_features = w0_features + neighbours

        return self.relu(new_features)


class ResGraphConv(nn.Module):
    # ResGraphConv(D1→D2)consists of two graph convolution layers (each preceeded by ReLU)
    # and an additive skip connection, with a linear projection if the input and output dimensions are different

    def __init__(self, in_channels: int, out_channels: int):
        super(ResGraphConv, self).__init__()
        self.conv0 = GraphConv(in_channels, out_channels)
        self.conv1 = GraphConv(out_channels, out_channels)

        self.project = (in_channels == out_channels)

        if self.project:
            self.projection = nn.Linear(in_channels, out_channels)
        else:
            self.projection = None

    def forward(self, vertix_features: Tensor, vertex_adjacency: Tensor) -> Tensor:
        # NxVxOut
        skip = vertix_features if not self.project else self.projection(
            vertix_features)

        # NxVxOut
        out = self.conv0(vertix_features, vertex_adjacency)
        # NxVxOut
        out = self.conv1(out, vertex_adjacency)

        return skip+out


# vertix refinement stages for ShapeNet
class ResVertixRefineShapenet(nn.Module):
    # explained in the article
    def __init__(self, use_input_features: bool = True):
        super(ResVertixRefineShapenet, self).__init__()

        self.vertAlign0 = VertexAlign()
        self.vertAlign1 = VertexAlign()
        self.vertAlign2 = VertexAlign()
        self.vertAlign3 = VertexAlign()

        self.linear = nn.Linear(3840, 128)

        in_channels = 259 if use_input_features else 131
        self.resGraphConv0 = ResGraphConv(in_channels, 128)
        self.use_input_features = use_input_features

        self.resGraphConv1 = ResGraphConv(128, 128)
        self.resGraphConv2 = ResGraphConv(128, 128)
        self.graphConv = GraphConv(128, 3)

        self.tanh = nn.Tanh()

    def forward(self, conv2_3: Tensor, conv3_4: Tensor, conv4_6: Tensor, conv5_3: Tensor,
                vertex_adjacency: Tensor, vertex_positions: Tensor, vertex_features: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        vert0 = self.vertAlign0(conv2_3, vertex_positions)
        vert1 = self.vertAlign1(conv3_4, vertex_positions)
        vert2 = self.vertAlign2(conv4_6, vertex_positions)
        vert3 = self.vertAlign3(conv5_3, vertex_positions)

        # NxVx3840
        concat0 = torch.cat([vert0, vert1, vert2, vert3], dim=2)
        # NxVx128
        projected = self.linear(concat0)

        # NxVx259 if there are initial vertex_features
        # and NxVx131 otherwise
        to_concat = [vertex_positions, projected]
        if vertex_features != None:
            assert self.use_input_features
            to_concat = [vertex_features]+to_concat
        else:
            assert not self.use_input_features
        vertex_features = torch.cat(to_concat, dim=2)

        # transforms input features to 128 features
        new_features = self.resGraphConv0(vertex_features, vertex_adjacency)
        # NxVx128
        new_features = self.resGraphConv1(new_features, vertex_adjacency)
        new_features = self.resGraphConv2(new_features, vertex_adjacency)

        # new_positions is of shape NxVx3
        new_positions = self.graphConv(new_features, vertex_adjacency)
        new_positions = self.tanh(new_positions)
        new_positions = vertex_positions+new_positions

        return new_positions, new_features


class VertixRefineShapeNet(nn.Module):
    # explained in the article
    def __init__(self, use_input_features: bool = True):
        super(VertixRefineShapeNet, self).__init__()
        self.vertAlign0 = VertexAlign()
        self.vertAlign1 = VertexAlign()
        self.vertAlign2 = VertexAlign()
        self.vertAlign3 = VertexAlign()

        self.linear0 = nn.Linear(3840, 128)

        in_channels = 259 if use_input_features else 131
        self.graphConv0 = GraphConv(in_channels, 131)
        self.use_input_features = use_input_features

        self.graphConv1 = GraphConv(131, 128)
        self.graphConv2 = GraphConv(131, 128)
        self.linear1 = nn.Linear(128, 3)
        self.tanh = nn.Tanh()

    def forward(self, conv2_3: Tensor, conv3_4: Tensor, conv4_6: Tensor, conv5_3: Tensor,
                vertex_adjacency: Tensor, vertex_positions: Tensor, vertex_features: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        vert0 = self.vertAlign0(conv2_3, vertex_positions)
        vert1 = self.vertAlign1(conv3_4, vertex_positions)
        vert2 = self.vertAlign2(conv4_6, vertex_positions)
        vert3 = self.vertAlign3(conv5_3, vertex_positions)

        # NxVx3840
        concat0 = torch.cat([vert0, vert1, vert2, vert3], dim=2)
        # NxVx128
        projected = self.linear(concat0)

        # NxVx259 if there are initial vertex_features
        # and NxVx131 otherwise
        to_concat = [vertex_positions, projected]
        if vertex_features != None:
            assert self.use_input_features
            to_concat = [vertex_features]+to_concat
        else:
            assert not self.use_input_features
        vertex_features = torch.cat(to_concat, dim=2)

        # transforms input features to 128 features
        new_features = self.graphConv0(vertex_features, vertex_adjacency)

        # NxVx131
        new_features = torch.cat([vertex_positions, new_features], dim=2)
        # NxVx128
        new_features = self.graphConv1(new_features, vertex_adjacency)
        # NxVx131
        new_features = torch.cat([vertex_positions, new_features], dim=2)
        # NxVx128
        new_features = self.graphConv2(new_features, vertex_adjacency)

        # NxVx3
        new_positions = self.linear1(new_features)
        new_positions = self.tanh(new_positions)
        new_positions = vertex_positions+new_positions

        return new_positions, new_features


# vertix refinement stage for Pix3D
class VertixRefinePix3D(nn.Module):
    # explained in the article
    def __init__(self, use_input_features: bool = True):
        super(VertixRefinePix3D, self).__init__()
        self.vertAlign = VertexAlign()

        in_channels = 387 if use_input_features else 259
        self.graphConv0 = GraphConv(in_channels, 128)

        self.use_input_features = use_input_features

        self.graphConv1 = GraphConv(131, 128)
        self.graphConv2 = GraphConv(131, 128)
        self.linear = nn.Linear(131, 3)
        self.tanh = nn.Tanh()

    def forward(self, back_bone_features: Tensor, vertex_adjacency: Tensor,
                vertex_positions: Tensor, vertex_features: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        algined = self.vertAlign(back_bone_features, vertex_positions)

        # NxVx387 if there are initial vertex_features
        # and NxVx259 otherwise
        to_concat = [vertex_positions, algined]
        if vertex_features != None:
            assert self.use_input_features
            to_concat = [vertex_features]+to_concat
        else:
            assert not self.use_input_features
        vertex_features = torch.cat(to_concat, dim=2)

        # tramsform to input features to 128 features
        new_featues = self.graphConv0(vertex_features, vertex_adjacency)
        # NxVx131
        new_featues = torch.cat([vertex_positions, new_featues], dim=2)

        # NxVx128
        new_featues = self.graphConv1(new_featues, vertex_adjacency)
        # NxVx131
        new_featues = torch.cat([vertex_positions, new_featues], dim=2)
        # NxVx128
        new_featues = self.graphConv2(new_featues, vertex_adjacency)

        # NxVx131
        new_positions = torch.cat([vertex_positions, new_featues], dim=2)

        # NxVx3
        new_positions = self.linear(new_featues)
        new_positions = self.tanh(new_positions)
        new_positions = vertex_positions+new_positions

        return new_positions, new_featues


class VertexAlign(nn.Module):
    # explained in the article http://openaccess.thecvf.com/content_ECCV_2018/papers/Nanyang_Wang_Pixel2Mesh_Generating_3D_ECCV_2018_paper.pdf
    # as perceptual feature pooling
    # TODO vertexAlign
    def forward(self, features, vertex_positions):
        pass


# conversion from voxels to meshes

# there is also the marching cube algorithm https://github.com/pmneila/PyMCubes
# explained https://medium.com/zeg-ai/voxel-to-mesh-conversion-marching-cube-algorithm-43dbb0801359
# we might want to compare between them
class Cubify(nn.Module):
    # explained in the article
    # each voxel is replaced with a triangular mesh cube (each face is replaced with 2 triangles)
    # resulting in 8 vertices 18 edges 12 faces
    # I assume that z,y,x is the center of the cube so the vertices are at
    # bottom face z-0.5,y+-0.5,x+-0.5
    # top face z+0.5,y+-0.5,x+-0.5
    def __init__(self, threshold: float, output_device: torch.device):
        super(Cubify, self).__init__()
        assert 0.0 <= threshold <= 1.0
        self.threshold = threshold
        self.out_device = output_device

    def forward(self, voxel_probas: Tensor) -> Tuple[Tensor, Tensor]:
        # output is vertices NxVx3 , faces NxFx3
        N, D, H, W = voxel_probas.shape
        batched_vertices, batched_faces = [], []
        # slow implementation just to know what I'm doing
        for n in range(N):
            vertices, faces = [], []
            for z in range(D):
                for y in range(H):
                    for x in range(W):
                        # we predicted a voxel at z,y,x
                        if voxel_probas[n, z, y, x] > self.threshold:
                            # this sections determines which vertices and cube faces to add
                            # the idea is if an ajacent cell has no voxel
                            # then the current voxel resides on the edge of the mesh
                            # so we add the 4 vertices and 2 traingle faces that are shared with
                            # the adjacent cell (they represent the border between the background and the object)
                            if voxel_probas[n, z-1, y, x] <= self.threshold:
                                # we predicted there is no voxel at z-1 ,y ,x
                                # add bottom faces
                                v0, v1, v2, v3 = [(z-0.5, y-0.5, x-0.5),
                                                  (z-0.5, y-0.5, x+0.5),
                                                  (z-0.5, y+0.5, x-0.5),
                                                  (z-0.5, y+0.5, x+0.5)]
                                vertices.extend([
                                    v0, v1, v2, v3
                                ])
                                faces.extend([
                                    (v0, v1, v2),
                                    (v1, v2, v3)
                                ])
                            if voxel_probas[n, z+1, y, x] <= self.threshold:
                                # we predicted there is no voxel at z+1 ,y ,x
                                # add top faces
                                v0, v1, v2, v3 = [
                                    (z+0.5, y-0.5, x-0.5),
                                    (z+0.5, y-0.5, x+0.5),
                                    (z+0.5, y+0.5, x-0.5),
                                    (z+0.5, y+0.5, x+0.5),
                                ]
                                faces.extend([
                                    (v0, v1, v2),
                                    (v1, v2, v3)
                                ])
                            if voxel_probas[n, z, y-1, x] <= self.threshold:
                                # we predicted there is no voxel at z ,y-1 ,x
                                # add back faces
                                v0, v1, v2, v3 = [
                                    (z+0.5, y-0.5, x-0.5),
                                    (z+0.5, y-0.5, x+0.5),
                                    (z-0.5, y-0.5, x-0.5),
                                    (z-0.5, y-0.5, x+0.5),
                                ]
                                vertices.extend([v0, v1, v2, v3])
                                faces.extend([
                                    (v0, v1, v2),
                                    (v1, v2, v3)
                                ])
                            if voxel_probas[n, z, y+1, x] <= self.threshold:
                                # we predicted there is no voxel at z ,y+1 ,x
                                # add front faces
                                v0, v1, v2, v3 = [
                                    (z-0.5, y+0.5, x-0.5),
                                    (z-0.5, y+0.5, x+0.5),
                                    (z+0.5, y+0.5, x-0.5),
                                    (z+0.5, y+0.5, x+0.5),
                                ]
                                vertices.extend([v0, v1, v2, v3])
                                faces.extend([
                                    (v0, v1, v2),
                                    (v1, v2, v3)
                                ])
                            if voxel_probas[n, z, y, x-1] <= self.threshold:
                                # we predicted there is no voxel at z ,y ,x-1
                                # add left faces
                                v0, v1, v2, v3 = [
                                    (z+0.5, y-0.5, x-0.5),
                                    (z-0.5, y-0.5, x-0.5),
                                    (z+0.5, y+0.5, x-0.5),
                                    (z-0.5, y+0.5, x-0.5),
                                ]
                                vertices.extend([v0, v1, v2, v3])
                                faces.extend([
                                    (v0, v1, v2),
                                    (v1, v2, v3)
                                ])
                            if voxel_probas[n, z, y, x+1] <= self.threshold:
                                # we predicted there is no voxel at z ,y ,x+1
                                # add right faces
                                v0, v1, v2, v3 = [
                                    (z-0.5, y-0.5, x+0.5),
                                    (z+0.5, y-0.5, x+0.5),
                                    (z-0.5, y+0.5, x+0.5),
                                    (z+0.5, y+0.5, x+0.5),
                                ]
                                vertices.extend([v0, v1, v2, v3])
                                faces.extend([
                                    (v0, v1, v2),
                                    (v1, v2, v3)
                                ])

            cannonic_vs, cannonic_fs = self.remove_shared_vertices(vertices,
                                                                   faces)
            batched_vertices.append(cannonic_vs)
            batched_faces.append(cannonic_fs)

        return torch.stack(batched_vertices), torch.stack(batched_faces)

    def remove_shared_vertices(self, vertices, faces) -> Tuple[Tensor, Tensor]:
        # for performence reasons in the construction phase we duplicate shared vertices
        # and also we save the vertices coordinates explicitly inside the faces (each face is a tuple of 3 3D coordinates)

        # in this function we remove duplicate vertices and construct memory efficient face representaion
        # f(v0,v1,v2) => f(i0,i1,i2) such as vertices[i0]=v0 vertices[i1]=v1 vertices[i2]=v2
        # remember v0,v1,... are 3d points represented as 3 numbers
        # so the new representation uses ~3x less memory

        # in this stage we output tensor representation of vertices positions and faces
        # where the faces is represented as a shortTensor 16 bits per v_idx is plenty
        assert len(vertices) > 0 and len(vertices) % 4 == 0
        assert len(faces) > 0 and len(faces) % 2 == 0

        cannonic_vertices = list(dict.fromkeys(vertices))
        vertex_indices = {v: i for i, v in enumerate(cannonic_vertices)}

        efficient_faces = [[vertex_indices[v] for v in f] for f in faces]

        return torch.Tensor(cannonic_vertices, device=self.out_device), torch.Tensor(efficient_faces, device=self.out_device).short()


class VoxelBranch(nn.Sequential):
    # explained in the article
    def __init__(self, in_channels: int, out_channels: int):
        super(VoxelBranch, self).__init__(
            # N x in_channels x V/2 x V/2
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            # N x 256 x V/2 x V/2
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # N x 256 x V/2 x V/2
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            # N x 256 x V x V
            nn.Conv2d(256, out_channels, kernel_size=1)
            # N x out_channels x V x V
        )
