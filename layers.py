import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from utils import conv_output, convT_output,to_block_diagonal,from_block_diagonal


#data representation for graphs:
# adjacency matrix: we create block diagonal matrix (possibly sparse in the future)
# vertex features matrix: we concatenate over the vertex dim resulting in total_vertices x Num_features
# this representation allows us to batch graph operations

# TODO create adjacency matrices from the faces matrices

#TODO check that all dimensions are correct I suspect the article has them switched

#TODO we use the N,Z,Y,X notation in Cubify make sure we are consistent (I think Pix2Mesh used the N,Z,X,Y notation)

#TODO perform batch operations in all layers

#TODO maybe use block matrices in sparse format(only if the dense representation is slow)

# TODO fast efficient cubify now it envolves 4 loops over all dimensions

# TODO maybe add more modularity to everything such as specify number of layers kernels etc

Point = Tuple[float, float, float]
Face = Tuple[Point, Point, Point]



class GraphConv(nn.Module):
    '''GraphConv(D1→D2) computes new features based on a linear projection of current features
       and a linear projection of adjacent vertices features as denoted by:
        f′i = ReLU(W0xfi +∑ j∈N(i) W1xfj)
    '''
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

    def forward(self, vertex_features: Tensor, vertex_adjacency: Tensor) -> Tensor:
        #note that vertex_features is the concatination of all feature matrices of the batch
        #along the vertex dimension (we stack them vertically)

        #note that vertex_adjacency is a block diagonal matrix containing all adjacency matrices of the batch
        #as block on the diagonal and all off diagonal blocks are zero blocks

        # ∑VxIn @ InxOut => ∑VxOut
        w0_features = torch.mm(vertex_features, self.w0)

        # ∑VxIn @ InxOut => ∑VxOut
        w1_features = torch.mm(vertex_features, self.w1)

        # batch ∑j∈N(i)W1xfj
        # use the adjacency matrix as a mask
        # note a vertex is not connected to itself
        # ∑Vx∑V @ ∑VxOut => ∑VxOut
        neighbours = torch.mm(vertex_adjacency, w1_features)

        # aggregate features of neighbours
        new_features = w0_features + neighbours

        return self.relu(new_features)


class ResGraphConv(nn.Module):
    '''ResGraphConv(D1→D2) layers consists of two graph convolution, layers each preceeded by ReLU,
       and an additive skip connection with linear projection if input dimension D1 differs from output dimension D2
    '''
    def __init__(self, in_channels: int, out_channels: int):
        super(ResGraphConv, self).__init__()
        self.conv0 = GraphConv(in_channels, out_channels)
        self.conv1 = GraphConv(out_channels, out_channels)

        self.project = (in_channels == out_channels)

        if self.project:
            self.projection = nn.Linear(in_channels, out_channels,bias=False)
        else:
            self.projection = None

    def forward(self, vertex_features: Tensor, vertex_adjacency: Tensor) -> Tensor:
         #note that vertex_features is the concatination of all feature matrices of the batch
        #along the vertex dimension (we stack them vertically)

        #note that vertex_adjacency is a block diagonal matrix containing all adjacency matrices of the batch
        #as block on the diagonal and all off diagonal blocks are zero blocks

        # ∑VxIn @ InxOut => ∑VxOut
        skip = vertex_features if not self.project else self.projection(
            vertex_features)

        # ∑VxIn => ∑VxOut
        out = self.conv0(vertex_features, vertex_adjacency)
        # ∑VxOut=> ∑VxOut
        out = self.conv1(out, vertex_adjacency)

        return skip+out


class ResVertixRefineShapenet(nn.Module):
    ''' VertixRefine are cells which given an image feature maps and a 3D mesh\n
        outputs an updated 3D mesh and vertex features
    '''
    def __init__(self, use_input_features: bool = True):
        super(ResVertixRefineShapenet, self).__init__()

        self.vertAlign0 = VertexAlign()
        self.vertAlign1 = VertexAlign()
        self.vertAlign2 = VertexAlign()
        self.vertAlign3 = VertexAlign()

        self.linear = nn.Linear(3840, 128,bias=False)

        in_channels = 259 if use_input_features else 131
        self.resGraphConv0 = ResGraphConv(in_channels, 128)
        self.use_input_features = use_input_features

        self.resGraphConv1 = ResGraphConv(128, 128)
        self.resGraphConv2 = ResGraphConv(128, 128)
        self.graphConv = GraphConv(128, 3)

        self.tanh = nn.Tanh()

    def forward(self, conv2_3: Tensor, conv3_4: Tensor, conv4_6: Tensor, conv5_3: Tensor,
                vertex_adjacency: Tensor, vertex_positions: Tensor, vertex_features: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        
        #note that vertex_features is the concatination of all feature matrices of the batch
        #along the vertex dimension (we stack them vertically)

        #note that vertex_positions is the concatination of all position matrices of the batch
        #along the vertex dimension (we stack them vertically)

        #note that vertex_adjacency is a block diagonal matrix containing all adjacency matrices of the batch
        #as block on the diagonal and all off diagonal blocks are zero blocks

        #note that the conv_feature are batched NxCxHxW 

        #project the 3D mesh to the 2D feature planes and pool new features
        #TODO the conv features are batched but vertex positions are concatenated
        vert0 = self.vertAlign0(conv2_3, vertex_positions)
        vert1 = self.vertAlign1(conv3_4, vertex_positions)
        vert2 = self.vertAlign2(conv4_6, vertex_positions)
        vert3 = self.vertAlign3(conv5_3, vertex_positions)
        # ∑Vx3840
        concat0 = torch.cat([vert0, vert1, vert2, vert3], dim=2)
        
        # ∑Vx128
        projected = self.linear(concat0)

        # ∑Vx259 if there are initial vertex_features
        # and ∑Vx131 otherwise
        to_concat = [vertex_positions, projected]
        if vertex_features != None:
            assert self.use_input_features
            to_concat = [vertex_features]+to_concat
        else:
            assert not self.use_input_features
        vertex_features = torch.cat(to_concat, dim=2)

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
    def __init__(self, use_input_features: bool = True):
        super(VertixRefineShapeNet, self).__init__()
        self.vertAlign0 = VertexAlign()
        self.vertAlign1 = VertexAlign()
        self.vertAlign2 = VertexAlign()
        self.vertAlign3 = VertexAlign()

        self.linear0 = nn.Linear(3840, 128,bias=False)

        in_channels = 259 if use_input_features else 131
        self.graphConv0 = GraphConv(in_channels, 131)
        self.use_input_features = use_input_features

        self.graphConv1 = GraphConv(131, 128)
        self.graphConv2 = GraphConv(131, 128)
        self.linear1 = nn.Linear(128, 3,bias=False)
        self.tanh = nn.Tanh()

    def forward(self, conv2_3: Tensor, conv3_4: Tensor, conv4_6: Tensor, conv5_3: Tensor,
                vertex_adjacency: Tensor, vertex_positions: Tensor, vertex_features: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        #note that vertex_features is the concatination of all feature matrices of the batch
        #along the vertex dimension (we stack them vertically)

        #note that vertex_positions is the concatination of all position matrices of the batch
        #along the vertex dimension (we stack them vertically)

        #note that vertex_adjacency is a block diagonal matrix containing all adjacency matrices of the batch
        #as block on the diagonal and all off diagonal blocks are zero blocks

        #note that the conv_feature are batched NxCxHxW

        #project the 3D mesh to the 2D feature planes and pool new features
        #TODO the conv features are batched but vertex positions are concatenated
        vert0 = self.vertAlign0(conv2_3, vertex_positions)
        vert1 = self.vertAlign1(conv3_4, vertex_positions)
        vert2 = self.vertAlign2(conv4_6, vertex_positions)
        vert3 = self.vertAlign3(conv5_3, vertex_positions)
        # ∑Vx3840
        concat0 = torch.cat([vert0, vert1, vert2, vert3], dim=2)
        
        # ∑Vx128
        projected = self.linear(concat0)

        # ∑Vx259 if there are initial vertex_features
        # and ∑Vx131 otherwise
        to_concat = [vertex_positions, projected]
        if vertex_features != None:
            assert self.use_input_features
            to_concat = [vertex_features]+to_concat
        else:
            assert not self.use_input_features
        vertex_features = torch.cat(to_concat, dim=2)

        # transforms input features to 128 features
        new_features = self.graphConv0(vertex_features, vertex_adjacency)

        # ∑Vx131
        new_features = torch.cat([vertex_positions, new_features], dim=2)
        # ∑Vx128
        new_features = self.graphConv1(new_features, vertex_adjacency)
        # ∑Vx131
        new_features = torch.cat([vertex_positions, new_features], dim=2)
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
    def __init__(self, use_input_features: bool = True):
        super(VertixRefinePix3D, self).__init__()
        self.vertAlign = VertexAlign()

        in_channels = 387 if use_input_features else 259
        self.graphConv0 = GraphConv(in_channels, 128)

        self.use_input_features = use_input_features

        self.graphConv1 = GraphConv(131, 128)
        self.graphConv2 = GraphConv(131, 128)
        self.linear = nn.Linear(131, 3,bias=False)
        self.tanh = nn.Tanh()

    def forward(self, back_bone_features: Tensor, vertex_adjacency: Tensor,
                vertex_positions: Tensor, vertex_features: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        #note that vertex_features is the concatination of all feature matrices of the batch
        #along the vertex dimension (we stack them vertically)

        #note that vertex_positions is the concatination of all position matrices of the batch
        #along the vertex dimension (we stack them vertically)

        #note that vertex_adjacency is a block diagonal matrix containing all adjacency matrices of the batch
        #as block on the diagonal and all off diagonal blocks are zero blocks

        #note that the back_bone_features are batched NxCxHxW

        #project the 3D mesh to the 2D feature planes and pool new features
        #TODO the conv features are batched but vertex positions are concatenated
        algined = self.vertAlign(back_bone_features, vertex_positions)

        # ∑Vx387 if there are initial vertex_features
        # and ∑Vx259 otherwise
        to_concat = [vertex_positions, algined]
        if vertex_features != None:
            assert self.use_input_features
            to_concat = [vertex_features]+to_concat
        else:
            assert not self.use_input_features
        vertex_features = torch.cat(to_concat, dim=2)

        # tramsform to input features to 128 features
        new_featues = self.graphConv0(vertex_features, vertex_adjacency)
        # ∑Vx131
        new_featues = torch.cat([vertex_positions, new_featues], dim=2)

        # ∑Vx128
        new_featues = self.graphConv1(new_featues, vertex_adjacency)
        # ∑Vx131
        new_featues = torch.cat([vertex_positions, new_featues], dim=2)
        # ∑Vx128
        new_featues = self.graphConv2(new_featues, vertex_adjacency)

        # ∑Vx131
        new_positions = torch.cat([vertex_positions, new_featues], dim=2)

        # ∑Vx3
        new_positions = self.linear(new_featues)
        new_positions = self.tanh(new_positions)
        new_positions = vertex_positions+new_positions

        return new_positions, new_featues



# there is also the marching cube algorithm https://github.com/pmneila/PyMCubes
# explained https://medium.com/zeg-ai/voxel-to-mesh-conversion-marching-cube-algorithm-43dbb0801359
# we might want to compare between them
class Cubify(nn.Module):
    ''' Cubify is the process which takes a voxel occupancy probabilities grid and a threshold for binarizing occupancy.
        and outputing a list of 3D meshes.
        \n each occupied voxel is replaced with a cuboid triangle mesh with 8 vertices, 18 edges, and 12 faces.
    '''
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

    def remove_shared_vertices(self, vertices: List[Point], faces: List[Face]) -> Tuple[Tensor, Tensor]:
        # for performence reasons in the construction phase we duplicate shared vertices
        # and also we save the vertices coordinates explicitly inside the faces (each face is a tuple of 3 3D coordinates)

        # in this function we remove duplicate vertices and construct memory efficient face representaion
        # f(v0,v1,v2) => f(i0,i1,i2) such as vertices[i0]=v0 vertices[i1]=v1 vertices[i2]=v2
        # remember v0,v1,... are 3d points represented as 3 numbers
        # so the new representation uses ~3x less memory

        # in this stage we output tensor representation of vertices positions and faces
        # where the faces is represented as a shortTensor 16 bits per v_idx is plenty

        cannonic_vertices = list(dict.fromkeys(vertices))
        vertex_indices = {v: i for i, v in enumerate(cannonic_vertices)}

        efficient_faces = [[vertex_indices[v] for v in f] for f in faces]

        return torch.Tensor(cannonic_vertices, device=self.out_device), torch.Tensor(efficient_faces, device=self.out_device).short()


class VoxelBranch(nn.Sequential):
    ''' the VoxelBranch predicts a grid of voxel occupancy probabilities by applying a fully convolutional network
        to the input feature map 
    '''
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


class FCN(nn.Module):
    ''' FCN is a fully convolutional network based on the VGG16 architecture emmitting 4 feature maps\n
        given an input of shape NxCxHxW the 4 feature maps are of shapes:\n
        Nx256xH/4xW/4 , Nx512xH/8xW/8 , Nx1024xH/16xW/16 , Nx2048xH/32xW/32
    '''
    def __init__(self, in_channels: int):
        super(FCN, self).__init__()
        self.conv0_1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1)
        self.conv0_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.conv1_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)  # cut h and w by half
        self.conv1_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(128, 256, 3, stride=2, padding=1)  # cut h and w by half
        self.conv2_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(256, 512, 3, stride=2, padding=1)  # cut h and w by half
        self.conv3_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(512, 1024, 5, stride=2, padding=2)  # cut h and w by half
        self.conv4_2 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(1024, 2048, 5, stride=2, padding=2)  # cut h and w by half
        self.conv5_2 = nn.Conv2d(2048, 2048, 3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(2048, 2048, 3, stride=1, padding=1)
        self.conv5_4 = nn.Conv2d(2048, 2048, 3, stride=1, padding=1)

    def forward(self, img) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        img = F.relu(self.conv0_1(img))
        img = F.relu(self.conv0_2(img))

        img = F.relu(self.conv1_1(img))
        img = F.relu(self.conv1_2(img))
        img = F.relu(self.conv1_3(img))

        img = F.relu(self.conv2_1(img))
        img = F.relu(self.conv2_2(img))
        img = F.relu(self.conv2_3(img))
        img2 = torch.squeeze(img)  # 56

        img = F.relu(self.conv3_1(img))
        img = F.relu(self.conv3_2(img))
        img = F.relu(self.conv3_3(img))
        img3 = torch.squeeze(img)  # 28

        img = F.relu(self.conv4_1(img))
        img = F.relu(self.conv4_2(img))
        img = F.relu(self.conv4_3(img))
        img4 = torch.squeeze(img)  # 14

        img = F.relu(self.conv5_1(img))
        img = F.relu(self.conv5_2(img))
        img = F.relu(self.conv5_3(img))
        img = F.relu(self.conv5_4(img))
        img5 = torch.squeeze(img)  # 7

        return img2, img3, img4, img5


class VertexAlign(nn.Module):
    # explained in the article http://openaccess.thecvf.com/content_ECCV_2018/papers/Nanyang_Wang_Pixel2Mesh_Generating_3D_ECCV_2018_paper.pdf
    # as perceptual feature pooling https://github.com/Tong-ZHAO/Pixel2Mesh-Pytorch
    # https://github.com/nywang16/Pixel2Mesh original source code

    """VertexAlign layer, which pool 2D features to mesh
    The layer projects a vertex of the mesh to the 2D image and use
    bilinear interpolation to get the corresponding feature.
    """
    #TODO image shape is hard coded

    def __init__(self):
        super(VertexAlign, self).__init__()

    def forward(self, img_features: List[Tensor], vertex_positions: Tensor) -> Tensor:
        
        #TODO should be Y/ Z
        #               X/ -Z
        h = 248 * (vertex_positions[:, 1] / vertex_positions[:, 2]) + 111.5
        w = 248 * (vertex_positions[:, 0] / -vertex_positions[:, 2]) + 111.5


        shapes="feat shapes"
        for f in img_features:
            shapes+=f" {f.shape}"
        print(f"single forward feat_shape {shapes}\nvertex_pos shape {vertex_positions.shape}")
        print(f"single_forward h shape {h.shape}  w shape {w.shape}")

        h = torch.clamp(h, min=0, max=136)
        w = torch.clamp(w, min=0, max=136)

        feats = [self.project(img_feat, h, w)
                 for img_feat in img_features]

        output = torch.cat(feats, 1)

        print(f"single align output shape {output.shape}")

        return output

    def project(self, img_feat: Tensor, h: Tensor, w: Tensor) -> Tensor:
        print("\nsingle_project")
        print(f"image features shape {img_feat.shape}")
        img_size = img_feat.shape[-1]
        x = h / (137. / img_size)
        y = w / (137. / img_size)

        x1, x2 = torch.floor(x).long(), torch.ceil(x).long()
        y1, y2 = torch.floor(y).long(), torch.ceil(y).long()

        x2 = torch.clamp(x2, max=img_size - 1)
        y2 = torch.clamp(y2, max=img_size - 1)

        print(f"x1 shape {x1.shape} y1 shape {y1.shape}")
        print(f"x2 shape {x2.shape} y2 shape {y2.shape}\n")

        Q11 = img_feat[:, x1, y1].clone()
        Q12 = img_feat[:, x1, y2].clone()
        Q21 = img_feat[:, x2, y1].clone()
        Q22 = img_feat[:, x2, y2].clone()

        print("Qs original shapes")
        print(f"Q11 shape {Q11.shape} Q12 shape {Q12.shape}")
        print(f"Q21 shape {Q21.shape} Q22 shape {Q22.shape}\n")

        x, y = x.long(), y.long()

        print("Q11 calculation")
        weights = torch.mul(x2 - x, y2 - y)
        print(
            f"weight shape {weights.shape} weightT shape {weights.view(-1,1).shape}")
        print(f"Q11T shape {torch.transpose(Q11, 0, 1).shape}\n")
        Q11 = torch.mul(weights.float().view(-1, 1),
                        torch.transpose(Q11, 0, 1))

        print("Q12 calculation")
        weights = torch.mul(x2 - x, y - y1)
        print(
            f"weight shape {weights.shape} weightT shape {weights.view(-1,1).shape}")
        print(f"Q12T shape {torch.transpose(Q12, 0, 1).shape}\n")
        Q12 = torch.mul(weights.float().view(-1, 1),
                        torch.transpose(Q12, 0, 1))

        print("Q21 calculation")
        weights = torch.mul(x - x1, y2 - y)
        print(
            f"weight shape {weights.shape} weightT shape {weights.view(-1,1).shape}")
        print(f"Q21T shape {torch.transpose(Q21, 0, 1).shape}\n")
        Q21 = torch.mul(weights.float().view(-1, 1),
                        torch.transpose(Q21, 0, 1))

        print("Q22 calculation")
        weights = torch.mul(x - x1, y - y1)
        print(
            f"weight shape {weights.shape} weightT shape {weights.view(-1,1).shape}")
        print(f"Q22T shape {torch.transpose(Q22, 0, 1).shape}\n")
        Q22 = torch.mul(weights.float().view(-1, 1),
                        torch.transpose(Q22, 0, 1))

        output = Q11 + Q21 + Q12 + Q22

        print(f"single_projection shape {output.shape}")
        print("\n")

        return output

if __name__ == "__main__":
    V = 128
    align = VertexAlign()

    img2,img3,img4,img5=FCN(3)(torch.randn(1,3,137,137))
    single_pos: Tensor = torch.rand(V,3)*137
    single_features = [img2, img3, img4, img5]

  
    print("aligning vertices")
    out = align(single_features,single_pos,)
