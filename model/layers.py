import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple, Optional, List
import datetime
import math
from .utils import aggregate_neighbours, dummy

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
        neighbours = aggregate_neighbours(vertex_adjacency,w1_features)

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

    def __init__(self, original_image_size: Tuple[int, int], use_input_features: bool = True,
                 num_features: int = 128, alignment_size: int = 3840,
                 ndims: int = 3):
        super(ResVertixRefineShapenet, self).__init__()

        self.vertAlign = VertexAlign(original_image_size)

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
                vertex_features: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        # note that vertex_features is the concatination of all feature matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_positions is the concatination of all position matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_adjacency is a block diagonal matrix containing all adjacency matrices of the batch
        # as block on the diagonal and all off diagonal blocks are zero blocks

        # note that the conv_feature are batched NxCxHxW

        # project the 3D mesh to the 2D feature planes and pool new features
        # ∑Vx3840
        aligned_vertices = self.vertAlign(img_feature_maps, vertex_positions,
                                          vertice_index)

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

    def __init__(self, original_image_size: Tuple[int, int], use_input_features: bool = True,
                 num_features: int = 128, alignment_size: int = 3840,
                 ndims: int = 3):
        super(VertixRefineShapeNet, self).__init__()
        self.vertAlign = VertexAlign(original_image_size)

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
                vertex_features: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        # note that vertex_features is the concatination of all feature matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_positions is the concatination of all position matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_adjacency is a block diagonal matrix containing all adjacency matrices of the batch
        # as block on the diagonal and all off diagonal blocks are zero blocks

        # note that the conv_feature are batched NxCxHxW

        # project the 3D mesh to the 2D feature planes and pool new features
        # ∑Vx3840
        aligned_vertices = self.vertAlign(img_feature_maps, vertex_positions,
                                          vertice_index)
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

#TODO every image in shapenet is with different shape
class VertixRefinePix3D(nn.Module):
    ''' VertixRefine are cells which given an image feature maps and a 3D mesh\n
        outputs an updated 3D mesh and vertex features
    '''

    def __init__(self, original_image_size: Tuple[int, int], use_input_features: bool = True,
                 num_features: int = 128, alignment_size: int = 256,
                 ndims: int = 3):

        super(VertixRefinePix3D, self).__init__()
        self.vertAlign = VertexAlign(original_image_size)

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

    def forward(self, vertice_index: List[int], back_bone_features: Tensor, vertex_adjacency: Tensor,
                vertex_positions: Tensor, vertex_features: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        # note that vertex_features is the concatination of all feature matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_positions is the concatination of all position matrices of the batch
        # along the vertex dimension (we stack them vertically)

        # note that vertex_adjacency is a block diagonal matrix containing all adjacency matrices of the batch
        # as block on the diagonal and all off diagonal blocks are zero blocks

        # note that the back_bone_features are batched NxCxHxW

        # project the 3D mesh to the 2D feature planes and pool new features
        algined = self.vertAlign([back_bone_features], vertex_positions,
                                 vertice_index)

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
    ''' Cubify is the process which takes a voxel occupancy probabilities grid and a threshold for binarizing occupancy.
        and outputing a list of 3D meshes.
        \n each occupied voxel is replaced with a cuboid triangle mesh with 8 vertices, 18 edges, and 12 faces.
    '''
    remove_shared_vertices_time=[]
    create_undirected_adjacency_matrix_time=[]
    iter_time=[]
    # I assume that z,y,x is the center of the cube so the vertices are at
    # bottom face z-0.5,y+-0.5,x+-0.5
    # top face z+0.5,y+-0.5,x+-0.5

    # Cubify will generate a mesh for each grid given we can think of a mesh as a group of disjoint graphs
    def __init__(self, threshold: float):
        super(Cubify, self).__init__()
        assert 0.0 <= threshold <= 1.0
        self.threshold = threshold

    def forward(self, voxel_probas: Tensor) -> Tuple[List[int], List[int], Tensor, Tensor, Tensor]:
        # output is vertices Vx3 , faces Fx3
        self.reset_stats()
        assert voxel_probas.ndim == 4
        out_device=voxel_probas.device
        N, C, H, W = voxel_probas.shape
        batched_vertex_positions, batched_faces, batched_adjacency_matrices = [], [], ([],[])
        vertice_index, faces_index = [], []
        offset=0

        start = datetime.datetime.now()
        # slow implementation just to know what I'm doing
        for n in range(N):
            vertices, faces = [], []
            iter_start=datetime.datetime.now()
            #TODO the problem is literlly this for loop
            for z in range(C):
                for y in range(H):
                    for x in range(W):
                        # we predicted a voxel at z,y,x
                        if voxel_probas[n, z, y, x] > self.threshold:
                            # this sections determines which vertices and cube faces to add
                            # the idea is if an ajacent cell has no voxel
                            # then the current voxel resides on the edge of the mesh
                            # so we add the 4 vertices and 2 traingle faces that are shared with
                            # the adjacent cell (they represent the border between the background and the object)
                            if z == 0 or voxel_probas[n, z-1, y, x] <= self.threshold:
                                # we predicted there is no voxel at z-1 ,y ,x
                                # add back faces
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
                            if z == C-1 or voxel_probas[n, z+1, y, x] <= self.threshold:
                                # we predicted there is no voxel at z+1 ,y ,x
                                # add front faces
                                v0, v1, v2, v3 = [
                                    (z+0.5, y-0.5, x-0.5),
                                    (z+0.5, y-0.5, x+0.5),
                                    (z+0.5, y+0.5, x-0.5),
                                    (z+0.5, y+0.5, x+0.5),
                                ]
                                vertices.extend([v0, v1, v2, v3])
                                faces.extend([
                                    (v0, v1, v2),
                                    (v1, v2, v3)
                                ])
                            if y == 0 or voxel_probas[n, z, y-1, x] <= self.threshold:
                                # we predicted there is no voxel at z ,y-1 ,x
                                # add top faces
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
                            if y == H - 1 or voxel_probas[n, z, y+1, x] <= self.threshold:
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
                            if x == 0 or voxel_probas[n, z, y, x-1] <= self.threshold:
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
                            if x == W-1 or voxel_probas[n, z, y, x+1] <= self.threshold:
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

            iter_end=datetime.datetime.now()
            self.iter_time.append(iter_end-iter_start)
            vertices, faces = self.remove_shared_vertices(vertices,faces,out_device)
            batched_vertex_positions.append(vertices)
            batched_faces.append(faces)
            adj_i,adj_j=self.create_undirected_adjacency_matrix(faces,offset)
            batched_adjacency_matrices[0].append(adj_i)
            batched_adjacency_matrices[1].append(adj_j)
            vertice_index.append(vertices.shape[0])
            faces_index.append(faces.shape[0])
            offset+=vertices.shape[0]

        #merge all meshes
        vertex_positions = torch.cat(batched_vertex_positions)
        mesh_faces = torch.cat(batched_faces)
        #we have a list of lists of i indices and a list of lists of j indices
        #just reduce to i indices and j indices
        idx_i,idx_j=batched_adjacency_matrices
        idx_i = torch.cat(idx_i, dim=0)
        idx_j = torch.cat(idx_j, dim=0)
        edge_index=torch.stack([idx_i,idx_j])

        fin = datetime.datetime.now()

        delta=fin-start

        self.exec_time=delta

        assert sum(vertice_index) == vertex_positions.shape[0]
        assert sum(faces_index) == mesh_faces.shape[0]

        self.summary(N)
        return vertice_index, faces_index, vertex_positions, edge_index, mesh_faces

    def remove_shared_vertices(self, vertices: List[Point], faces: List[Face],out_device:torch.device) -> Tuple[Tensor, Tensor]:
        # for performence reasons in the construction phase we duplicate shared vertices
        # and also we save the vertices coordinates explicitly inside the faces (each face is a tuple of 3 3D coordinates)

        # in this function we remove duplicate vertices and construct memory efficient face representaion
        # f(v0,v1,v2) => f(i0,i1,i2) such as vertices[i0]=v0 vertices[i1]=v1 vertices[i2]=v2
        # remember v0,v1,... are 3d points represented as 3 numbers
        # so the new representation uses ~3x less memory

        # in this stage we output tensor representation of vertices positions and faces
        # where the faces is represented as a shortTensor 16 bits per v_idx is plenty
        start = datetime.datetime.now()
        cannonic_vertices = list(dict.fromkeys(vertices))
        vertex_indices = {v: i for i, v in enumerate(cannonic_vertices)}

        efficient_faces = [[vertex_indices[v] for v in f] for f in faces]

        pos_class = torch.Tensor if out_device == torch.device('cpu') else torch.cuda.FloatTensor
        idx_class = torch.LongTensor if out_device == torch.device('cpu') else torch.cuda.LongTensor

        mesh_vertices = pos_class(cannonic_vertices,device=out_device)
        mesh_faces = idx_class(efficient_faces,device=out_device)

        fin = datetime.datetime.now()
        self.remove_shared_vertices_time.extend([fin-start])
        return mesh_vertices, mesh_faces

    def create_undirected_adjacency_matrix(self, faces: Tensor,offset:int) -> List:
        start = datetime.datetime.now()
        
        faces_t=faces.t()

        #get all directed edges
        idx_i,idx_j=torch.cat([faces_t[:2],faces_t[1:],faces_t[::2]],dim=1)+offset

        #duplicate to get undirected edges
        idx_i, idx_j = torch.cat([idx_i, idx_j], dim=0), torch.cat([idx_j, idx_i], dim=0)
        
        fin = datetime.datetime.now()

        self.create_undirected_adjacency_matrix_time.extend([fin-start])

        return idx_i,idx_j
        
    def summary(self,b_size):
        total_remove_shared_time=self.remove_shared_vertices_time[0]
        total_remove_shared_time=sum(self.remove_shared_vertices_time[1:],total_remove_shared_time)
                
        total_adj_creation_time=self.create_undirected_adjacency_matrix_time[0]
        total_adj_creation_time=sum(self.create_undirected_adjacency_matrix_time[1:],total_adj_creation_time)


        total_iter_time = self.iter_time[0]
        total_iter_time = sum(self.iter_time[1:], total_iter_time)

        assert len(self.iter_time)==b_size
        assert len(self.remove_shared_vertices_time) == b_size
        assert len(self.create_undirected_adjacency_matrix_time) == b_size
        

        loop_time = self.exec_time - (total_adj_creation_time+total_remove_shared_time)


        print(f"total {self.exec_time}")
        print(f"loop time {loop_time}")
        print(f"total iter time {total_iter_time}")
        print(f"avg iter time {total_iter_time/b_size}")
        print(f"face and pos creation {total_remove_shared_time}")
        print(f"adj creation {total_adj_creation_time}\n")

    def reset_stats(self):
        self.remove_shared_vertices_time = []
        self.create_undirected_adjacency_matrix_time = []
        self.merge_index_time = []
        self.iter_time = []
        
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

    def __init__(self, original_image_size: Tuple[int, int]):
        '''
        original_image_size is a tuple (h,w) of sizes of the original image fed into the network
        '''
        super(VertexAlign, self).__init__()
        assert len(original_image_size) == 2
        self.h, self.w = original_image_size

    def forward(self, img_features: List[Tensor], vertex_positions: Tensor, vertices_per_mesh: List[int]) -> Tensor:
        # right now it's a possibly ugly hack we iterate over individual meshes
        # and compute the projection on the respective feature maps
        # img_features is a list of batched features map
        # so for eg. the first mesh will be projected into img_features[0][0],...img_features[len(img_features)-1][0]
        assert len(vertices_per_mesh) == img_features[0].shape[0]
        vertices = vertex_positions.split(vertices_per_mesh)

        feats = []
        for idx, positions in enumerate(vertices):
            sample_maps = [f_map[idx] for f_map in img_features]
            feats.append(self.single_projection(sample_maps, positions))

        # ∑V x ∑ image channels
        return torch.cat(feats, dim=0)

    def single_projection(self, img_features: List[Tensor], vertex_positions: Tensor) -> Tensor:
        # perform a projection of vertex_positions accross all given feature maps

        # dimentions are addresed in order X,Y,Z
        # Y/ Z
        # X/ -Z
        # TODO magic numbers for camera intrinsics
        # http://bigvid.fudan.edu.cn/pixel2mesh/eccv2018/Pixel2Mesh-supp.pdf
        # ∑V
        h = 248 * (vertex_positions[:, 1] / vertex_positions[:, 2]) + 111.5
        w = 248 * (vertex_positions[:, 0] / -vertex_positions[:, 2]) + 111.5

        # scale upto original image size
        h = torch.clamp(h, min=0, max=self.h-1)
        w = torch.clamp(w, min=0, max=self.w-1)

        feats = [self.project(img_feat, h, w)
                 for img_feat in img_features]

        # ∑V x ∑image channels
        output = torch.cat(feats, 1)

        return output

    def project(self, img_feat: Tensor, h: Tensor, w: Tensor) -> Tensor:
        size_y, size_x = img_feat.shape[-2:]

        # scale to current feature map size
        # ∑V
        x = w / (float(self.w) / size_x)
        y = h / (float(self.h) / size_y)

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


def tesst_cubify():
    cube = Cubify(0.2)

    inp = torch.randn(1, 48, 48, 48, device='cuda:0')
    _ = cube(inp)


if __name__ == "__main__":
    tesst_cubify()
