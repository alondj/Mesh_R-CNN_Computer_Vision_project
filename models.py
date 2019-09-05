import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import VoxelBranch, Cubify, VertixRefinePix3D, VertixRefineShapeNet,\
    ResVertixRefineShapenet
from typing import Tuple


class ShapeNetModel(nn.Module):
    def __init__(self, feature_extractor: nn.Module, residual: bool = True,
                 cubify_threshold: float = 0.2, image_shape: Tuple[int, int] = (137, 137),
                 voxelBranchChannels: Tuple[int, int] = (2048, 48),
                 alignmenet_channels: int = 3840,
                 vertex_feature_dim: int = 128,
                 num_refinement_stages: int = 3):
        super(ShapeNetModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.voxelBranch = VoxelBranch(*voxelBranchChannels)
        self.cubify = Cubify(cubify_threshold)

        refineClass = ResVertixRefineShapenet if residual else VertixRefineShapeNet

        stages = [refineClass(image_shape, alignment_size=alignmenet_channels,
                              use_input_features=False,
                              num_features=vertex_feature_dim)]

        for _ in range(num_refinement_stages-1):
            stages.append(refineClass(image_shape, alignment_size=alignmenet_channels,
                                      num_features=vertex_feature_dim,
                                      use_input_features=True))

        self.refineStages = nn.ModuleList(stages)

    def forward(self, img: torch.Tensor) -> dict:
        img_feature_maps = self.feature_extractor(img)

        upscaled = F.interpolate(img_feature_maps[-1], scale_factor=4.8,
                                 mode='bilinear', align_corners=True)

        voxelGrid = self.voxelBranch(upscaled)

        vertice_index, faces_index, vertex_positions0, edge_index, mesh_faces = self.cubify(
            voxelGrid)

        vertex_features, vertex_positions1 = self.refineStages[0](vertice_index, img_feature_maps,
                                                                  edge_index, vertex_positions0)

        vertex_positions = [vertex_positions0, vertex_positions1]

        for stage in self.refineStages[1:]:
            vertex_features, new_positions = stage(vertice_index, img_feature_maps,
                                                   edge_index, vertex_positions[-1], vertex_features=vertex_features)
            vertex_positions.append(new_positions)

        output = dict()
        output['vertex_postions'] = vertex_positions
        output['edge_index'] = edge_index
        output['face_index'] = faces_index
        output['vertice_index'] = vertice_index
        output['faces'] = mesh_faces
        output['voxels'] = voxelGrid
        output['feature_maps'] = img_feature_maps

        return output


class Pix3DModel(nn.Module):
    def __init__(self, image_input_size: Tuple[int, int], feature_extractor: nn.Module,
                 cubify_threshold: float = 0.2,
                 voxelBranchChannels: Tuple[int, int] = (256, 24),
                 alignmenet_channels: int = 256,
                 vertex_feature_dim: int = 128,
                 num_refinement_stages: int = 3):

        super(Pix3DModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.voxelBranch = VoxelBranch(*voxelBranchChannels)
        self.cubify = Cubify(cubify_threshold)

        stages = [VertixRefinePix3D(image_input_size, alignment_size=alignmenet_channels,
                                    use_input_features=False,
                                    num_features=vertex_feature_dim)]

        for _ in range(num_refinement_stages-1):
            stages.append(VertixRefinePix3D(image_input_size, alignment_size=alignmenet_channels,
                                            num_features=vertex_feature_dim,
                                            use_input_features=True))

        self.refineStages = nn.ModuleList(stages)

    def forward(self, image: torch.Tensor) -> dict:
        # TODO for now assume feature extractor handles
        # classification segmentation and masking and return a dictionary with results
        features = self.feature_extractor(image)

        roiAlign = features['roiAlign']
        voxelGrid = self.voxelBranch(roiAlign)

        vertice_index, faces_index, vertex_positions0, edge_index, mesh_faces = self.cubify(
            voxelGrid)

        vertex_features, vertex_positions1 = self.refineStages[0](vertice_index, roiAlign,
                                                                  edge_index, vertex_positions0)

        vertex_positions = [vertex_positions0, vertex_positions1]

        for stage in self.refineStages[1:]:
            vertex_features, new_positions = stage(vertice_index, roiAlign,
                                                   edge_index, vertex_positions[-1], vertex_features=vertex_features)
            vertex_positions.append(new_positions)

        output = dict()
        output['vertex_postions'] = vertex_positions
        output['edge_index'] = edge_index
        output['face_index'] = faces_index
        output['vertice_index'] = vertice_index
        output['faces'] = mesh_faces
        output['voxels'] = voxelGrid
        output.update(features)

        return output
