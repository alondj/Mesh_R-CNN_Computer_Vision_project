import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import VoxelBranch, Cubify, VertixRefinePix3D, VertixRefineShapeNet, ResVertixRefineShapenet
from typing import Tuple


# TODO add modularity
class ShapeNetModel(nn.Module):
    def __init__(self, feature_extractor: nn.Module, residual: bool = True):
        super(ShapeNetModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.voxelBranch = VoxelBranch(2048, 48)
        self.cubify = Cubify(0.2)

        refineClass = ResVertixRefineShapenet if residual else VertixRefineShapeNet

        self.refineStage0 = refineClass((137, 137), use_input_features=False)
        self.refineStage1 = refineClass((137, 137))
        self.refineStage2 = refineClass((137, 137))

    def forward(self, img: torch.Tensor) -> dict:
        img_feature_maps = self.feature_extractor(img)
        upscaled = F.interpolate(img_feature_maps[-1], scale_factor=4.8,
                                 mode='bilinear', align_corners=True)
        voxelGrid = self.voxelBranch(upscaled)
        vertice_index, faces_index, vertex_positions0, edge_index, mesh_faces = self.cubify(
            voxelGrid)

        vertex_features, vertex_positions1 = self.refineStage0(vertice_index, img_feature_maps,
                                                               edge_index, vertex_positions0)

        vertex_features, vertex_positions2 = self.refineStage0(vertice_index, img_feature_maps,
                                                               edge_index, vertex_positions1, vertex_features=vertex_features)

        vertex_features, vertex_positions3 = self.refineStage0(vertice_index, img_feature_maps,
                                                               edge_index, vertex_positions2, vertex_features=vertex_features)

        vertex_positions = [vertex_positions0, vertex_positions1,
                            vertex_positions2, vertex_positions3]

        output = dict()
        output['vertex_postions'] = vertex_positions
        output['edge_index'] = edge_index
        output['face_index'] = faces_index
        output['vertice_index'] = vertice_index
        output['faces'] = mesh_faces
        output['voxels'] = voxelGrid

        return output


class Pix3DModel(nn.Module):
    def __init__(self, feature_extractor: nn.Module, original_image_size: Tuple[int, int]):
        super(Pix3DModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.voxelBranch = VoxelBranch(256, 24)
        self.cubify = Cubify(0.2)
        self.refineStage0 = VertixRefinePix3D(original_image_size,
                                              use_input_features=False)
        self.refineStage1 = VertixRefinePix3D(original_image_size,
                                              use_input_features=True)
        self.refineStage2 = VertixRefinePix3D(original_image_size,
                                              use_input_features=True)

    def forward(self, image: torch.Tensor) -> dict:
        # TODO for now assume feature extractor handles
        # classification segmentation and masking and return a dictionary with results
        features = self.feature_extractor(image)

        roiAlign = features['roiAlign']
        voxelGrid = self.voxelBranch(roiAlign)

        vertice_index, faces_index, vertex_positions0, edge_index, mesh_faces = self.cubify(
            voxelGrid)

        vertex_features, vertex_positions1 = self.refineStage0(vertice_index, roiAlign,
                                                               edge_index, vertex_positions0)

        vertex_features, vertex_positions2 = self.refineStage0(vertice_index, roiAlign,
                                                               edge_index, vertex_positions1, vertex_features=vertex_features)

        vertex_features, vertex_positions3 = self.refineStage0(vertice_index, roiAlign,
                                                               edge_index, vertex_positions2, vertex_features=vertex_features)

        vertex_positions = [vertex_positions0, vertex_positions1,
                            vertex_positions2, vertex_positions3]

        output = dict()
        output['vertex_postions'] = vertex_positions
        output['edge_index'] = edge_index
        output['face_index'] = faces_index
        output['vertice_index'] = vertice_index
        output['faces'] = mesh_faces
        output['voxels'] = voxelGrid
        output.update(features)

        return output
