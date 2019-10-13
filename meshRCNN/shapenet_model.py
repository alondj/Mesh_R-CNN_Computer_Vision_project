from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.model_zoo import load_url
from torchvision.models.resnet import ResNet, Bottleneck, model_urls as res_urls

from .layers import (Cubify, ResVertixRefineShapenet,
                     VertixRefineShapeNet, VoxelBranch)
from .loss_functions import voxel_loss, batched_mesh_loss
from data.dataloader import Batch


class ShapeNetModel(nn.Module):
    def __init__(self, backbone: nn.Module, residual: bool = False,
                 cubify_threshold: float = 0.2,
                 voxelBranchChannels: Tuple[int, int] = (2048, 48),
                 alignmenet_channels: int = 3840,
                 vertex_feature_dim: int = 128,
                 num_refinement_stages: int = 3,
                 voxel_only: bool = False):
        super(ShapeNetModel, self).__init__()
        self.backbone = backbone
        self.voxelBranch = VoxelBranch(*voxelBranchChannels)
        self.cubify = Cubify(cubify_threshold)
        self.voxel_only = voxel_only
        refineClass = ResVertixRefineShapenet if residual else VertixRefineShapeNet

        stages = [refineClass(alignment_size=alignmenet_channels,
                              use_input_features=False,
                              num_features=vertex_feature_dim)]

        for _ in range(num_refinement_stages - 1):
            stages.append(refineClass(alignment_size=alignmenet_channels,
                                      num_features=vertex_feature_dim,
                                      use_input_features=True))

        self.refineStages = nn.ModuleList(stages)

    def forward(self, images: Tensor, targets: Batch = None) -> dict:
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        backbone_targets = targets.backbone_targets if self.training else None
        backbone_out, feature_maps = self.backbone(images,
                                                   backbone_targets)
        sizes = [i.shape[1:] for i in images]
        upscaled = F.interpolate(feature_maps[-1], scale_factor=4.8,
                                 mode='bilinear', align_corners=True)

        voxelGrid = self.voxelBranch(upscaled)
        output = dict()

        if self.training and self.backbone.training:
            output['backbone_loss'] = backbone_out
        elif not self.training:
            assert not self.backbone.training
            output['backbone'] = backbone_out

        if self.training:
            output['voxel_loss'] = voxel_loss(voxelGrid, targets.voxels)
        else:
            output['voxels'] = voxelGrid

        if self.voxel_only:
            return output

        mesh_index = [1 for _ in images]
        vertex_positions0, vertice_index, faces, face_index, adj_index = self.cubify(
            voxelGrid)

        vertex_positions1, vertex_features = self.refineStages[0](vertice_index, feature_maps,
                                                                  adj_index, vertex_positions0,
                                                                  sizes, mesh_index=mesh_index)

        vertex_positions = [vertex_positions0, vertex_positions1]

        for stage in self.refineStages[1:]:
            new_positions, vertex_features = stage(vertice_index, feature_maps,
                                                   adj_index, vertex_positions[-1],
                                                   sizes, mesh_index=mesh_index,
                                                   vertex_features=vertex_features)
            vertex_positions.append(new_positions)

        if self.training:
            chamfer, normal, edge = batched_mesh_loss(vertex_positions[1:], faces, adj_index,
                                                      vertice_index, face_index, targets)
            output.update({'chamfer_loss': chamfer,
                           "edge_loss": edge, "normal_loss": normal})
        else:
            output['vertex_positions'] = vertex_positions
            output['edge_index'] = adj_index
            output['face_index'] = face_index
            output['vertice_index'] = vertice_index
            output['faces'] = faces
            output['mesh_index'] = mesh_index

        return output


class ShapeNetResNet50(ResNet):
    '''
    this model outputs logits for class scores
    '''

    def __init__(self, loss_function, *res_args, **res_kwargs):
        super(ShapeNetResNet50, self).__init__(*res_args, **res_kwargs)
        self.loss = loss_function

    def forward(self, x: Tensor, targets: Optional[Tensor] = None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        img0 = self.layer1(x)
        img1 = self.layer2(img0)
        img2 = self.layer3(img1)
        img3 = self.layer4(img2)

        x = self.avgpool(img3)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.softmax(x, -1)

        if self.training:
            targets = targets.type(torch.cuda.LongTensor)
            return self.loss(x, targets), [img0, img1, img2, img3]

        return x, [img0, img1, img2, img3]


def pretrained_ResNet50(loss_function, num_classes=10, pretrained=True):
    # when that time comes remove the pretrained arg and set path as deafult model path
    url = res_urls['resnet50']
    model = ShapeNetResNet50(loss_function, Bottleneck, [3, 4, 6, 3])

    if pretrained:
        model.load_state_dict(load_url(url, progress=True))

    if num_classes != model.fc.out_features:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
