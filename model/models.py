from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.model_zoo import load_url
from torchvision.models.resnet import ResNet, Bottleneck, model_urls as res_urls
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.mask_rcnn import model_urls as mask_urls
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign, RoIAlign

from model.our_roi_head import build_RoI_head
from model.utils import filter_pix3d_input

from .layers import (Cubify, ResVertixRefineShapenet, VertixRefinePix3D,
                     VertixRefineShapeNet, VoxelBranch)


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

    def forward(self, images: Tensor, targets=None) -> dict:
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        backbone_out, feature_maps = self.backbone(images, targets)
        sizes = [i.shape[1:] for i in images]
        upscaled = F.interpolate(feature_maps[-1], scale_factor=4.8,
                                 mode='bilinear', align_corners=True)

        voxelGrid = self.voxelBranch(upscaled)

        output = dict()
        output['voxels'] = voxelGrid
        output['backbone'] = backbone_out

        if self.voxel_only:
            return output

        vertex_positions0, vertice_index, faces, face_index, adj_index = self.cubify(
            voxelGrid)

        vertex_positions1, vertex_features = self.refineStages[0](vertice_index, feature_maps,
                                                                  adj_index, vertex_positions0,
                                                                  sizes)

        vertex_positions = [vertex_positions0, vertex_positions1]

        for stage in self.refineStages[1:]:
            new_positions, vertex_features = stage(vertice_index, feature_maps,
                                                   adj_index, vertex_positions[-1],
                                                   sizes, vertex_features=vertex_features)
            vertex_positions.append(new_positions)

        output['vertex_positions'] = vertex_positions
        output['edge_index'] = adj_index
        output['face_index'] = face_index
        output['vertice_index'] = vertice_index
        output['faces'] = faces
        output['graphs_per_image'] = [1]

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
    # TODO we should have our own pretrained model and not the default one
    # when that time comes remove the pretrained arg and set path as deafult model path
    url = res_urls['resnet50']
    model = ShapeNetResNet50(loss_function, Bottleneck, [3, 4, 6, 3])

    if pretrained:
        model.load_state_dict(load_url(url, progress=True))

    if num_classes != model.fc.out_features:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


class Pix3DModel(nn.Module):
    def __init__(self, backbone: nn.Module,
                 cubify_threshold: float = 0.2,
                 voxelBranchChannels: Tuple[int, int] = (256, 24),
                 alignmenet_channels: int = 256,
                 vertex_feature_dim: int = 128,
                 num_refinement_stages: int = 3,
                 voxel_only: bool = False):

        super(Pix3DModel, self).__init__()
        self.backbone = backbone
        self.voxelBranch = VoxelBranch(*voxelBranchChannels)
        self.cubify = Cubify(cubify_threshold)
        self.voxel_only = voxel_only
        stages = [VertixRefinePix3D(alignment_size=alignmenet_channels,
                                    use_input_features=False,
                                    num_features=vertex_feature_dim)]

        for _ in range(num_refinement_stages - 1):
            stages.append(VertixRefinePix3D(alignment_size=alignmenet_channels,
                                            num_features=vertex_feature_dim,
                                            use_input_features=True))

        self.refineStages = nn.ModuleList(stages)

    def forward(self, images: List[Tensor], targets: Optional[List[Dict]] = None) -> dict:
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        backbone_out, roiAlign, graphs_per_image = self.backbone(
            images, targets)

        voxelGrid = self.voxelBranch(roiAlign)

        output = dict()
        output['voxels'] = voxelGrid
        output['backbone'] = backbone_out
        output['roi_input'] = roiAlign

        if self.voxel_only:
            return output

        vertex_positions0, vertice_index, faces, face_index, adj_index = self.cubify(
            voxelGrid)

        sizes = [i.shape[1:] for i in images]
        vertex_positions1, vertex_features = self.refineStages[0](vertice_index, roiAlign,
                                                                  adj_index, vertex_positions0,
                                                                  sizes)

        vertex_positions = [vertex_positions0, vertex_positions1]

        for stage in self.refineStages[1:]:
            new_positions, vertex_features = stage(vertice_index, roiAlign,
                                                   adj_index, vertex_positions[-1], sizes,
                                                   vertex_features=vertex_features)
            vertex_positions.append(new_positions)

        output['vertex_positions'] = vertex_positions
        output['edge_index'] = adj_index
        output['face_index'] = face_index
        output['vertice_index'] = vertice_index
        output['faces'] = faces
        output['graphs_per_image'] = graphs_per_image

        return output


class Pix3DMask_RCNN(MaskRCNN):
    def __init__(self, num_classes: int, **MaskRCNN_kwargs):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(Pix3DMask_RCNN, self).__init__(
            backbone, num_classes=num_classes, **MaskRCNN_kwargs)

    def forward(self, images: List[Tensor], targets: Optional[List[Dict]] = None):
        """
        Arguments:
            images (List[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        # additional ROI features for pix3d
        graphs_per_image = [p.shape[0] for p in proposals]
        detections, pix3d_input, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets)
        if self.training:
            # during training we limit the nuber of roi features
            # that are passed to the GCN as each feature creates a graph
            pix3d_input = filter_pix3d_input(targets, proposals,
                                             pix3d_input)

        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if self.training:
            return losses, pix3d_input, graphs_per_image

        return detections, pix3d_input, graphs_per_image


def pretrained_MaskRcnn(num_classes=10, pretrained=True):
    # TODO we should have our own pretrained model and not the default one
    # when that time comes remove the pretrained arg and set path as deafult model path
    url = mask_urls['maskrcnn_resnet50_fpn_coco']
    model = Pix3DMask_RCNN(91)
    if pretrained:
        model.load_state_dict(load_url(url, progress=True))

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads = build_RoI_head(model.backbone.out_channels, num_classes=num_classes, box_detections_per_img=1,
                                     box_roi_pool=MultiScaleRoIAlign(featmap_names=[0, 1, 2, 3],
                                                                     output_size=12,
                                                                     sampling_ratio=1),
                                     mask_predictor=MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes))

    return model
