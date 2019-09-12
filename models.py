from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from layers import VoxelBranch, Cubify, VertixRefinePix3D, VertixRefineShapeNet,\
    ResVertixRefineShapenet
from typing import Tuple, List, Optional, Dict
# MaskRCNN FasterRCNN, GeneralizedRCNN, RoIHeads MultiScaleRoIAlign RoIAlign
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN
from torchvision.models.detection.mask_rcnn import model_urls as mask_urls
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet, resnet50
from torchvision.models.resnet import Bottleneck, model_urls as res_urls
from torchvision.ops import MultiScaleRoIAlign, RoIAlign
from collections import OrderedDict
from torch.utils.model_zoo import load_url
from torchvision.models.detection.image_list import ImageList


class ShapeNetModel(nn.Module):
    def __init__(self, feature_extractor: nn.Module, residual: bool = False,
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

    def forward(self, img: Tensor, targets=None) -> dict:
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        backbone_out, feature_maps = self.feature_extractor(img, targets)

        upscaled = F.interpolate(feature_maps[-1], scale_factor=4.8,
                                 mode='bilinear', align_corners=True)

        voxelGrid = self.voxelBranch(upscaled)

        vertice_index, faces_index, vertex_positions0, edge_index, mesh_faces = self.cubify(
            voxelGrid)

        vertex_features, vertex_positions1 = self.refineStages[0](vertice_index, feature_maps,
                                                                  edge_index, vertex_positions0)

        vertex_positions = [vertex_positions0, vertex_positions1]

        for stage in self.refineStages[1:]:
            vertex_features, new_positions = stage(vertice_index, feature_maps,
                                                   edge_index, vertex_positions[-1], vertex_features=vertex_features)
            vertex_positions.append(new_positions)

        output = dict()
        output['vertex_postions'] = vertex_positions
        output['edge_index'] = edge_index
        output['face_index'] = faces_index
        output['vertice_index'] = vertice_index
        output['faces'] = mesh_faces
        output['voxels'] = voxelGrid
        output['backbone'] = backbone_out
        output['graphs_per_image'] = [1]

        return output


class ShapeNetResNet50(ResNet):
    def __init__(self, loss_function, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ShapeNetResNet50, self).__init__(block, layers, num_classes=1000,
                                               zero_init_residual=False,
                                               groups=1, width_per_group=64,
                                               replace_stride_with_dilation=None,
                                               norm_layer=None)
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

        if self.training:
            return self.loss(x, targets), [img0, img1, img2, img3]

        return x, [img0, img1, img2, img3]


def pretrained_ResNet50(loss_function, num_classes=10, pretrained=True):
    url = res_urls['resnet50']
    model = ShapeNetResNet50(loss_function, Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_url(url, progress=True)
        model.load_state_dict(state_dict)

    if num_classes != model.fc.out_features:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


class Pix3DModel(nn.Module):
    def __init__(self, feature_extractor: nn.Module, image_shape: Tuple[int, int] = (224, 224),
                 cubify_threshold: float = 0.2,
                 voxelBranchChannels: Tuple[int, int] = (256, 24),
                 alignmenet_channels: int = 256,
                 vertex_feature_dim: int = 128,
                 num_refinement_stages: int = 3):

        super(Pix3DModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.voxelBranch = VoxelBranch(*voxelBranchChannels)
        self.cubify = Cubify(cubify_threshold)

        stages = [VertixRefinePix3D(image_shape, alignment_size=alignmenet_channels,
                                    use_input_features=False,
                                    num_features=vertex_feature_dim)]

        for _ in range(num_refinement_stages-1):
            stages.append(VertixRefinePix3D(image_shape, alignment_size=alignmenet_channels,
                                            num_features=vertex_feature_dim,
                                            use_input_features=True))

        self.refineStages = nn.ModuleList(stages)

    def forward(self, image: Tensor, targets: Optional[List[Dict]] = None) -> dict:
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        backbone_out, roiAlign, graphs_per_image = self.feature_extractor(
            image, targets)

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
        output['backbone'] = backbone_out
        output['roi_input'] = roiAlign
        output['graphs_per_image'] = graphs_per_image

        return output


class Pix3DMask_RCNN(MaskRCNN):
    def __init__(self, num_classes: int, **MaskRCNN_kwargs):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(Pix3DMask_RCNN, self).__init__(
            backbone, num_classes=num_classes, **MaskRCNN_kwargs)

        # TODO the output shape of this layer is
        # output(Tensor[K, C, output_size[0], output_size[1]])
        # how will it work with the voxel branch?
        # this layer has no parameters which is nice
        self.mesh_ROI = MultiScaleRoIAlign(featmap_names=[0, 1, 2, 3],
                                           output_size=12,
                                           sampling_ratio=1)

    def forward(self, images: Tensor, targets: Optional[List[Dict]] = None):
        """
        Arguments:
            images (Tensor): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        images = list(images.split(1))
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
        pix3d_input = self.mesh_ROI(features, proposals, images.image_sizes)

        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets)

        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses, pix3d_input, graphs_per_image

        return detections, pix3d_input, graphs_per_image


def pretrained_MaskRcnn(num_classes=10, pretrained=True):
    url = mask_urls['maskrcnn_resnet50_fpn_coco']
    model = Pix3DMask_RCNN(91)
    if pretrained:
        state_dict = load_url(url, progress=True)
        model.load_state_dict(state_dict)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
