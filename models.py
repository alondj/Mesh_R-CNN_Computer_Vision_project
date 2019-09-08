from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from layers import VoxelBranch, Cubify, VertixRefinePix3D, VertixRefineShapeNet,\
    ResVertixRefineShapenet
from typing import Tuple
# MaskRCNN FasterRCNN, GeneralizedRCNN, RoIHeads MultiScaleRoIAlign RoIAlign
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign, RoIAlign
from collections import OrderedDict
from torch.utils.model_zoo import load_url


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

    def forward(self, img: Tensor) -> dict:
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
        output['backbone'] = img_feature_maps

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

    def forward(self, image: Tensor) -> dict:
        backbone_out, roiAlign = self.feature_extractor(image)

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

        return output


class ShapeNetFeatureExtractor(nn.Module):
    ''' ShapeNetFeatureExtractor is a fully convolutional network based on the VGG16 architecture emmitting 4 feature maps\n
        given an input of shape NxCxHxW and filters=f the 4 feature maps are of shapes:\n
        Nx(4f)xH/4xW/4 , Nx(8f)xH/8xW/8 , Nx(16f)xH/16xW/16 , Nx(32f)xH/32xW/32
    '''

    def __init__(self, in_channels: int, filters: int = 64):
        super(ShapeNetFeatureExtractor, self).__init__()
        self.conv0_1 = nn.Conv2d(in_channels, filters, 3, stride=1, padding=1)
        self.conv0_2 = nn.Conv2d(filters, filters, 3, stride=1, padding=1)

        # cut h and w by half
        self.conv1_1 = nn.Conv2d(filters, 2*filters, 3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(2*filters, 2*filters, 3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(2*filters, 2*filters, 3, stride=1, padding=1)

        # cut h and w by half
        self.conv2_1 = nn.Conv2d(2*filters, 4*filters, 3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(4*filters, 4*filters, 3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(4*filters, 4*filters, 3, stride=1, padding=1)

        # cut h and w by half
        self.conv3_1 = nn.Conv2d(4*filters, 8*filters, 3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(8*filters, 8*filters, 3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(8*filters, 8*filters, 3, stride=1, padding=1)

        # cut h and w by half
        self.conv4_1 = nn.Conv2d(8*filters, 16*filters, 5, stride=2, padding=2)
        self.conv4_2 = nn.Conv2d(16*filters, 16*filters,
                                 3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(16*filters, 16*filters,
                                 3, stride=1, padding=1)

        # cut h and w by half
        self.conv5_1 = nn.Conv2d(16*filters, 32*filters,
                                 5, stride=2, padding=2)
        self.conv5_2 = nn.Conv2d(32*filters, 32*filters,
                                 3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(32*filters, 32*filters,
                                 3, stride=1, padding=1)
        self.conv5_4 = nn.Conv2d(32*filters, 32*filters,
                                 3, stride=1, padding=1)

    def forward(self, img) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        img = F.relu(self.conv0_1(img))
        img = F.relu(self.conv0_2(img))

        img = F.relu(self.conv1_1(img))
        img = F.relu(self.conv1_2(img))
        img = F.relu(self.conv1_3(img))

        img = F.relu(self.conv2_1(img))
        img = F.relu(self.conv2_2(img))
        img = F.relu(self.conv2_3(img))
        img0 = img

        img = F.relu(self.conv3_1(img))
        img = F.relu(self.conv3_2(img))
        img = F.relu(self.conv3_3(img))
        img1 = img

        img = F.relu(self.conv4_1(img))
        img = F.relu(self.conv4_2(img))
        img = F.relu(self.conv4_3(img))
        img2 = img

        img = F.relu(self.conv5_1(img))
        img = F.relu(self.conv5_2(img))
        img = F.relu(self.conv5_3(img))
        img = F.relu(self.conv5_4(img))
        img3 = img

        return img0, img1, img2, img3


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

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
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

        pix3d_input = self.mesh_ROI(features, proposals, images.image_sizes)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets)

        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses, pix3d_input

        return detections, pix3d_input


def pretrained_MaskRcnn(num_classes=100, pretrained=True):
    url = 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'
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
    print("done")
    return model


if __name__ == "__main__":
    model = pretrained_MaskRcnn()
