from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.model_zoo import load_url
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.mask_rcnn import model_urls as mask_urls
from torchvision.ops import MultiScaleRoIAlign

from .utils import filter_ROI_input

from .layers import (Cubify, VertixRefinePix3D, VoxelBranch, build_RoI_head)
from .loss_functions import voxel_loss, batched_mesh_loss
from data.dataloader import Batch


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

    def forward(self, images: List[Tensor], targets: Batch = None) -> dict:
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        backbone_targets = targets.backbone_targets if self.training else None
        backbone_out, ROI_features = self.backbone(images,
                                                   backbone_targets)
        # in this part we set the mesh index (how many meshes per image)
        # and filtering roi_input for training
        if self.training:
            # backbone returns loss,(ROI_features,detection)
            if self.backbone.training:
                ROI_features, detections = ROI_features
            else:
                # backbone returns detection,ROI_features
                detections = backbone_out
            ROI_features = filter_ROI_input(backbone_targets, detections,
                                            ROI_features)
            mesh_index = [1 for _ in images]
            detections = None
        else:
            mesh_index = [f.shape[0] for f in ROI_features]
            ROI_features = torch.cat(ROI_features)

        voxelGrid = self.voxelBranch(ROI_features)

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

        vertex_positions0, vertice_index, faces, face_index, adj_index = self.cubify(
            voxelGrid)

        sizes = [i.shape[1:] for i in images]
        vertex_positions1, vertex_features = self.refineStages[0](vertice_index, ROI_features,
                                                                  adj_index, vertex_positions0,
                                                                  sizes, mesh_index=mesh_index)

        vertex_positions = [vertex_positions0, vertex_positions1]

        for stage in self.refineStages[1:]:
            new_positions, vertex_features = stage(vertice_index, ROI_features,
                                                   adj_index, vertex_positions[-1], sizes,
                                                   vertex_features=vertex_features,
                                                   mesh_index=mesh_index)
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
        detections, detector_losses, ROI_features = self.roi_heads(features, proposals,
                                                                   images.image_sizes, targets)

        detections = self.transform.postprocess(detections, images.image_sizes,
                                                original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if self.training:
            return losses, (ROI_features, detections)

        return detections, ROI_features


def pretrained_MaskRcnn(num_classes=10, pretrained=True):
    # when that time comes remove the pretrained arg and set path as deafult model path
    url = mask_urls['maskrcnn_resnet50_fpn_coco']
    model = Pix3DMask_RCNN(91)
    if pretrained:
        model.load_state_dict(load_url(url, progress=True))

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads = build_RoI_head(model.backbone.out_channels, num_classes=num_classes, box_detections_per_img=3,
                                     box_roi_pool=MultiScaleRoIAlign(featmap_names=[0, 1, 2, 3],
                                                                     output_size=12,
                                                                     sampling_ratio=1),
                                     mask_predictor=MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes))

    return model
