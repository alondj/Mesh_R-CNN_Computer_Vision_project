import torch
import torch.nn as nn
from parallel.scatter import custom_scatter
from parallel.gather import pix3d_backbone_gather, pix3d_gather, shapenet_backbone_gather, shapenet_gather
from data.dataloader import Batch
from parallel.replicate import replicate


class CustomDP(nn.Module):
    def __init__(self, model: nn.Module, is_backbone=False, pix3d=False,
                 device_ids=None, output_device=None):
        super(CustomDP, self).__init__()
        self.model = model
        assert torch.cuda.is_available(), "dataParallel requires GPUS"
        if device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))
        else:
            self.device_ids = device_ids

        if output_device is None:
            self.output_device = self.device_ids[0]
        else:
            self.output_device = output_device

        self.is_backbone = is_backbone

    def forward(self, images, targets: Batch = None):
        inputs = custom_scatter(images, targets, self.device_ids)

        replicas = replicate(self.model, self.device_ids[:len(inputs)])
        outputs = nn.parallel.parallel_apply(replicas, inputs)

        if self.is_backbone:
            if self.pix3d:
                return pix3d_backbone_gather(outputs, self.output_device, train=self.model.training)
            else:
                return shapenet_backbone_gather(outputs, self.output_device, train=self.model.training)

        if self.is_pix3d:
            return pix3d_gather(outputs, self.output_device, voxel_only=self.model.voxel_only,
                                backbone_train=self.model.backbone.training, train=self.model.training)

        return shapenet_gather(outputs, self.output_device, voxel_only=self.model.voxel_only,
                               backbone_train=self.model.backbone.training, train=self.model.training)
