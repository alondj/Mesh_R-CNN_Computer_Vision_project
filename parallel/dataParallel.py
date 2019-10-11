import torch
import torch.nn as nn
from parallel.scatter import custom_scatter
from parallel.gather import pix3d_backbone_gather, pix3d_gather, shapenet_backbone_gather, shapenet_gather
from data.dataloader import Batch
from parallel.replicate import replicate


class CustomDP(nn.Module):
    def __init__(self, module: nn.Module, is_backbone=False, pix3d=False,
                 device_ids=None, output_device=None):
        super(CustomDP, self).__init__()
        self.module = module
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
        self.is_pix3d = pix3d

    def forward(self, images, targets: Batch = None):
        inputs = custom_scatter(images, targets, self.device_ids,
                                pix3d=self.is_pix3d)

        replicas = replicate(self.module, self.device_ids[:len(inputs)])
        outputs = nn.parallel.parallel_apply(replicas, inputs)

        if self.is_backbone:
            if self.is_pix3d:
                return pix3d_backbone_gather(outputs, self.output_device, train=self.module.training)
            else:
                return shapenet_backbone_gather(outputs, self.output_device, train=self.module.training)

        if self.is_pix3d:
            return pix3d_gather(outputs, self.output_device, voxel_only=self.module.voxel_only,
                                backbone_train=self.module.backbone.training, train=self.module.training)

        return shapenet_gather(outputs, self.output_device, voxel_only=self.module.voxel_only,
                               backbone_train=self.module.backbone.training, train=self.module.training)
