import torch
import torch.nn as nn
from parallel.scatter import custom_scatter
from parallel.gather import pix3d_backbone_gather, pix3d_gather, shapenet_backbone_gather, shapenet_gather
from data.dataloader import Batch
from itertools import chain
from torchvision.models.detection._utils


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
        # determine scatter gather functions
        if is_backbone:
            if pix3d:
                self.gather = pix3d_backbone_gather
            else:
                self.gather = shapenet_backbone_gather
        else:
            if pix3d:
                self.gather = pix3d_gather
            else:
                self.gather = shapenet_gather

        self.scatter = custom_scatter

    def forward(self, images, targets: Batch = None):
        inputs = self.scatter(images, targets, self.device_ids)

        replicas = nn.parallel.replicate(self.model,
                                         self.device_ids[:len(inputs)])
        for r, i in zip(replicas, inputs):
            device = list(r.parameters())[0].device

            for t in chain(r.parameters(), r.buffers()):
                assert t.device == device

            for im in i[0]:
                assert im.device == device

        outputs = nn.parallel.parallel_apply(replicas, inputs)
        if self.is_backbone:
            return self.gather(outputs, self.output_device, train=self.model.training)

        return self.gather(outputs, self.output_device, voxel_only=self.model.voxel_only,
                           backbone_train=self.model.backbone.training, train=self.model.training)
