import time
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
from typing import List, Dict


def save_state(model: Module, file_name: str):
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    torch.save(state_dict, file_name)


def load_dict(path: str) -> OrderedDict:
    state_dict: OrderedDict = torch.load(path)

    res_dict = OrderedDict()

    for k, v in state_dict.items():
        new_k = k
        if k.startswith('model.'):
            new_k = k[len('model.'):]
        res_dict[new_k] = v

    return res_dict


def safe_print(rank: int, msg: str):
    if rank == 0:
        print(msg)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ':f', rank: int = 0):
        self.name = name
        self.fmt = fmt
        self.rank = rank
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if np.isfinite(val):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
        else:
            safe_print(self.rank,
                       f"meter {self.name} recieved bad update value {val}")

    def __str__(self):
        fmtstr = '{name} latest: {val' + self.fmt + \
            '} average: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = "", rank: int = 0):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.rank = rank

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        safe_print(self.rank, '\n'.join(entries) + '\n')

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def basic_metrics(rank: int = 0) -> Dict[str, AverageMeter]:
    return {'batch_time': AverageMeter('Batch Time', ':6.3f', rank=rank),
            'data_loading': AverageMeter('Data Loading Time', ':6.3f', rank=rank)}


def maskrcnn_metrics(rank: int = 0) -> Dict[str, AverageMeter]:
    return {'loss_box_reg': AverageMeter('Box Regularization Loss', ':.4e', rank=rank),
            'loss_mask': AverageMeter('Mask Loss', ':.4e', rank=rank),
            'loss_objectness': AverageMeter('Objectness Loss', ':.4e', rank=rank),
            'loss_rpn_box_reg': AverageMeter('RPN Regularization Loss', ':.4e', rank=rank)}


def gcn_metrics(rank: int = 0, voxel_only: bool = False) -> Dict[str, AverageMeter]:
    metrics = {'voxel_loss': AverageMeter('Voxel Loss', ':.4e', rank=rank)}
    if not voxel_only:
        metrics.update({'edge_loss': AverageMeter('Edge Loss', ':.4e', rank=rank),
                        'normal_loss': AverageMeter('Normal Loss', ':.4e', rank=rank),
                        'chamfer_loss': AverageMeter('Chamfer Loss', ':.4e', rank=rank)})
    return metrics


def train_backbone(rank: int, model: Module, optimizer: Optimizer, dataloader: DataLoader,
                   epoch: int, is_pix3d: bool = False,
                   lr_count: int = 0, curr_lr: float = 0, print_freq: int = 10):
    assert torch.cuda.is_available(), "gpu is required for training"
    metrics = basic_metrics(rank=rank)
    metrics['loss_classifier'] = AverageMeter('Classifier Loss', ':.4e',
                                              rank=rank)
    if is_pix3d:
        metrics.update(maskrcnn_metrics(rank))

    progress = ProgressMeter(
        len(dataloader),
        list(metrics.values()),
        prefix="Epoch: [{}]".format(epoch),
        rank=rank)

    end = time.time()
    for i, batch in enumerate(dataloader):
        # measure data loading time
        metrics['data_loading'].update(time.time() - end)
        batch = batch.to(rank, non_blocking=True)
        images, backbone_targets = batch.images, batch.backbone_targets

        # compute output
        try:
            output = model(images, backbone_targets)[0]
        except Exception as _:
            continue

        # compute loss update metrics
        if is_pix3d:
            loss = sum(output.values())
            for k, l in output.items():
                metrics[k].update(l.item(), len(images))
        else:
            loss = output
            metrics['loss_classifier'].update(loss.item(), len(images))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        metrics['batch_time'].update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

        # on pix3d linearly increase learning rate
        lr_count += 1
        if is_pix3d:
            if lr_count < 1000:
                curr_lr += (0.02 - 0.002) / 1000.0
            if lr_count in [8000, 10000]:
                curr_lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

    progress.display(len(dataloader))
    return metrics, lr_count, curr_lr


def train_gcn(rank: int, model: Module, optimizer: Optimizer, dataloader: DataLoader,
              epoch: int, loss_weights: Dict[str, float], backbone_train: bool = True,
              is_pix3d: bool = False, curr_lr: float = 0, lr_count: int = 0, print_freq: int = 10):
    assert torch.cuda.is_available(), "gpu is required for training"
    metrics = basic_metrics(rank=rank)

    metrics.update(gcn_metrics(rank=rank, voxel_only=model.voxel_only))

    if backbone_train:
        metrics['loss_classifier'] = AverageMeter(
            'Classifier Loss', ':.4e', rank=rank)

    if is_pix3d and backbone_train:
        metrics.update(maskrcnn_metrics(rank=rank))

    progress = ProgressMeter(
        len(dataloader),
        list(metrics.values()),
        prefix="Epoch: [{}]".format(epoch), rank=rank)

    end = time.time()
    for i, batch in enumerate(dataloader):
        # measure data loading time
        metrics['data_loading'].update(time.time() - end)

        batch = batch.to(rank, non_blocking=True)
        images, targets = batch.images, batch
        # compute output
        try:
            output = model(images, targets)
        except Exception as _:
            continue

        # compute loss update metrics
        loss = torch.zeros(1).cuda()

        for k, v in output.items():
            if k in metrics:
                # gcn losses
                loss += (v * loss_weights[k])
                metrics[k].update(v.item(), len(images))
            elif not is_pix3d:
                # shapenet backbone loss
                loss += (v * loss_weights['backbone_loss'])
                metrics['loss_classifier'].update(v.item(), len(images))
            else:
                # maskrcnn backbone losses
                maskrcnn_loss = torch.zeros(1).cuda()
                for rcnn_l, l in v.items():
                    maskrcnn_loss += l.item()
                    metrics[rcnn_l].update(l, len(images))
                loss += (maskrcnn_loss * loss_weights['backbone_loss'])

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        metrics['batch_time'].update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

        # on pix3d linearly increase learning rate
        lr_count += 1
        if is_pix3d:
            if lr_count < 1000:
                curr_lr += (0.02 - 0.002) / 1000.0
            if lr_count in [8000, 10000]:
                curr_lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

    progress.display(len(dataloader))
    return metrics, lr_count, curr_lr
