import time
import torch
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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
            print(f"meter {self.name} recieved bad update value {val}")

    def __str__(self):
        fmtstr = '{name} latest: {val' + self.fmt + \
            '} average: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\n'.join(entries))
        print()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train_backbone(model, optimizer, dataloader, epoch, is_pix3d=False,
                   lr_count=0, curr_lr=0, print_freq=10):
    assert torch.cuda.is_available(), "gpu is required for training"
    metrics = {'batch_time': AverageMeter('Batch Time', ':6.3f'),
               'data_loading': AverageMeter('Data Loading Time', ':6.3f'),
               'loss_classifier': AverageMeter('Classifier Loss', ':.4e')}
    if is_pix3d:
        metrics.update({'loss_box_reg': AverageMeter('Box Regularization Loss', ':.4e'),
                        'loss_mask': AverageMeter('Mask Loss', ':.4e'),
                        'loss_objectness': AverageMeter('Objectness Loss', ':.4e'),
                        'loss_rpn_box_reg': AverageMeter('RPN Regularization Loss', ':.4e')})

    progress = ProgressMeter(
        len(dataloader),
        list(metrics.values()),
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, batch in enumerate(dataloader):
        # measure data loading time
        metrics['data_loading'].update(time.time() - end)
        batch = batch.to('cuda:0', non_blocking=True)
        images, backbone_targets = batch.images, batch.backbone_targets

        # compute output
        try:
            output = model(images, backbone_targets)[0]
        except Exception as _:
            continue

        # compute loss update metrics
        if is_pix3d:
            loss = sum(output.values())
            for k, l in output.values():
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
                curr_lr += (0.02-0.002)/1000.0
            if lr_count in [8000, 10000]:
                curr_lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

    progress.display(len(dataloader))
    return metrics, lr_count, curr_lr


def train_gcn(model, optimizer, dataloader, epoch, loss_weights, backbone_train=True,
              is_pix3d=False, curr_lr=0, lr_count=0, print_freq=10):
    assert torch.cuda.is_available(), "gpu is required for training"
    metrics = {'batch_time': AverageMeter('Batch Time', ':6.3f'),
               'data_loading': AverageMeter('Data Loading Time', ':6.3f'),
               'voxel_loss': AverageMeter('Voxel Loss', ':.4e'),
               'edge_loss': AverageMeter('Edge Loss', ':.4e'),
               'normal_loss': AverageMeter('Normal Loss', ':.4e'),
               'chamfer_loss': AverageMeter('Chamfer Loss', ':.4e')}

    if backbone_train:
        metrics['loss_classifier'] = AverageMeter('Classifier Loss', ':.4e')

    if is_pix3d and backbone_train:
        metrics.update({'loss_box_reg': AverageMeter('Box Regularization Loss', ':.4e'),
                        'loss_mask': AverageMeter('Mask Loss', ':.4e'),
                        'loss_objectness': AverageMeter('Objectness Loss', ':.4e'),
                        'loss_rpn_box_reg': AverageMeter('RPN Regularization Loss', ':.4e')})

    progress = ProgressMeter(
        len(dataloader),
        list(metrics.values()),
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, batch in enumerate(dataloader):
        # measure data loading time
        metrics['data_loading'].update(time.time() - end)

        batch = batch.to('cuda:0', non_blocking=True)
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
                curr_lr += (0.02-0.002)/1000.0
            if lr_count in [8000, 10000]:
                curr_lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr

    progress.display(len(dataloader))
    return metrics, lr_count, curr_lr
