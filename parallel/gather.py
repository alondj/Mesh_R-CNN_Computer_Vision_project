import numpy as np
from itertools import chain

from torch.autograd import Function
import torch.cuda.comm as comm
from torch.nn.parallel._functions import Broadcast, Gather


class Reduce(Function):
    @staticmethod
    def forward(ctx, target_gpu, *inputs):
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
        inputs = sorted(inputs, key=lambda i: i.get_device())
        return comm.reduce_add(inputs, destination=target_gpu)

    @staticmethod
    def backward(ctx, gradOutput):
        return (None,) + Broadcast.apply(ctx.target_gpus, gradOutput)


def reduce_add(ts, out_device):
    return Reduce.apply(out_device, *ts)


def gather(ts, out_device, dim=0):
    return Gather.apply(out_device, dim, *ts)


def pix3d_backbone_gather(outs, out_device, train=True):
    # during training reduce list of loss dictionaries to a single dictionary
    # containing sum of losses
    # during eval flatten the list of dictionary outputs
    if train:
        # outs loss,(roi,detections)
        loss = {k: reduce_add([out[0][k] for out in outs], out_device)
                for k in outs[0][0].keys()}

        return loss, None
    else:
        # outs detections,roi
        detections = [{k: v.to(out_device)}
                      for out in outs for d in out[0] for k, v in d.items()]

        return detections, None


def shapenet_backbone_gather(outs, out_device, train=True):
    if train:
            # outs is list of (loss,features)
        loss = reduce_add([out[0] for out in outs], out_device)

        return loss, None
    else:
        # outs is list of (probas,features)
        probas = gather([out[0] for out in outs], out_device)

        return probas, None


def gather_GCN_outputs(outs, out_device, voxel_only=False):
    res = dict()
    res['voxels'] = gather([out['voxels']for out in outs], out_device)
    if voxel_only:
        return res

    # collect each refine stage

    res['vertex_positions'] = [gather([out['vertex_positions'][i]for out in outs], out_device)
                               for i in range(len(outs[0]['vertex_positions']))]

    res['vertice_index'] = list(chain(*[out['vertice_index']
                                        for out in outs]))

    offsets = [np.sum(out['vertice_index']) for out in outs]
    offsets = np.cumsum(offsets)-offsets
    res['edge_index'] = gather([out['edge_index']+off for out, off in zip(outs, offsets)],
                               out_device, dim=1)

    res['face_index'] = list(chain(*[out['face_index']
                                     for out in outs]))

    res['faces'] = gather([out['faces'] for out in outs], out_device)

    res['mesh_index'] = list(chain(*[out['mesh_index']
                                     for out in outs]))
    return res


def shapenet_gather(outs, out_device, voxel_only=False, backbone_train=True, train=True):
    # train=True outs is list of loss dicts reduce to one loss dict
    # train=False outs is dict of various predictions reduce to one dict
    # train_backbone=True reduce losses
    # train_backbone=False and Train append predictions
    res = dict()
    if backbone_train:
        assert train
        backbone_loss = reduce_add([out['backbone_loss'] for out in outs],
                                   out_device)
        res['backbone_loss'] = backbone_loss
        # we pop backbone_loss so that when we reduce gcn_loss we do not overwrite it
        outs[0].pop('backbone_loss')

    if train:
        gcn_losses = {k: reduce_add([out[k] for out in outs], out_device)
                      for k in outs[0]}
        res.update(gcn_losses)
    else:
        assert not backbone_train
        res['backbone'] = gather([out['backbone']for out in outs], out_device)
        gcn_out = gather_GCN_outputs(outs, out_device, voxel_only=voxel_only)
        res.update(gcn_out)

    return res


def pix3d_gather(outs, out_device, voxel_only=False, backbone_train=True, train=True):
    # train=True outs is list of loss dicts reduce to one loss dict
    # train=False outs is a list of dict of various predictions reduce to one dict
    # train_backbone=True reduce losses
    # train_backbone=False and Train append predictions

    res = dict()
    if backbone_train:
        assert train
        backbone_keys = list(outs[0]['backbone_loss'].keys())
        backbone_loss = {k: [] for k in backbone_keys}
        for out in outs:
            for k, v in out['backbone_loss'].items():
                backbone_loss[k].append(v)

        b_loss = {k: reduce_add(ts, out_device)
                  for k, ts in backbone_loss.items()}

        res['backbone_loss'] = b_loss
        # we pop backbone_loss so that when we reduce gcn_loss we do not overwrite it
        outs[0].pop('backbone_loss')

    if train:
        gcn_losses = {k: reduce_add([out[k] for out in outs], out_device)
                      for k in outs[0]}
        res.update(gcn_losses)
    else:
        assert not backbone_train
        detections = [{k: v.to(out_device) for k, v in d.items()}
                      for out in outs for d in out['backbone']]

        res['backbone'] = detections

        gcn_out = gather_GCN_outputs(outs, out_device, voxel_only=voxel_only)
        res.update(gcn_out)

    return res
