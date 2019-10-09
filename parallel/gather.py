import numpy as np
from torch.cuda.comm import reduce_add, gather
from itertools import chain


def pix3d_backbone_gather(outs, out_device, train=True):
    # during training reduce list of loss dictionaries to a single dictionary
    # containing sum of losses
    # during eval flatten the list of dictionary outputs
    if train:
        # outs loss,(roi,detections)
        loss = {k: reduce_add([out[0][k] for out in outs],
                              destination=out_device) for k in outs[0]}
        return loss, None
    else:
        # outs detections,roi
        detections = [{k: v.to(out_device) for k, v in d.items()}
                      for out_list in outs for d in out_list[0]]

        return detections, None


def shapenet_backbone_gather(outs, out_device, train=True):
    if train:
        # outs is list of (loss,features)
        loss = reduce_add([out[0] for out in outs], destination=out_device)

        return loss, None
    else:
        # outs is list of (probas,features)
        probas = gather([out[0] for out in outs], destination=out_device)

        return probas, None


def gather_GCN_outputs(outs, out_device, voxel_only=False):
    res = dict()
    res['voxels'] = gather([out['voxels']for out in outs],
                           destination=out_device)
    if voxel_only:
        return res

    res['vertex_positions'] = gather([out['vertex_positions'] for out in outs],
                                     dim=0, destination=out_device)

    res['vertice_index'] = list(chain(*[out['vertice_index']
                                        for out in outs]))

    offsets = [np.sum(out['vertice_index']) for out in outs]
    offsets = np.cumsum(offsets)-offsets
    res['edge_index'] = gather([out['edge_index']+off for out, off in zip(outs, offsets)],
                               destination=out_device)

    res['face_index'] = list(chain(*[out['face_index']
                                     for out in outs]))

    res['faces'] = gather([out['faces'] for out in outs],
                          dim=0, destination=out_device)

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
                                   destination=out_device)
        res['backbone_loss'] = backbone_loss
        # we pop backbone_loss so that when we reduce gcn_loss we do not overwrite it
        outs[0].pop('backbone_loss')

    if train:
        gcn_losses = {k: reduce_add([out[k] for out in outs],
                                    destination=out_device) for k in outs[0]}
        res.update(gcn_losses)
    else:
        assert not backbone_train
        res['backbone'] = gather([out['backbone']for out in outs],
                                 destination=out_device)
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
        backbone_loss = {k: reduce_add([out['backbone_loss'][k] for out in outs],
                                       destination=out_device) for k in outs[0]}
        res['backbone_loss'] = backbone_loss
        # we pop backbone_loss so that when we reduce gcn_loss we do not overwrite it
        outs[0].pop('backbone_loss')

    if train:
        gcn_losses = {k: reduce_add([out[k] for out in outs],
                                    destination=out_device) for k in outs[0]}
        res.update(gcn_losses)
    else:
        assert not backbone_train
        detections = [{k: v.to(out_device) for k, v in d.items()}
                      for out in outs for d in out['backbone']]

        res['backbone'] = detections

        gcn_out = gather_GCN_outputs(outs, out_device, voxel_only=voxel_only)
        res.update(gcn_out)

    return res
