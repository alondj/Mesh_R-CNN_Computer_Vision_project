
import time
import torch
import numpy as np
from torchvision.ops.boxes import box_iou
from meshRCNN.loss_functions import voxel_loss, batched_mesh_loss
from utils.train_utils import gcn_metrics, AverageMeter, ProgressMeter, safe_print, basic_metrics
from utils.metrics import f_score, calc_precision_box, calc_precision_mask, mesh_precision_recall


def get_max_box(boxes, gt_box):
    iou = box_iou(boxes, gt_box)
    max_idx = torch.argmax(iou, dim=0)[0]
    return boxes[max_idx], max_idx


def extract_pix3d_gts(gts):
    boxes, labels, masks = [], [], []
    for gt in gts:
        boxes.append(gt['boxes'])
        labels.append(gt['labels'][0])
        masks.append(gt['masks'])

    return boxes, labels, masks


def get_out_of_dicts(predictions, gt_boxes):
    boxes, labels, masks, max_indexes = [], [], [], []

    for prediction, gt_box in zip(predictions, gt_boxes):
        max_box, max_idx = get_max_box(prediction['boxes'], gt_box)
        max_indexes.append(max_idx)
        boxes.append(max_box)
        labels.append(prediction['labels'][max_idx])
        masks.append(prediction['masks'][max_idx])

    return boxes, labels, masks, max_indexes


def get_only_max(max_indexes, voxels, vertex_positions, faces, vertice_index, face_index, mesh_index):
    # take voxels
    vxls = [g[idx] for g, idx in zip(voxels.split(mesh_index), max_indexes)]
    vxls = torch.stack(vxls)
    assert len(vxls) == len(mesh_index)
    fs = faces.split(face_index)
    i = 0
    res_fs = []
    res_f_index = []
    # take faces
    for n, idx in zip(mesh_index, max_indexes):
        res_fs.append(fs[i:i + n][idx])
        i += n
        res_f_index.append(res_fs[-1].size(0))
    assert len(res_fs) == len(mesh_index)

    # take vertices and vertice_index
    res_vs = []
    res_vs_index = []
    for j, stage in enumerate(vertex_positions):
        i = 0
        stage_vs = []
        vs = stage.split(vertice_index)
        for n, idx in zip(mesh_index, max_indexes):
            stage_vs.append(vs[i:i + n][idx])
            if j == 0:
                res_vs_index.append(stage_vs[-1].size(0))
            i += n
        assert len(stage_vs) == len(mesh_index)
        res_vs.append(torch.cat(stage_vs))

    assert len(res_vs_index) == len(mesh_index)

    # create_adj matrix
    offsets = np.cumsum(res_vs_index) - res_vs_index
    tmp_fs = [f + offset for f, offset in zip(res_fs, offsets)]
    res_fs = torch.cat(res_fs)

    # create adj_matrix
    faces_t = torch.cat(tmp_fs).t()
    # get all directed edges
    idx_i, idx_j = torch.cat(
        [faces_t[:2], faces_t[1:], faces_t[::2]], dim=1)

    # duplicate to get undirected edges
    idx_i, idx_j = torch.cat([idx_i, idx_j], dim=0), torch.cat(
        [idx_j, idx_i], dim=0)

    adj_index = torch.stack([idx_i, idx_j], dim=0).unique(dim=1)

    return vxls, res_vs, res_fs, adj_index, res_vs_index, res_f_index


def validate(rank, model, val_loader, num_classes, is_pix3d=False, print_freq=10):
    metrics = basic_metrics(rank=rank)
    metrics.update(gcn_metrics(rank=rank))
    if is_pix3d:
        metrics['AP_mask'] = AverageMeter('Average Precision Mask',
                                          ":4e", rank=rank)
        metrics['AP_box'] = AverageMeter('Average Precision Box',
                                         ":4e", rank=rank)
        metrics['AP_mesh'] = AverageMeter('Average Precision Mesh',
                                          ":4e", rank=rank)

    metrics['f0_1'] = AverageMeter('F0.1 Loss', ":4e", rank=rank)
    metrics['f0_3'] = AverageMeter('F0.3 Loss', ":4e", rank=rank)
    metrics['f0_5'] = AverageMeter('F0.5 Loss', ":4e", rank=rank)

    progress = ProgressMeter(
        len(val_loader),
        list(metrics.values()),
        prefix='Test: ', rank=rank)

    # switch to evaluate mode
    model.eval()

    safe_print(rank, "validating model")
    # evaluate dataset
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            metrics['data_loading'].update(time.time() - end)

            batch = batch.to(rank, non_blocking=True)
            images, targets = batch.images, batch.backbone_targets

            voxel_gts = batch.voxels

            # predict
            model_output = model(images)

            # update backbone metrics
            if is_pix3d:
                gt_boxes, gt_labels, gt_masks = extract_pix3d_gts(targets)

                boxes, preds, masks, max_indexes = get_out_of_dicts(model_output['backbone'],
                                                                    gt_boxes)

                pred = get_only_max(max_indexes, model_output['voxels'],
                                    model_output['vertex_positions'],
                                    model_output['faces'], model_output['vertice_index'],
                                    model_output['face_index'], model_output['mesh_index'])

                metrics['AP_box'].update(calc_precision_box(boxes,
                                                            gt_boxes), n=1)
                metrics['AP_mask'].update(calc_precision_mask(masks=masks,
                                                              gt_masks=gt_masks), n=1)
                voxels, vs, fs, e_index, v_index, f_index = pred
            else:
                voxels = model_output['voxels']
                vs = model_output['vertex_positions']
                fs = model_output['faces']
                e_index = model_output['edge_index']
                v_index = model_output['vertice_index']
                f_index = model_output['face_index']

                gt_labels = targets
                preds = model_output['backbone']
                preds = torch.argmax(preds, dim=1)

            # compute gcn loss
            vxl_loss = voxel_loss(voxels, voxel_gts)
            chamfer_loss, normal_loss, edge_loss = batched_mesh_loss(vs, fs, e_index,
                                                                     v_index, f_index, batch)

            # update gcn metrics
            metrics['voxel_loss'].update(vxl_loss.item(), images.size(0))
            metrics['chamfer_loss'].update(chamfer_loss.item(), images.size(0))
            metrics['edge_loss'].update(edge_loss.item(), images.size(0))
            metrics['normal_loss'].update(normal_loss.item(), images.size(0))

            # update confusion matrix
            for p, t in zip(preds, gt_labels):
                confusion_matrix[p, t] += 1

            # update f scores
            for f in [1, 3, 5]:
                metrics[f'f0_{f}'].update(f_score(confusion_matrix,
                                                  f / 10).mean().item())

            if is_pix3d:
                f1_0_3_score = metrics['f0_3'].val
                metrics['AP_mesh'].update(mesh_precision_recall(confusion_matrix,
                                                                f1_0_3_score))

            # measure elapsed time
            metrics['batch_time'].update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

    progress.display(len(val_loader))
    return metrics
