
import time
import torch
import numpy as np
from torchvision.ops.boxes import box_iou
from model.loss_functions import voxel_loss, batched_mesh_loss
from utils.train_utils import gcn_metrics, AverageMeter, ProgressMeter, safe_print, basic_metrics
from utils.metrics import f_score, calc_precision_box, calc_precision_mask, mesh_precision_recall


def get_max_box(boxes, gt_box):
    iou = box_iou(boxes, gt_box)
    max_idx = torch.argmax(iou, dim=0)[0]
    return boxes[max_idx], max_idx


def get_out_of_dicts(bbo, gt_bbox=None):
    boxes, labels, masks, max_indexes = [], [], [], []
    if gt_bbox is None:
        for dic in bbo:
            boxes.append(dic['boxes'])
            labels.append(dic['labels'][0])
            masks.append(dic['masks'])

    else:
        for dic, gt_box in zip(bbo, gt_bbox):
            max_box, max_idx = get_max_box(dic['boxes'], gt_box)
            max_indexes.append(max_idx)
            boxes.append(max_box)
            labels.append(dic['labels'][max_idx][0])
            masks.append(dic['masks'][max_idx])

    return boxes, labels, masks, max_indexes


def get_only_max(max_indexes, voxels, vertex_positions, faces, vertice_index, face_index):
    new_max_indexes_lst = []
    for i in range(0, 3 * len(max_indexes), step=3):
        new_max_indexes_lst.append(i + max_indexes[i])

    new_max_indexes = torch.Tensor(new_max_indexes_lst).type(torch.LongTensor)

    old_offset = np.cumsum(vertice_index) - vertice_index

    vertex_positions_return = []

    voxels = voxels[new_max_indexes]

    for stage in vertex_positions:
        stage = stage.split(vertice_index)
        filtered_stage = torch.cat([stage[idx] for idx in new_max_indexes_lst])
        vertex_positions_return.append(filtered_stage)

    faces = faces.split(face_index)
    faces = [f + off for f, off in zip(faces, old_offset)]
    filtered_faces = [faces[idx] for idx in new_max_indexes_lst]
    faces = torch.cat(filtered_faces)

    vertice_index = [vertice_index[idx] for idx in new_max_indexes_lst]
    face_index = [face_index[idx] for idx in new_max_indexes_lst]

    faces_t = faces.t()
    idx_i, idx_j = torch.cat([faces_t[:2], faces_t[1:], faces_t[::2]], dim=1)
    idx_i, idx_j = torch.cat([idx_i, idx_j], dim=0), torch.cat(
        [idx_j, idx_i], dim=0)
    adj_index = torch.stack([idx_i, idx_j], dim=0).unique(dim=1)

    new_offset = np.cumsum(vertice_index) - vertice_index
    faces = torch.cat(
        [f - off for f, off in zip(faces.split(face_index), new_offset)])

    return voxels, vertex_positions_return, faces, adj_index, vertice_index, face_index


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
                gt_boxes, gt_labels, gt_masks, _ = get_out_of_dicts(targets,
                                                                    gt_bbox=None)

                boxes, preds, masks, max_indexes = get_out_of_dicts(model_output['backbone'],
                                                                    gt_bbox=gt_boxes)

                pred = get_only_max(max_indexes, model_output['voxels'],
                                    model_output['vertex_positions'],
                                    model_output['faces'], model_output['vertice_index'],
                                    model_output['face_index'])

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
                preds = torch.argmax(preds, gt_labels)

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
                                                  f/10).mean().item())

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
