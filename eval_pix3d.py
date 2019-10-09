import argparse
import sys
import torch
import torch.nn as nn
import tqdm
from data.dataloader import (pix3dDataset, dataLoader)
from model import (Pix3DModel, pretrained_MaskRcnn)
from model.loss_functions import batched_mesh_loss, voxel_loss
from utils.metrics import f_score, calc_precision_box, calc_precision_mask, mesh_precision_recall
from torchvision.ops.boxes import box_iou
import numpy as np
import copy
from parallel import CustomDP

assert torch.cuda.is_available(), "the training process is slow and requires gpu"

parser = argparse.ArgumentParser()

# model args
parser.add_argument('--featDim', type=int, default=128,
                    help='number of vertex features')
parser.add_argument("--model_path",
                    help="the path to the model we wish to evaluate", type=str)
parser.add_argument('--num_refinement_stages', "-nr", type=int,
                    default=3, help='number of mesh refinement stages')
parser.add_argument('--threshold', '-th',
                    help='Cubify threshold', type=float, default=0.2)
parser.add_argument("--residual", default=False,
                    action="store_true", help="whether to use residual refinement for ShapeNet")

# dataset/loader arguments
parser.add_argument('--test_ratio', type=float,
                    help='ratio of samples to test', default=1.)

parser.add_argument('-c', '--classes',
                    help='classes of the exampels in the dataset', type=str, default=None)

parser.add_argument('--dataRoot', type=str, help='file root')

parser.add_argument('--batchSize', '-b', type=int,
                    default=16, help='batch size')

parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=4)

options = parser.parse_args()

devices = [torch.device('cuda', i)
           for i in range(torch.cuda.device_count())]

# print header
model_name = 'Pix3D'

gpus = [torch.cuda.get_device_name(device) for device in devices]

num_classes = 10
model = Pix3DModel(pretrained_MaskRcnn(num_classes=num_classes, pretrained=False),
                   cubify_threshold=options.threshold,
                   vertex_feature_dim=options.featDim,
                   num_refinement_stages=options.num_refinement_stages)

classes = None
if options.classes is not None:
    classes = [item for item in options.classes.split(',')]
dataset = pix3dDataset(options.dataRoot, classes=classes)

testloader = dataLoader(dataset, options.batch_size, 24, options.num_workers,
                        test=True, num_train_samples=None, train_ratio=1 - options.train_ratio)

# load checkpoint
model.load_state_dict(torch.load(options.model_path))

if len(devices) > 1:
    model = CustomDP(model, is_backbone=False, pix3d=True)

model: nn.Module = model.to(devices[0]).eval()

losses_and_scores = {'chamfer': 0.,
                     'voxel': 0.,
                     'edge': 0.,
                     'normal': 0.,
                     'AP_box': 0.,
                     'AP_mask': 0.,
                     }


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


confusion_matrix = torch.zeros(num_classes, num_classes)
with torch.no_grad():
    with tqdm.tqdm(total=len(testloader.batch_sampler), file=sys.stdout) as pbar:
        for i, batch in enumerate(testloader, 0):
            batch = batch.to(devices[0])
            images, backbone_targets = batch.images, batch.backbone_targets
            voxel_gts = batch.voxels

            gt_boxes, gt_labels, gt_masks, _ = get_out_of_dicts(backbone_targets,
                                                                gt_bbox=None)

            # predict and comput loss
            model_output = model(images)

            boxes, preds, masks, max_indexes = get_out_of_dicts(model_output['backbone'],
                                                                gt_bbox=gt_boxes)

            pred = get_only_max(max_indexes, model_output['voxels'],
                                model_output['vertex_positions'],
                                model_output['faces'], model_output['vertice_index'],
                                model_output['face_index'])

            voxels, vertex_positions, faces, edge_index, vertice_index, face_index = pred

            vxl_loss = voxel_loss(voxels, voxel_gts)
            chamfer_loss, normal_loss, edge_loss = batched_mesh_loss(
                vertex_positions,
                faces,
                edge_index,
                vertice_index,
                face_index,
                batch
            )

            # update losses
            losses_and_scores['chamfer'] += chamfer_loss.item()
            losses_and_scores['voxel'] += vxl_loss.item()
            losses_and_scores['edge'] += edge_loss.item()
            losses_and_scores['normal'] += normal_loss.item()
            losses_and_scores['AP_box'] += calc_precision_box(boxes, gt_boxes)
            losses_and_scores['AP_mask'] += calc_precision_mask(masks=masks,
                                                                gt_masks=gt_masks)
            # update confusion matrix
            for p, t in zip(preds, gt_labels):
                confusion_matrix[p, t] += 1

    # compute final metrics
    f1_0_3_score = f_score(confusion_matrix, 0.3)
    AP_mesh = mesh_precision_recall(confusion_matrix, f1_0_3_score)

    avg_chamfer = losses_and_scores['chamfer'] / len(testloader.batch_sampler)
    avg_edge = losses_and_scores['edge'] / len(testloader.batch_sampler)
    avg_voxel = losses_and_scores['voxel'] / len(testloader.batch_sampler)
    avg_normal = losses_and_scores['normal'] / len(testloader.batch_sampler)
    avg_AP_box = losses_and_scores['AP_box'] / len(testloader.batch_sampler)
    avg_AP_mask = losses_and_scores['AP_mask'] / len(testloader.batch_sampler)

    print(f"evaluated {model_name} dataset")
    print(f"avg precision of boxes:{avg_AP_box}")
    print(f"avg precision of masks:{avg_AP_mask}")
    print(f"avg precision of mesh {AP_mesh}")
    print(f"avg chamfer loss {avg_chamfer:.2f}")
    print(f"avg edge loss {avg_edge:.2f}")
    print(f"avg voxel loss {avg_voxel:.2f}")
    print(f"avg normal loss {avg_normal:.2f}")
