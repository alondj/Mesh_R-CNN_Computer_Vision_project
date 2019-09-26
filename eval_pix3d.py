import argparse
import sys
from torchvision.ops.boxes import box_iou
# import numpy as np
import torch
import torch.nn as nn
import tqdm


from data.dataloader import (pix3dDataset, pix3dDataLoader)
from model import (Pix3DModel, pretrained_MaskRcnn)
from model.loss_functions import batched_mesh_loss, voxel_loss
from utils import f_score

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
parser.add_argument('--num_sampels', type=int,
                    help='number of sampels to dataset', default=None)

parser.add_argument('-c', '--classes', help='classes of the exampels in the dataset', type=str, default=None)

parser.add_argument('--dataRoot', type=str, help='file root')

parser.add_argument('--batchSize', '-b', type=int,
                    default=16, help='batch size')

parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=4)

options = parser.parse_args()

devices = [torch.device('cuda', i)
           for i in range(torch.cuda.device_count())]

# print header
model_name = options.model

gpus = [torch.cuda.get_device_name(device) for device in devices]

num_classes = 10
model = Pix3DModel(pretrained_MaskRcnn(num_classes=num_classes, pretrained=False),
                   cubify_threshold=options.threshold,
                   vertex_feature_dim=options.featDim,
                   num_refinement_stages=options.num_refinement_stages)

if options.classes is not None:
    classes = [item for item in options.classes.split(',')]
    dataset = pix3dDataset(options.dataRoot, options.num_sampels, classes=classes)
else:
    dataset = pix3dDataset(options.dataRoot, options.num_sampels)

testloader = pix3dDataLoader(dataset, batch_size=options.batchSize, num_voxels=24, num_workers=options.workers)

# load checkpoint
model.load_state_dict(torch.load(options.model_path))

if len(devices) > 1:
    model = nn.DataParallel(model)

model: nn.Module = model.to(devices[0]).eval()

losses_and_scores = {'chamfer': 0.,
                     'voxel': 0.,
                     'edge': 0.,
                     'normal': 0.,
                     'AP_box': 0.,
                     'AP_mask': 0.,
                     'AP_mesh': 0.
                     }


def get_out_of_dicts(bbo):
    boxes, labels, masks = [], [], []
    for dic in bbo:
        boxes.append(dic['boxes'])
        labels.append(dic['labels'][0])
        masks.append(dic['masks'])
    return boxes, labels, masks


def calc_precision_box(boxes, gt_boxes):
    count = 0
    num_sampels = len(boxes)

    for gt_box, pred_box in zip(gt_boxes, boxes):
        if box_iou(gt_boxes, pred_box)[0][0] > 0.5:
            count += 1
    return count / num_sampels


def calc_precision_mask(masks, gt_masks):
    count = 0
    num_sampels = len(masks)

    for mask, gt_mask in zip(masks, gt_masks):
        intersection = mask & gt_mask
        union = mask | gt_mask
        iou_score = torch.sum(intersection) / torch.sum(union)
        if iou_score > 0.5:
            count += 1
    return count / num_sampels


confusion_matrix = torch.zeros(num_classes, num_classes)
with torch.no_grad():
    with tqdm.tqdm(total=len(testloader.batch_sampler), file=sys.stdout) as pbar:
        for i, batch in enumerate(testloader, 0):
            batch = batch.to(devices[0])
            images, backbone_targets = batch.images, batch.targets
            voxel_gts = batch.voxels

            # predict and comput loss
            model_output = model(images, backbone_targets)

            vxl_loss = voxel_loss(model_output['voxels'], voxel_gts)
            chamfer_loss, normal_loss, edge_loss = batched_mesh_loss(
                model_output['vertex_positions'],
                model_output['faces'],
                model_output['edge_index'],
                model_output['vertice_index'],
                model_output['face_index'],
                batch
            )

            # TODO how to get masks from mask_RCNN backbone
            # TODO add AP mesh
            gt_boxes, gt_labels, gt_masks = get_out_of_dicts(backbone_targets)
            boxes, preds, masks = get_out_of_dicts(model_output['backbone'])

            # update losses
            losses_and_scores['chamfer'] += chamfer_loss.item()
            losses_and_scores['voxel'] += vxl_loss.item()
            losses_and_scores['edge'] += edge_loss.item()
            losses_and_scores['normal'] += normal_loss.item()
            losses_and_scores['AP_box'] += calc_precision_box(boxes, gt_boxes)
            losses_and_scores['AP_mask'] += calc_precision_mask(masks=masks, gt_masks=gt_masks)
            # update confusion matrix
            for p, t in zip(preds, gt_labels):
                confusion_matrix[p, t] += 1

    # compute final metrics
    f_0_1loss = f_score(confusion_matrix, 0.1).mean().item()
    f_0_3loss = f_score(confusion_matrix, 0.3).mean().item()
    f_0_5loss = f_score(confusion_matrix, 0.5).mean().item()

    avg_chamfer = losses_and_scores['chamfer'] / len(testloader.batch_sampler)
    avg_edge = losses_and_scores['edge'] / len(testloader.batch_sampler)
    avg_voxel = losses_and_scores['voxel'] / len(testloader.batch_sampler)
    avg_normal = losses_and_scores['normal'] / len(testloader.batch_sampler)
    avg_AP_box = losses_and_scores['AP_box'] / len(testloader.batch_sampler)
    avg_AP_mask = losses_and_scores['AP_mask'] / len(testloader.batch_sampler)

    print(f"evaluated {model_name} dataset")
    print(f"avg precision of boxes:{avg_AP_box}")
    print(f"avg precision of masks:{avg_AP_mask}")
    print(f"avg chamfer loss {avg_chamfer:.2f}")
    print(f"avg edge loss {avg_edge:.2f}")
    print(f"avg voxel loss {avg_voxel:.2f}")
    print(f"avg normal loss {avg_normal:.2f}")
    print(f"avg f0.1 1loss {f_0_1loss:.2f}")
    print(f"avg f0.3 loss {f_0_3loss:.2f}")
    print(f"avg f0.5 1loss {f_0_5loss:.2f}")
