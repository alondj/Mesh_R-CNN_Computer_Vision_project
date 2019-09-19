import argparse
import datetime
import os
import platform
import sys
from itertools import chain

# import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch import Tensor
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, random_split

from data.dataloader import (pix3dDataset, pix3DTarget, pix3DTargetList,
                             shapeNet_Dataset, pix3dDataLoader, shapenetDataLoader)
from model import (Pix3DModel, ShapeNetModel, pretrained_MaskRcnn,
                   pretrained_ResNet50)
from model.loss_functions import batched_mesh_loss, voxel_loss
from utils import f_score

assert torch.cuda.is_available(), "the training process is slow and requires gpu"


parser = argparse.ArgumentParser()

# model args
parser.add_argument(
    "--model", "-m", help="the model we wish to evaluate", choices=["ShapeNet", "Pix3D"], required=True)
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
parser.add_argument('--num_samples', type=int,
                    help='number of sampels to dataset', default=None)
parser.add_argument('--dataRoot', type=str, help='file root')

parser.add_argument('--batchSize', '-b', type=int,
                    defaults=16, help='batch size')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=4)

options = parser.parse_args()

devices = [torch.device('cuda', i)
           for i in range(torch.cuda.device_count())]

# print header
model_name = options.model

gpus = [torch.cuda.get_device_name(device) for device in devices]

# model and datasets/loaders definition
if model_name == 'ShapeNet':
    num_classes = 13
    model = ShapeNetModel(pretrained_ResNet50(nn.functional.nll_loss,
                                              num_classes=13,
                                              pretrained=True),
                          residual=options.residual,
                          cubify_threshold=options.threshold,
                          vertex_feature_dim=options.featDim,
                          num_refinement_stages=options.num_refinement_stages)

    dataset = shapeNet_Dataset(options.dataRoot, options.num_sampels)
    trainloader = shapenetDataLoader(
        dataset, batch_size=options.batchSize, num_voxels=48, num_workers=options.workers)
else:
    model = Pix3DModel(pretrained_MaskRcnn(num_classes=10, pretrained=True),
                       cubify_threshold=options.threshold,
                       vertex_feature_dim=options.featDim,
                       num_refinement_stages=options.num_refinement_stages)
    dataset = pix3dDataset(options.dataRoot, options.num_sampels)
    testLoader = pix3dDataLoader(
        dataset, batch_size=options.batchSize, num_voxels=24, num_workers=options.workers)
    num_classes = 10
# load checkpoint
model.load_state_dict(torch.load(options.model_path))


# use data parallel if possible
# TODO i do not know if it will work for mask rcnn
if len(devices > 1):
    model = nn.DataParallel(model)

model: nn.Module = model.to(devices[0]).eval()


# evaluate model on the dataset

losses_and_scores = {'chamfer': 0.,
                     'voxel': 0.,
                     'edge': 0.,
                     'normal': 0.,
                     }


# TODO how to pix3d?
def get_labels(targets):
    if model_name == 'ShapeNet':
        return targets
    assert isinstance(targets, pix3DTargetList)

    return [t['']]


confusion_matrix = torch.zeros(num_classes, num_classes)
with torch.no_grad():
    with tqdm.tqdm(total=len(testLoader.batch_sampler), file=sys.stdout) as pbar:
        for i, batch in enumerate(testLoader, 0):
            batch = batch.to(devices[0])
            images, backbone_targets = batch.images, batch.targets
            voxel_gts = batch.voxels

            # predict and comput loss
            model_output = model(images, backbone_targets)
            vxl_loss = voxel_loss(model_output['voxels'], voxel_gts)
            chamfer_loss, normal_loss, edge_loss = batched_mesh_loss(
                model_output['vertex_postions'],
                model_output['faces'],
                model_output['edge_index'],
                model_output['vertice_index'],
                model_output['face_index'],
                batch
            )
            # update losses
            losses_and_scores['chamfer'] += chamfer_loss.item()
            losses_and_scores['voxel'] += vxl_loss.item()
            losses_and_scores['edge'] += edge_loss.item()
            losses_and_scores['normal'] += normal_loss.item()

            # update confusion matrix
            # TODO how to handle pix3d
            preds = torch.argmax(model_output['preds'], 1)
            for p, t in zip(preds, get_labels(backbone_targets)):
                confusion_matrix[p, t] += 1

    # compute final metrics
    f_0_1loss = f_score(confusion_matrix, 0.1).mean().item()
    f_0_3loss = f_score(confusion_matrix, 0.3).mean().item()
    f_0_5loss = f_score(confusion_matrix, 0.5).mean().item()

    avg_chamfer = losses_and_scores['chamfer'] / len(testLoader.batch_sampler)
    avg_edge = losses_and_scores['edge']/len(testLoader.batch_sampler)
    avg_voxel = losses_and_scores['voxel']/len(testLoader.batch_sampler)
    avg_normal = losses_and_scores['normal']/len(testLoader.batch_sampler)

    print(f"evaluated {model_name} dataset")
    print(f"avg chamfer loss {avg_chamfer:.2f}")
    print(f"avg edge loss {avg_edge:.2f}")
    print(f"avg voxel loss {avg_voxel:.2f}")
    print(f"avg normal loss {avg_normal:.2f}")
    print(f"avg f0.1 1loss {f_0_1loss:.2f}")
    print(f"avg f0.3 loss {f_0_3loss:.2f}")
    print(f"avg f0.5 1loss {f_0_5loss:.2f}")
