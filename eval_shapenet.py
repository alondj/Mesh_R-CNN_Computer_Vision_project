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

from data.dataloader import shapeNet_Dataset, dataLoader
from model import ShapeNetModel, pretrained_ResNet50
from model.loss_functions import batched_mesh_loss, voxel_loss
from utils import f_score
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
model_name = 'ShapeNet'

gpus = [torch.cuda.get_device_name(device) for device in devices]

# model and datasets/loaders definition
num_classes = 13
model = ShapeNetModel(pretrained_ResNet50(nn.functional.nll_loss,
                                          num_classes=num_classes,
                                          pretrained=False),
                      residual=options.residual,
                      cubify_threshold=options.threshold,
                      vertex_feature_dim=options.featDim,
                      num_refinement_stages=options.num_refinement_stages)

classes = None
if options.classes is not None:
    classes = [item for item in options.classes.split(',')]
dataset = shapeNet_Dataset(options.dataRoot, classes=classes)

testloader = dataLoader(dataset, options.batch_size, 48, options.num_workers,
                        test=True, num_train_samples=None, train_ratio=1-options.test_ratio)

# load checkpoint
model.load_state_dict(torch.load(options.model_path))

if len(devices) > 1:
    model = CustomDP(model, is_backbone=False, pix3d=False)

model: nn.Module = model.to(devices[0]).eval()

losses_and_scores = {'chamfer': 0.,
                     'voxel': 0.,
                     'edge': 0.,
                     'normal': 0.,
                     }

confusion_matrix = torch.zeros(num_classes, num_classes)
with torch.no_grad():
    with tqdm.tqdm(total=len(testloader.batch_sampler), file=sys.stdout) as pbar:
        for i, batch in enumerate(testloader, 0):
            batch = batch.to(devices[0])
            images, backbone_targets = batch.images, batch.backbone_targets
            voxel_gts = batch.voxels

            # predict and comput loss
            model_output = model(images)
            vxl_loss = voxel_loss(model_output['voxels'], voxel_gts)
            chamfer_loss, normal_loss, edge_loss = batched_mesh_loss(
                model_output['vertex_positions'],
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
            preds = torch.argmax(model_output['backbone'], 1)
            for p, t in zip(preds, backbone_targets):
                confusion_matrix[p, t] += 1

    # compute final metrics
    f_0_1loss = f_score(confusion_matrix, 0.1).mean().item()
    f_0_3loss = f_score(confusion_matrix, 0.3).mean().item()
    f_0_5loss = f_score(confusion_matrix, 0.5).mean().item()

    avg_chamfer = losses_and_scores['chamfer'] / len(testloader.batch_sampler)
    avg_edge = losses_and_scores['edge'] / len(testloader.batch_sampler)
    avg_voxel = losses_and_scores['voxel'] / len(testloader.batch_sampler)
    avg_normal = losses_and_scores['normal'] / len(testloader.batch_sampler)

    print(f"evaluated {model_name} dataset")
    print(f"avg chamfer loss {avg_chamfer:.2f}")
    print(f"avg edge loss {avg_edge:.2f}")
    print(f"avg voxel loss {avg_voxel:.2f}")
    print(f"avg normal loss {avg_normal:.2f}")
    print(f"avg f0.1 1loss {f_0_1loss:.2f}")
    print(f"avg f0.3 loss {f_0_3loss:.2f}")
    print(f"avg f0.5 1loss {f_0_5loss:.2f}")
