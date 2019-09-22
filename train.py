import argparse
import datetime
import os
import platform
import sys
from itertools import chain
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, random_split

from data.dataloader import pix3dDataset, shapeNet_Dataset, pix3dDataLoader, shapenetDataLoader
from model import (Pix3DModel, ShapeNetModel, pretrained_MaskRcnn,
                   pretrained_ResNet50, total_loss)

assert torch.cuda.is_available(), "the training process is slow and requires gpu"


def test_train_split(ds, train_ratio, batch_size, num_workers):
    train_amount = int(train_ratio * len(ds))
    test_amount = len(ds) - train_amount
    train_set, test_set = random_split(ds, (train_amount, test_amount))
    train_dl = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers)
    test_dl = DataLoader(test_set, batch_size=batch_size,
                         shuffle=False, num_workers=num_workers)
    return train_dl, test_dl


parser = argparse.ArgumentParser()

# model args
parser.add_argument(
    "--model", "-m", help="the model we wish to train", choices=["ShapeNet", "Pix3D"], required=True)
parser.add_argument('--featDim', type=int, default=128,
                    help='number of vertex features')
parser.add_argument(
    "--model_path", help="path of a pretrained model to cintinue training", default='')
parser.add_argument('--num_refinement_stages', "-nr", type=int,
                    default=3, help='number of mesh refinement stages')
parser.add_argument('--threshold', '-th',
                    help='Cubify threshold', type=float, default=0.2)
parser.add_argument("--residual", default=False,
                    action="store_true", help="whether to use residual refinement for ShapeNet")
parser.add_argument("--train_backbone", default=False, action="store_true",
                    help="whether to train the backbone in additon to the GCN")

# loss args
parser.add_argument("--chamfer", help="weight of the chamfer loss",
                    type=float, default=1.0)
parser.add_argument("--voxel", help="weight of the voxel loss",
                    type=float, default=1.0)
parser.add_argument("--normal", help="weight of the normal loss",
                    type=float, default=0)
parser.add_argument("--edge", help="weight of the edge loss",
                    type=float, default=0.5)
parser.add_argument("--backbone", help="weight of the backbone loss",
                    type=float, default=1.0)

# dataset/loader arguments
parser.add_argument('--num_sampels', type=int,
                    help='number of sampels to dataset', default=None)
parser.add_argument('--dataRoot', type=str, help='file root')

parser.add_argument('--batchSize', '-b', type=int,
                    default=16, help='batch size')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=4)

parser.add_argument('--nEpoch', type=int, default=10,
                    help='number of epochs to train for')

# optimizer parameters
parser.add_argument('--optim', type=str,
                    help='optimizer to use', choices=['Adam', 'SGD'])
parser.add_argument('--weightDecay', type=float,
                    default=5e-6, help='weight decay for L2 loss')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

options = parser.parse_args()

epochs = options.nEpoch

devices = [torch.device('cuda', i)
           for i in range(torch.cuda.device_count())]

# print header
model_name = options.model
title = f'{model_name} training \n used_gpus: {len(devices)}\n epochs: {epochs}\n'
print(title)

gpus = [torch.cuda.get_device_name(device) for device in devices]
print('system information\n python: %s, torch: %s, cudnn: %s, cuda: %s, \ngpus: %s' % (
    platform.python_version(),
    torch.__version__,
    torch.backends.cudnn.version(),
    torch.version.cuda,
    gpus))
print("\n")

print(f"options were:\n{options}\n")

pretrained = options.model_path != ''
# model and datasets/loaders definition
if model_name == 'ShapeNet':
    model = ShapeNetModel(pretrained_ResNet50(nn.functional.nll_loss,
                                              num_classes=13,
                                              pretrained=pretrained),
                          residual=options.residual,
                          cubify_threshold=options.threshold,
                          vertex_feature_dim=options.featDim,
                          num_refinement_stages=options.num_refinement_stages)

    dataset = shapeNet_Dataset(options.dataRoot, options.num_sampels)
    trainloader = shapenetDataLoader(
        dataset, batch_size=options.batchSize, num_voxels=48, num_workers=options.workers)
else:
    model = Pix3DModel(pretrained_MaskRcnn(num_classes=10, pretrained=pretrained),
                       cubify_threshold=options.threshold,
                       vertex_feature_dim=options.featDim,
                       num_refinement_stages=options.num_refinement_stages)
    dataset = pix3dDataset(options.dataRoot, options.num_sampels)
    trainloader = pix3dDataLoader(
        dataset, batch_size=options.batchSize, num_voxels=24, num_workers=options.workers)

# load checkpoint if possible
if options.model_path != '':
    model.load_state_dict(torch.load(options.model_path))

# select trainable parameters
trained_parameters = chain(model.refineStages.parameters(),
                           model.voxelBranch.parameters())
model.refineStages.train()
model.voxelBranch.train()
if options.train_backbone:
    trained_parameters = chain(trained_parameters, model.backbone.parameters())
    model.backbone.train()
else:
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()

# use data parallel if possible
# TODO i do not know if it will work for mask rcnn
if len(devices) > 1:
    model = nn.DataParallel(model)

model: nn.Module = model.to(devices[0])

# Create Optimizer
lrate = options.lr
decay = options.weightDecay
if options.optim == 'Adam':
    optimizer = Adam(trained_parameters, lr=lrate, weight_decay=decay)
else:
    optimizer = SGD(trained_parameters, lr=lrate, weight_decay=decay)
    # TODO they increased learning rate do we wish to do the same?
    # linearly increasing the learning rate from 0.002 to 0.02 over the first 1K iterations,

# loss weights
loss_weights = {'c': options.chamfer,
                'v': options.voxel,
                'n': options.normal,
                'e': options.edge,
                'b': options.backbone
                }

# checkpoint directories
now = datetime.datetime.now()
save_path = now.isoformat()
GCN_path = os.path.join('checkpoints', model_name, 'GCN', save_path)
backbone_path = os.path.join('checkpoints', options.model,
                             'backbone', save_path)
if not os.path.exists(GCN_path):
    Path(GCN_path).mkdir(parents=True, exist_ok=True)
if options.train_backbone and not os.path.exists(backbone_path):
    Path(backbone_path).mkdir(parents=True, exist_ok=True)

# Train model on the dataset
losses = []
for epoch in range(epochs):
    epoch_loss = []
    print(f'--- EPOCH {epoch+1}/{epochs} ---')
    with tqdm.tqdm(total=len(trainloader.batch_sampler), file=sys.stdout) as pbar:
        for i, batch in enumerate(trainloader, 0):
            optimizer.zero_grad()
            batch = batch.to(devices[0])
            images, backbone_targets = batch.images, batch.targets
            voxel_gts = batch.voxels
            # predict and comput loss
            output = model(images, backbone_targets)

            loss = total_loss(loss_weights, output,
                              voxel_gts, batch,
                              train_backbone=options.train_backbone,
                              backbone_type=model_name)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            pbar.update()
            avg_loss = np.mean(epoch_loss)

            # prediodic loss updates
            if (i + 1) % 128 == 0:
                print(f"Epoch {epoch+1} batch {i+1}")
                print(f"avg loss for this epoch sor far {avg_loss:.2f}")

    # epoch ended
    losses.append(epoch_loss)
    print(
        f'--- EPOCH {epoch+1}/{epochs} --- avg epoch loss {np.mean(epoch_loss):.2f}')
    print(f"total avg loss so far {np.mean(losses):.2f}")

    # save the model
    print('saving net...')
    # for eg checkpoints/Pix3D/date/model_{1}.pth
    file_name = f"model_{epoch}.pth"
    torch.save(model.state_dict(),
               os.path.join(GCN_path, file_name))

print(f"training done avg loss {np.mean(losses)}")
