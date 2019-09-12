import argparse
import platform
import sys
import os
import datetime

import torch
import torch.nn as nn
import tqdm
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import numpy as np
from dataloader import pix3dDataset, shapeNet_Dataset
from loss_functions import total_loss
from models import (Pix3DModel, ShapeNetFeatureExtractor, ShapeNetModel,
                    pretrained_MaskRcnn)

assert torch.cuda.is_available(), "the training process is slow and requires gpu"

parser = argparse.ArgumentParser()

# model args
parser.add_argument(
    "--model", "-m", help="the model we wish to train", choices=["ShapeNet", "Pix3D"], required=True)
parser.add_argument('--featDim', type=int, default=128,
                    help='number of vertex features')
parser.add_argument('--num_refinement_stages', "-nr", type=int,
                    default=3, help='number of mesh refinement stages')
parser.add_argument('--threshold', '-th',
                    help='Cubify threshold', type=float, default=0.2)
parser.add_argument("--residual", default=False,
                    action="store_true", help="whether to use residual refinement for ShapeNet")
# loss args
parser.add_argument("--chamfer", help="weight of the chamfer loss",
                    type=float, default=1.0)
parser.add_argument("--voxel", help="weight of the voxel loss",
                    type=float, default=1.0)
parser.add_argument("--normal", help="weight of the normal loss",
                    type=float, default=0)
parser.add_argument("--edge", help="weight of the edge loss",
                    type=float, default=0.5)
# dataset/loader arguments
# TODO ben should handle this
parser.add_argument('--num_samples', type=int, help='number of sampels to ShapeNet dataset', default=None)
parser.add_argument('--dataRoot', type=str, help='file root')
parser.add_argument('--dataTrainList', type=str, help='train file list')
parser.add_argument('--dataTestList', type=str, help='test file list')
parser.add_argument('--batchSize', '-b', type=int,
                    defaults=16, help='batch size')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=4)

parser.add_argument('--nEpoch', type=int, default=10,
                    help='number of epochs to train for')

parser.add_argument('')

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

# model and datasets/loaders definition
if model_name == 'ShapeNet':
    model = ShapeNetModel(ShapeNetFeatureExtractor(3), residual=options.residual,
                          cubify_threshold=options.threshold,
                          image_shape=(137, 137),  # TODO ben verify // ok
                          vertex_feature_dim=options.featDim,
                          num_refinement_stages=options.num_refinement_stages)
    # TODO ben dataloading // added num_sampels arg
    dataset = shapeNet_Dataset(options.dataRoot, options.num_sampels)
    trainloader = DataLoader(
        dataset, batch_size=options.batchSize, shuffle=True, num_workers=options.workers)
    testloader = DataLoader(
        dataset, batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

else:
    model = Pix3DModel(pretrained_MaskRcnn(num_classes=10, pretrained=True),
                       cubify_threshold=options.threshold,
                       image_shape=(281, 187),  # TODO ben verify // 224,224 => 281,187
                       vertex_feature_dim=options.featDim,
                       num_refinement_stages=options.num_refinement_stages)
    # TODO ben dataloading
    dataset = pix3dDataset(options.dataRoot)
    trainloader = DataLoader(
        dataset, batch_size=options.batchSize, shuffle=True, num_workers=options.workers)
    testloader = DataLoader(
        dataset, batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

# use data parallel if possible
if len(devices > 1):
    model = nn.DataParallel(model)

model: nn.Module = model.to(devices[0])

# Create Optimizer
lrate = options.lr
decay = options.weightDecay
if options.optim == 'Adam':
    optimizer = Adam(model.parameters(), lr=lrate, weight_decay=decay)
else:
    optimizer = SGD(model.parameters(), lr=lrate, weight_decay=decay)
    # TODO they increased learning rate do we wish to do the same?
    # linearly increasing the learning rate from 0.002 to 0.02 over the first 1K iterations,

# loss weights
loss_weights = {'c': options.chamfer,
                'v': options.voxel,
                'n': options.normal,
                'e': options.edge
                }

# checkpoint directory
now = datetime.datetime.now()
save_path = now.isoformat()
dir_name = os.path.join('checkpoints', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

# Train model on the dataset
losses = []
for epoch in range(epochs):
    epoch_loss = []
    print(f'--- EPOCH {epoch+1}/{epochs} ---')

    # Set to Train mode
    model.train()
    with tqdm.tqdm(total=len(trainloader.batch_sampler), file=sys.stdout) as pbar:
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            # TODO ben is this what we are getting? yes
            images, voxel_gts, pts_gts = data

            images = images.cuda()
            voxels_gts = voxels_gts.cuda()
            pts_gts = pts_gts.cuda()

            # predict and comput loss
            output = model(images)

            # TODO how will we treat the backbone training?
            # this only takes our losses into account
            loss = total_loss(loss_weights, output, voxel_gts, pts_gts)

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
    torch.save(model.state_dict(), '%s/model%i.pth' % (dir_name, epoch))

print(f"training done avg loss {np.mean(losses)}")
