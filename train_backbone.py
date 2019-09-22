import argparse
import datetime
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim import SGD, Adam

from data.dataloader import pix3dDataset, shapeNet_Dataset, pix3dDataLoader, shapenetDataLoader
from model import pretrained_MaskRcnn, pretrained_ResNet50

assert torch.cuda.is_available(), "the training process is slow and requires gpu"

parser = argparse.ArgumentParser()

# model args
parser.add_argument(
    "--model", "-m", help="the backbone model we wish to train", choices=["ShapeNet", "Pix3D"], required=True)
parser.add_argument('--backbone_path', '-bp', type=str, default='',
                    help='path of a pretrained backbone if we wish to continue training from checkpoint must be provided with GCN_path')

# dataset/loader arguments
parser.add_argument('--num_samples', type=int,
                    help='number of sampels to dataset', default=None)
parser.add_argument('--dataRoot', type=str, help='file root')

parser.add_argument('--batchSize', '-b', type=int,
                    defaults=16, help='batch size')
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
if options.model == 'ShapeNet':
    disc = "training backbone ResNet50 for ShapeNet classification"
else:
    disc = "training backbone mask-RCNN for Pix3D detection,classification and instance segmentation"
title = f'{disc} \n used_gpus: {len(devices)}\n epochs: {epochs}\n'
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

# model and datasets/loaders definition
if options.model == 'ShapeNet':
    model = pretrained_ResNet50(nn.functional.nll_loss, num_classes=13,
                                pretrained=True)

    dataset = shapeNet_Dataset(options.dataRoot, options.num_sampels)
    trainloader = shapenetDataLoader(
        dataset, options.batchSize, 48, options.num_workers)
else:
    model = pretrained_MaskRcnn(num_classes=10, pretrained=True)
    dataset = pix3dDataset(options.dataRoot, options.num_sampels)
    trainloader = pix3dDataLoader(
        dataset, options.batchSize, 24, options.num_workers)

# use data parallel if possible
# TODO i do not know if it will work for mask rcnn
if len(devices) > 1:
    model = nn.DataParallel(model)

model: nn.Module = model.to(devices[0])

# Create Optimizer
lrate = options.lr
decay = options.weightDecay
if options.optim == 'Adam':
    optimizer = Adam(model.parameters(), lr=lrate, weight_decay=decay)
else:
    optimizer = SGD(model.parameters(), lr=lrate, weight_decay=decay)


now = datetime.datetime.now()
save_path = now.isoformat()
dir_name = os.path.join('checkpoints', options.model, 'backbone', save_path)
if not os.path.exists(dir_name):
    Path(dir_name).mkdir(parents=True, exist_ok=True)

# Train model on the dataset
losses = []
for epoch in range(epochs):
    epoch_loss = []
    print(f'--- EPOCH {epoch+1}/{epochs} ---')

    # Set to Train mode
    model.train()
    with tqdm.tqdm(total=len(trainloader.batch_sampler), file=sys.stdout) as pbar:
        for i, batch in enumerate(trainloader, 0):
            optimizer.zero_grad()

            batch = batch.to(devices[0])
            images, backbone_targets = batch.images, batch.targets

            # predict and comput loss
            out = model(images, backbone_targets)

            if options.model == 'ShapeNet':
                loss = out[0]
            else:
                loss = sum(out[0].values())

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

    # for eg checkpoints/ShapeNet/backbone/date/model_1.pth
    file_name = f"model_{epoch}.pth"
    torch.save(model.state_dict(), os.path.join(dir_name, file_name))

print(f"backbone training done avg loss {np.mean(losses)}")
