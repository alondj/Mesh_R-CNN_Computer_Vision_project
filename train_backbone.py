import argparse
import datetime
import os
import platform
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from collections import OrderedDict
from data.dataloader import pix3dDataset, shapeNet_Dataset, dataLoader
from meshRCNN import pretrained_MaskRcnn, pretrained_ResNet50, CustomDP
from utils.train_utils import train_backbone, load_dict
assert torch.cuda.is_available(), "the training process is slow and requires gpu"

parser = argparse.ArgumentParser()

# model args
parser.add_argument(
    "--model", "-m", help="the backbone model we wish to train", choices=["ShapeNet", "Pix3D"], required=True)
parser.add_argument('--backbone_path', '-bp', type=str, default='',
                    help='path of a pretrained backbone if we wish to continue training from checkpoint must be provided with GCN_path')

# dataset/loader arguments
parser.add_argument('--num_sampels', type=int,
                    help='number of sampels to dataset', default=None)
parser.add_argument('--train_ratio', type=float, help='ration of samples used for training',
                    default=None)
parser.add_argument(
    '-c', '--classes', help='classes of the exampels in the dataset', type=str, default=None)
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


def main():
    options = parser.parse_args()
    is_pix3d = options.model == 'Pix3D'
    epochs = options.nEpoch

    devices = [torch.device('cuda', i)
               for i in range(torch.cuda.device_count())]

    # print header
    if is_pix3d:
        disc = "training backbone mask-RCNN for Pix3D detection,classification and instance segmentation"
    else:
        disc = "training backbone ResNet50 for ShapeNet classification"
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
    classes = options.classes
    if classes != None:
        classes = [item for item in options.classes.split(',')]

    if is_pix3d:
        model = pretrained_MaskRcnn(num_classes=10, pretrained=True)
        num_voxels = 24
        dataset_cls = pix3dDataset
    else:
        model = pretrained_ResNet50(nn.functional.nll_loss, num_classes=13,
                                    pretrained=True)
        num_voxels = 48
        dataset_cls = shapeNet_Dataset

    dataset = dataset_cls(options.dataRoot, classes=classes)
    trainloader = dataLoader(dataset, options.batchSize, num_voxels=num_voxels,
                             num_workers=options.workers,
                             num_train_samples=options.num_sampels,
                             train_ratio=options.train_ratio)

    if options.backbone_path != '':
        model.load_state_dict(load_dict(options.backbone_path))

    # use data parallel if possible
    if len(devices) > 1:
        model = CustomDP(model, is_backbone=True, pix3d=is_pix3d)

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
    dir_name = os.path.join('checkpoints',
                            options.model, 'backbone', save_path)
    if not os.path.exists(dir_name):
        Path(dir_name).mkdir(parents=True, exist_ok=True)

    stats = OrderedDict()
    lr_count = 0
    curr_lr = lrate
    # Train model on the dataset
    for epoch in range(epochs):
        print(f'--- EPOCH {epoch+1}/{epochs} ---')
        epoch_stats, lr_count, curr_lr = train_backbone(0, model, optimizer, trainloader,
                                                        epoch, lr_count=lr_count,
                                                        curr_lr=curr_lr, is_pix3d=is_pix3d)
        stats[epoch] = epoch_stats

        # save the model
        print('saving net...')

        # for eg checkpoints/ShapeNet/backbone/date/model_1.pth
        file_name = f"model_{epoch}.pth"
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(dir_name, file_name))
        torch.save(stats, os.path.join(dir_name, f"stats_{epoch}.st"))

    file_name = f"model_final.pth"
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(dir_name, file_name))
    torch.save(stats, os.path.join(dir_name, f"stats_final.st"))
    print(f"all Done")


if __name__ == "__main__":
    main()
