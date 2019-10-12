import argparse
import datetime
import os
import platform
from itertools import chain
from pathlib import Path
import torch
import torch.nn as nn
from utils.train_utils import train_gcn, safe_print, load_dict
from torch.optim import SGD, Adam
from collections import OrderedDict
from data.dataloader import pix3dDataset, shapeNet_Dataset, dataLoader
from model import Pix3DModel, ShapeNetModel, pretrained_MaskRcnn, pretrained_ResNet50

import torch.distributed as dist

import torch.multiprocessing as mp
from parallel import CustomDP

assert torch.cuda.is_available(), "the training process is slow and requires gpu"

parser = argparse.ArgumentParser()

# model args
parser.add_argument(
    "--model", "-m", help="the model we wish to train", choices=["ShapeNet", "Pix3D"], required=True)
parser.add_argument('--featDim', type=int, default=128,
                    help='number of vertex features')
parser.add_argument(
    "--model_path", help="path of a pretrained model to cintinue training", default='')
parser.add_argument('--backbone_path', '-bp', type=str, default='',
                    help='path of a pretrained backbone if we wish to continue training from checkpoint must be provided with GCN_path')
parser.add_argument('--num_refinement_stages', "-nr", type=int,
                    default=3, help='number of mesh refinement stages')
parser.add_argument('--threshold', '-th',
                    help='Cubify threshold', type=float, default=0.2)
parser.add_argument('--voxel_only', default=False, action='store_true',
                    help='whether to return only the cubified mesh resulting from cubify')
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
                    type=float, default=0.1)
parser.add_argument("--edge", help="weight of the edge loss",
                    type=float, default=0.5)
parser.add_argument("--backbone", help="weight of the backbone loss",
                    type=float, default=1.0)

# dataset/loader arguments
parser.add_argument('--num_sampels', type=int,
                    help='number of sampels to dataset', default=None)
parser.add_argument('--train_ratio', type=float, help='ration of samples used for training',
                    default=None)
parser.add_argument('-c', '--classes', help='classes of the exampels in the dataset',
                    type=str, default=None)
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

parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'for each GPU a process will be spawned')


def main():
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

    if options.multiprocessing_distributed:
        world_size = len(devices)
        mp.spawn(worker, nprocs=len(devices),
                 args=(options, world_size))
    else:
        worker(0, options, len(devices))


def worker(gpu_id, options, world_size):
    if options.multiprocessing_distributed:
        dist.init_process_group('nccl', init_method='tcp://127.0.0.1:FREEPORT',
                                world_size=world_size, rank=gpu_id)
        torch.cuda.set_device(gpu_id)

    epochs = options.nEpoch
    is_pix3d = options.model == 'Pix3D'
    model_name = options.model

    pretrained_backbone = options.model_path != '' or options.backbone_path != ''
    # model and datasets/loaders definition
    classes = options.classes
    if classes != None:
        classes = [item for item in options.classes.split(',')]

    if is_pix3d:
        backbone = pretrained_MaskRcnn(
            num_classes=10, pretrained=pretrained_backbone)
        model = Pix3DModel(backbone,
                           cubify_threshold=options.threshold,
                           vertex_feature_dim=options.featDim,
                           num_refinement_stages=options.num_refinement_stages,
                           voxel_only=options.voxel_only)

        dataset_cls = pix3dDataset
        num_voxels = 24
    else:
        backbone = pretrained_ResNet50(nn.functional.nll_loss,
                                       num_classes=13,
                                       pretrained=pretrained_backbone)
        model = ShapeNetModel(backbone,
                              residual=options.residual,
                              cubify_threshold=options.threshold,
                              vertex_feature_dim=options.featDim,
                              num_refinement_stages=options.num_refinement_stages,
                              voxel_only=options.voxel_only)

        dataset_cls = shapeNet_Dataset
        num_voxels = 48

    dataset = dataset_cls(options.dataRoot, classes=classes)
    if options.multiprocessing_distributed:
        section = gpu_id
    else:
        section = -1
    trainloader = dataLoader(dataset, options.batchSize, num_voxels=num_voxels,
                             num_workers=options.workers,
                             num_train_samples=options.num_sampels,
                             train_ratio=options.train_ratio, rank=section)

    # load checkpoint if possible
    if options.backbone_path != '':
        safe_print(gpu_id, "loaded backbone checkpoint")
        model.backbone.load_state_dict(load_dict(options.backbone_path))

    # load checkpoint if possible
    if options.model_path != '':
        safe_print(gpu_id, "loaded gcn checkpoint")
        model.load_state_dict(load_dict(options.model_path))

    # select trainable parameters
    trained_parameters = model.voxelBranch.parameters()
    if not options.voxel_only:
        trained_parameters = chain(trained_parameters,
                                   model.refineStages.parameters())
        model.refineStages.train()

    model.voxelBranch.train()
    if options.train_backbone:
        trained_parameters = chain(trained_parameters,
                                   model.backbone.parameters())
        model.backbone.train()
    else:
        for p in model.backbone.parameters():
            p.requires_grad = False
        model.backbone.eval()

    model: nn.Module = model.cuda(gpu_id)

    # use data parallel or distributed data parallel if possible
    if options.multiprocessing_distributed:
        safe_print(gpu_id, "using multiprocessing distributed training")
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[gpu_id])
    elif torch.cuda.device_count() > 1:
        safe_print(gpu_id, "using dataParallel")
        model = CustomDP(model, is_backbone=False, pix3d=is_pix3d)

    # Create Optimizer
    lrate = options.lr
    decay = options.weightDecay
    if options.optim == 'Adam':
        optimizer = Adam(trained_parameters, lr=lrate, weight_decay=decay)
    else:
        optimizer = SGD(trained_parameters, lr=lrate, weight_decay=decay)

    # loss weights
    loss_weights = {'chamfer_loss': options.chamfer,
                    'voxel_loss': options.voxel,
                    'normal_loss': options.normal,
                    'edge_loss': options.edge,
                    'backbone_loss': options.backbone
                    }

    # checkpoint directories
    now = datetime.datetime.now()
    save_path = now.isoformat()
    GCN_path = os.path.join('checkpoints', model_name, 'GCN', save_path)

    if gpu_id == 0 and not os.path.exists(GCN_path):
        Path(GCN_path).mkdir(parents=True, exist_ok=True)

    # Train model on the dataset
    stats = OrderedDict()
    lr_count = 0
    curr_lr = lrate
    for epoch in range(epochs):
        safe_print(gpu_id, f'--- EPOCH {epoch+1}/{epochs} ---')

        epoch_stats, lr_count, curr_lr = train_gcn(gpu_id, model, optimizer, trainloader, epoch,
                                                   loss_weights, backbone_train=options.train_backbone,
                                                   is_pix3d=is_pix3d, lr_count=lr_count, curr_lr=curr_lr)
        stats[epoch] = epoch_stats
        # save the model
        safe_print(gpu_id, 'saving net...')
        # for eg checkpoints/Pix3D/date/model_{1}.pth
        file_name = f"model_{epoch}.pth"
        if gpu_id == 0:
            try:
                state_dict = model.module.state_dict()
            except AttributeError:
                state_dict = model.state_dict()
            torch.save(state_dict,
                       os.path.join(GCN_path, file_name))

    # finished training
    if options.multiprocessing_distributed:
        dist.barrier()
        dist.destroy_process_group()

    if gpu_id == 0:
        torch.save(stats, os.path.join(GCN_path, f"stats.st"))
    safe_print(gpu_id, "all Done")


if __name__ == "__main__":
    main()
