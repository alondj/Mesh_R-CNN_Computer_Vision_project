import argparse
import platform
import torch
import torch.nn as nn
import os
from data.dataloader import pix3dDataset, shapeNet_Dataset, dataLoader
from meshRCNN import Pix3DModel, ShapeNetModel, pretrained_MaskRcnn, pretrained_ResNet50
from dataParallel import CustomDP
from utils.eval_utils import validate, safe_print

assert torch.cuda.is_available(), "the training process is slow and requires gpu"

parser = argparse.ArgumentParser(description="dataset evaluation script")

# model args
parser.add_argument(
    "--model", "-m", help="the model we wish to train", choices=["ShapeNet", "Pix3D"], required=True)
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

parser.add_argument('--output_path', type=str, help='path to output folder')

options = parser.parse_args()

devices = [torch.device('cuda', i)
           for i in range(torch.cuda.device_count())]

# print header
model_name = options.model
title = f'{model_name} evaluation \n used_gpus: {len(devices)}\n'
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

gpus = [torch.cuda.get_device_name(device) for device in devices]
is_pix3d = model_name == 'Pix3D'

classes = options.classes
if classes != None:
    classes = [item for item in options.classes.split(',')]

if is_pix3d:
    num_classes = 10
    backbone = pretrained_MaskRcnn(
        num_classes=num_classes, pretrained=False)
    model = Pix3DModel(backbone,
                       cubify_threshold=options.threshold,
                       vertex_feature_dim=options.featDim,
                       num_refinement_stages=options.num_refinement_stages,
                       voxel_only=False)

    dataset_cls = pix3dDataset
    num_voxels = 24
else:
    num_classes = 13
    backbone = pretrained_ResNet50(nn.functional.nll_loss,
                                   num_classes=num_classes,
                                   pretrained=False)
    model = ShapeNetModel(backbone,
                          residual=options.residual,
                          cubify_threshold=options.threshold,
                          vertex_feature_dim=options.featDim,
                          num_refinement_stages=options.num_refinement_stages,
                          voxel_only=False)

    dataset_cls = shapeNet_Dataset
    num_voxels = 48

dataset = dataset_cls(options.dataRoot, classes=classes)

testloader = dataLoader(dataset, options.batchSize, num_voxels=num_voxels,
                        num_workers=options.workers,
                        num_train_samples=None,
                        train_ratio=1 - options.test_ratio)

# load checkpoint
safe_print(0, "loaded gcn checkpoint")
model.load_state_dict(torch.load(options.model_path))

if len(devices) > 1:
    safe_print(0, "using dataParallel")
    model = CustomDP(model, is_backbone=False, pix3d=is_pix3d)

model: nn.Module = model.to(devices[0]).eval()

metrics = validate(0, model, testloader, num_classes, is_pix3d=is_pix3d)

safe_print(0, "saving metrics")
torch.save(metrics, os.path.join(options.output_path,
                                 f"metrics_{model_name}.st"))
