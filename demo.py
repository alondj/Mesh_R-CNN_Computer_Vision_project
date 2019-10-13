import PIL.Image
import argparse
import datetime
import os
import sys
import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from meshRCNN import (Pix3DModel, ShapeNetModel, pretrained_MaskRcnn,
                      pretrained_ResNet50)
from utils import save_mesh, save_voxels, show_mesh, show_voxels, show_mesh_pointCloud

assert torch.cuda.is_available(), "the training process is slow and requires gpu"

parser = argparse.ArgumentParser("model inference script")

# model args
parser.add_argument(
    "--model", "-m", help="the model to run the demo with", choices=["ShapeNet", "Pix3D"], required=True)
parser.add_argument('--featDim', type=int, default=128,
                    help='number of vertex features')
parser.add_argument('--modelPath', type=str, required=True,
                    help='the path to find the trained model')
parser.add_argument('--num_refinement_stages', "-nr", type=int,
                    default=3, help='number of mesh refinement stages')
parser.add_argument('--threshold', '-th',
                    help='Cubify threshold', type=float, default=0.5)
parser.add_argument("--residual", default=False,
                    action="store_true", help="whether to use residual refinement for ShapeNet")

# sample to evaluate and output paths
parser.add_argument('--imagePath', type=str, help='the path to find the data')
parser.add_argument('--savePath', type=str, default='eval/',
                    help='the path to save the reconstructed meshes')

parser.add_argument('--show', default=False, action='store_true',
                    help='whether to display the predicted voxels and meshes')

options = parser.parse_args()

# model definition
if options.model == 'ShapeNet':
    model = ShapeNetModel(pretrained_ResNet50(nn.functional.nll_loss, num_classes=13,
                                              pretrained=False),
                          residual=options.residual,
                          cubify_threshold=options.threshold,
                          vertex_feature_dim=options.featDim,
                          num_refinement_stages=options.num_refinement_stages, voxel_only=False)
else:
    model = Pix3DModel(pretrained_MaskRcnn(num_classes=10, pretrained=False),
                       cubify_threshold=options.threshold,
                       vertex_feature_dim=options.featDim,
                       num_refinement_stages=options.num_refinement_stages, voxel_only=False)

# load checkpoint
model.load_state_dict(torch.load(options.modelPath))
model: nn.Module = model.to('cuda').eval()

rgba_image = PIL.Image.open(options.imagePath)
rgb_image = rgba_image.convert('RGB')
img = torch.from_numpy(np.array(rgb_image))
img = img.transpose(2, 0)
img = img.type(torch.cuda.FloatTensor)
img = img.unsqueeze(0)
output = model(img)

vertex_positions = output['vertex_positions']
edge_index = output['edge_index']  # adj matrix
face_index = output['face_index']  # faces per graph
vertice_index = output['vertice_index']  # vertices per graph
faces = output['faces']
voxels = output['voxels']

print(f"saving output to {options.savePath}")

if not os.path.exists(options.savePath):
    Path(options.savePath).mkdir(parents=True, exist_ok=True)

filename = os.path.basename(options.imgPath).split('.')[0]
# save voxels
for idx, v in enumerate(voxels.split(1)):
    f_name = f"{filename}_voxel_obj{idx}"
    save_voxels(v.squeeze(0), os.path.join(options.savePath, f_name))
    if options.show:
        show_voxels(v)

# save the intermediate meshes
for stage, vs in enumerate(vertex_positions):
    for idx, (pos, fs) in enumerate(zip(vs.split(vertice_index), faces.split(face_index))):
        mesh_file = os.path.join(options.savePath, f"{filename}_mesh_stage{stage}_obj_{idx}")
        pos = pos.detach()
        save_mesh(pos, fs, mesh_file)
        if options.show:
            show_mesh(pos, fs)
            show_mesh_pointCloud((pos, fs))

print("Finish!")
