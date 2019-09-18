import argparse
import datetime
import os
import sys
import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn as nn


from model import (Pix3DModel, ShapeNetModel, pretrained_MaskRcnn,
                   pretrained_ResNet50)
from utils import save_mesh, save_voxels

assert torch.cuda.is_available(), "the training process is slow and requires gpu"


parser = argparse.ArgumentParser()

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
                    help='Cubify threshold', type=float, default=0.2)
parser.add_argument("--residual", default=False,
                    action="store_true", help="whether to use residual refinement for ShapeNet")

# sample to evaluate and output paths
parser.add_argument('--imagePath', type=str, help='the path to find the data')
parser.add_argument('--savePath', type=str, default='eval/',
                    help='the path to save the reconstructed meshes')

options = parser.parse_args()

# model definition
if options.model == 'ShapeNet':
    model = ShapeNetModel(pretrained_ResNet50(nn.functional.nll_loss, num_classes=13,
                                              pretrained=True),
                          residual=options.residual,
                          cubify_threshold=options.threshold,
                          vertex_feature_dim=options.featDim,
                          num_refinement_stages=options.num_refinement_stages)
else:
    model = Pix3DModel(pretrained_MaskRcnn(num_classes=10, pretrained=True),
                       cubify_threshold=options.threshold,
                       vertex_feature_dim=options.featDim,
                       num_refinement_stages=options.num_refinement_stages)


# load checkpoint
model.load_state_dict(torch.load(options.model_path))
model: nn.Module = model.to('cuda').eval()


img = torch.from_numpy(mpimg.imread(options.imagePath))

output = model(img)


if not os.path.exists(options.savePath):
    os.mkdir(options.savePath)


vertex_positions = output['vertex_postions']
edge_index = output['edge_index']  # adj matrix
face_index = output['face_index']  # faces per graph
vertice_index = output['vertice_index']  # vertices per graph
mesh_faces = output['faces']
voxels = output['voxels']
graphs_per_image = output['graphs_per_image']

print(f"saving output to {options.savePath}")

filename = os.path.basename(options.imgPath).split('.')[0]
# save voxels
save_voxels(voxels, os.path.join(options.savePath, filename))

# TODO handle the graph_per_image vertex per graph nonsense
# save the intermediate meshes
for idx, (pos, faces) in enumerate(zip(vertex_positions.split(vertice_index), mesh_faces.split(face_index))):
    mesh_file = os.path.join(options.savePath, filename, f"_mesh_{idx}")
    save_mesh(pos, faces, mesh_file)

print("Finish!")
