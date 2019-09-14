import argparse
import datetime
import os
import sys

import numpy as np
import torch
import torch.nn as nn


from model import (Pix3DModel, ShapeNetModel, pretrained_MaskRcnn,
                   pretrained_ResNet50)

assert torch.cuda.is_available(), "the training process is slow and requires gpu"


parser = argparse.ArgumentParser()

# model args
parser.add_argument(
    "--model", "-m", help="the model we wish to train", choices=["ShapeNet", "Pix3D"], required=True)
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

# TODO nice to have
parser.add_argument("--elipsoid", default=False, action="store_true",
                    help="wether to skip the voxel branch and start always from elipsoid mesh")

# sample to evaluate and output paths
parser.add_argument('--imagePath', type=str, help='the path to find the data')
parser.add_argument('--savePath', type=str, default='eval/',
                    help='the path to save the reconstructed meshes')
parser.add_argument('--saveName', type=str, default='out_mesh',
                    help='the name of the output mesh')

options = parser.parse_args()

# model definition
if options.model == 'ShapeNet':
    model = ShapeNetModel(pretrained_ResNet50(nn.functional.cross_entropy, num_classes=10,
                                              pretrained=True),
                          residual=options.residual,
                          cubify_threshold=options.threshold,
                          image_shape=(137, 137),
                          vertex_feature_dim=options.featDim,
                          num_refinement_stages=options.num_refinement_stages)


else:
    model = Pix3DModel(pretrained_MaskRcnn(num_classes=10, pretrained=True),
                       cubify_threshold=options.threshold,
                       image_shape=(281, 187),
                       vertex_feature_dim=options.featDim,
                       num_refinement_stages=options.num_refinement_stages)


# load checkpoint
model.load_state_dict(torch.load(options.model_path))
model: nn.Module = model.to('cuda')


def load_file(file_path):
    import matplotlib.image as mpimg
    img = torch.from_numpy(mpimg.imread(file_path))
    return img


img = load_file(options.dataPath)

output = model(img)

# TODO save each predicted mesh as point_clouds? / vertices+faces?
# do what you think is best

if not os.path.exists(options.savePath):
    os.mkdir(options.savePath)


vertex_positions = output['vertex_postions']
edge_index = output['edge_index']  # adj matrix
face_index = output['face_index']  # faces per graph
vertice_index = output['vertice_index']  # vertices per graph
mesh_faces = output['faces']
voxels = output['voxels']
graphs_per_image = output['graphs_per_image']

# TODO save voxels and meshes

# TODO way to visualize the meshes pointclouds
# https://medium.com/@yzhong.cs/beyond-data-scientist-3d-plots-in-python-with-examples-2a8bd7aa654b

# TODO visualize voxels
# https://matplotlib.org/3.1.1/gallery/mplot3d/voxels_rgb.html

print("Finish!")
