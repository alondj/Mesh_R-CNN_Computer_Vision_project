import numpy as np
import argparse
import torch
import os
import sys

from model import ShapeNetModel, ShapeNetFeatureExtractor, pretrained_MaskRcnn, Pix3DModel

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

# sample to evaluate and output paths
parser.add_argument('--dataPath', type=str, help='the path to find the data')
parser.add_argument('--modelPath', type=str,
                    help='the path to find the trained model')
parser.add_argument('--savePath', type=str, default='eval/',
                    help='the path to save the reconstructed meshes')
parser.add_argument('--saveName', type=str, default='out_mesh',
                    help='the name of the output mesh')

options = parser.parse_args()
print(options)

# Check Device (CPU / GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor
device = torch.device(device)
print(f"evaluating on device: {device}")

# Create Network and load weights
if options.model == 'ShapeNet':
    model = ShapeNetModel(ShapeNetFeatureExtractor(3), residual=options.residual,
                          cubify_threshold=options.threshold,
                          image_shape=(137, 137),
                          vertex_feature_dim=options.featDim,
                          num_refinement_stages=options.num_refinement_stages)
else:
    model = Pix3DModel(pretrained_MaskRcnn(num_classes=10, pretrained=False),
                       cubify_threshold=options.threshold,
                       image_shape=(224, 224),
                       vertex_feature_dim=options.featDim,
                       num_refinement_stages=options.num_refinement_stages)

model.load_state_dict(torch.load(options.modelPath, map_location=device))
model.eval()
model = model.to(device)


def load_file(file_path):
    # TODO ben load the given image from options.dataPath
    import matplotlib.image as mpimg
    img = torch.from_numpy(mpimg.imread(file_path))
    return img


img = load_file(options.dataPath)

output = model(img)

# TODO save each predicted mesh as point_clouds? / vertices+faces?
# do what you think is best


print("Finish!")
