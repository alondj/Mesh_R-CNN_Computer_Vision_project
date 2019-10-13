
import torch
from torch.nn.functional import adaptive_max_pool3d, interpolate
from torch import Tensor
from .serialization import load_mesh, load_voxels
from .process import normalize_mesh
from .mesh_sampling import sample
from .rotation import rotation
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = 12.8, 9.6

# code based on examples from
# visualize meshes pointclouds
# https://medium.com/@yzhong.cs/beyond-data-scientist-3d-plots-in-python-with-examples-2a8bd7aa654b

# visualize voxels
# https://matplotlib.org/3.1.1/gallery/mplot3d/voxels_rgb.html


def show_mesh(mesh, alpha=0):
    if isinstance(mesh, str):
        mesh = load_mesh(mesh)

    vertices, triangles = mesh
    if not isinstance(vertices, np.ndarray):
        vertices = vertices.cpu().numpy()
        triangles = triangles.cpu().numpy()

    if np.absolute(vertices).max() > 1:
        vertices = normalize_mesh(vertices)

    if triangles.min() == 1:
        triangles -= 1

    vertices = np.matmul(vertices, rotation(alpha))
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2] * -1
    ax = plt.axes(projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.plot_trisurf(x, z, triangles, y, color='grey')
    plt.show()


def show_voxels(voxel_mask, threshold: float = 0.5):
    if isinstance(voxel_mask, str):
        voxel_mask = load_voxels(voxel_mask)
    if not isinstance(voxel_mask, np.ndarray):
        voxel_mask = voxel_mask.cpu().numpy()

    voxel_mask = (voxel_mask > threshold).astype(np.int32)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.voxels(voxel_mask, facecolors='grey', edgecolor='black')

    plt.show()


def show_mesh_pointCloud(mesh, alpha=-90):
    if isinstance(mesh, str):
        mesh = load_mesh(mesh, tensor=True)

    vertices, faces = mesh

    if isinstance(vertices, np.ndarray):
        vertices = torch.from_numpy(vertices)
        faces = torch.from_numpy(faces)

    points = sample(vertices, faces).cpu().numpy()

    points = np.matmul(points, rotation(alpha))
    ax = plt.axes(projection='3d')
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ax.scatter(x, y, z, linewidth=1)
    plt.show()
