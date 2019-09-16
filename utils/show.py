
from .save import load_mesh, load_voxels
from .normalize_mesh import normalize_mesh
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = 12.8, 9.6


# visualize meshes pointclouds
# https://medium.com/@yzhong.cs/beyond-data-scientist-3d-plots-in-python-with-examples-2a8bd7aa654b

# visualize voxels
# https://matplotlib.org/3.1.1/gallery/mplot3d/voxels_rgb.html


def show_mesh(mesh):
    if isinstance(mesh, str):
        mesh = load_mesh(mesh)

    vertices, triangles = mesh
    if vertices.abs().max() > 1:
        vertices = normalize_mesh(vertices)

    triangles -= 1
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    ax = plt.axes(projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.plot_trisurf(x, z, triangles, y, shade=True, color='white')
    plt.show()


def show_voxels(voxel_mask):
    if isinstance(voxel_mask, str):
        voxel_mask = load_voxels(voxel_mask)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxel_mask, facecolors='grey', edgecolor='black', shade=True)

    plt.show()
