
import numpy as np
from collections import namedtuple

import torch
from torch import Tensor
import pickle

Mesh = namedtuple('Mesh', ['vertices', 'faces'])


def save_voxels(voxels: Tensor, filename: str, threshold: float = 0.5):
    # create voxel mask
    vxls = voxels.cpu().data.numpy()
    vxls = (vxls > threshold).astype(np.int32)
    np.save(filename, vxls)


def save_mesh(vertices: Tensor, faces: Tensor, filename: str):
    # List of geometric vertices
    # v 0.123 0.234 0.345
    # v ...
    # Polygonal face element
    # f 1 2 3
    # f ...
    vert = vertices.cpu().data.numpy()
    vert = np.hstack((np.full([vert.shape[0], 1], 'v'), vert))

    face = faces.cpu().data.numpy()
    face = np.hstack((np.full([faces.shape[0], 1], 'f'), faces))
    mesh = np.vstack((vert, face))
    np.savetxt(filename + ".obj", mesh, fmt='%s', delimiter=' ')


def load_voxels(path: str) -> np.ndarray:
    return np.load(path)


def load_mesh(filename: str) -> Mesh:
    triangles = []
    vertices = []
    with open(filename) as file:
        for line in file:
            components = line.strip(' \n').split(' ')
            if components[0] == "f":  # face data
                # e.g. "f 1/1/1/ 2/2/2 3/3/3 4/4/4 ..."
                indices = list(
                    map(lambda c: int(c.split('/')[0]), components[1:]))
                for i in range(0, len(indices) - 2):
                    triangles.append(indices[i: i+3])
            elif components[0] == "v":  # vertex data
                # e.g. "v  30.2180 89.5757 -76.8089"
                vertex = list(map(float, components[1:]))
                vertices.append(vertex)

    return Mesh(np.array(vertices), np.array(triangles))


def read_pix3d_data():
    # read the mesh data return np arrays vertices and faces
    mesh = load_mesh("model.obj")

    print(mesh.vertices.shape, mesh.faces.shape)

    print(np.max(mesh.vertices, axis=0))
    print(np.min(mesh.vertices, axis=0))
    # read voxel as np array
    import scipy.io
    mat = scipy.io.loadmat('voxel.mat')['voxel']
    print(mat.shape)


if __name__ == "__main__":
    read_pix3d_data()
