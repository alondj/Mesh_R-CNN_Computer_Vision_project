
import numpy as np
import scipy.io
from collections import namedtuple

import torch
from torch import Tensor


Mesh = namedtuple('Mesh', ['vertices', 'faces'])


def save_voxels(voxels, filename: str, threshold: float = 0.5):
    if not isinstance(voxels, np.ndarray):
        voxels = voxels.cpu().data.numpy()
    # create voxel mask
    voxels = (voxels > threshold).astype(np.int32)
    np.save(filename, voxels)


def save_mesh(vertices, faces, filename: str):
    # List of geometric vertices
    # v 0.123 0.234 0.345
    # v ...
    # Polygonal face element
    # f 1 2 3
    # f ...
    if not isinstance(vertices, np.ndarray):
        vertices = vertices.cpu().numpy()
    vertices = np.hstack((np.full([vertices.shape[0], 1], 'v'), vertices))

    if not isinstance(faces, np.ndarray):
        faces = faces.cpu().numpy()

    if faces.min() == 0:
        faces = faces.copy()
        faces += 1

    faces = np.hstack((np.full([faces.shape[0], 1], 'f'), faces))
    mesh = np.vstack((vertices, faces))
    np.savetxt(filename + ".obj", mesh, fmt='%s', delimiter=' ')


def load_voxels(path: str, tensor=False):
    if path.endswith(".npy"):
        vxls = np.load(path)
    else:
        assert path.endswith(".mat")
        vxls = scipy.io.loadmat(path)['voxel']
    if tensor:
        return torch.from_numpy(vxls)
    return vxls


def load_mesh(filename: str, tensor=False) -> Mesh:
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

    vertices = np.array(vertices)
    triangles = np.array(triangles)
    if triangles.min() == 1:
        triangles -= 1

    assert triangles.min() == 0

    if tensor:
        vertices = torch.from_numpy(vertices)
        triangles = torch.from_numpy(triangles)

    return Mesh(vertices, triangles)


def read_pix3d_data():
    # read the mesh data return np arrays vertices and faces
    mesh = load_mesh("model.obj")

    print(mesh.vertices.shape, mesh.faces.shape)

    print(np.max(mesh.vertices, axis=0))
    print(np.min(mesh.vertices, axis=0))
    # read voxel as np array
    mat = scipy.io.loadmat('voxel.mat')['voxel']
    print(mat.shape)


if __name__ == "__main__":
    read_pix3d_data()
