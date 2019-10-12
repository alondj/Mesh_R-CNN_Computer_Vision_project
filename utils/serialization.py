
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


def _read_header(fp):
    """ Read binvox header. Mostly meant for internal use.
    """
    line = fp.readline().strip()
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    return dims, translate, scale


def _read_as_3d_array(fp, fix_coords=True):
    """ Read binary binvox format as array.

    Returns the model with accompanying metadata.

    Voxels are stored in a three-dimensional numpy array, which is simple and
    direct, but may use a lot of memory for large models. (Storage requirements
    are 8*(d^3) bytes, where d is the dimensions of the binvox model. Numpy
    boolean arrays use a byte per element).

    Doesn't do any checks on input except for the '#binvox' line.
    """
    dims, translate, scale = _read_header(
        fp)  # [32,32,32],[-0.302239, -0.169754,0.360326],[0.720652]
    l = fp.read()
    raw_data = np.fromstring(l, dtype=np.uint8)
    # if just using reshape() on the raw data:
    # indexing the array as array[i,j,k], the indices map into the
    # coords as:
    # i -> x
    # j -> z
    # k -> y
    # if fix_coords is true, then data is rearranged so that
    # mapping is
    # i -> x
    # j -> y
    # k -> z
    values, counts = raw_data[::2], raw_data[1::2]
    data = np.repeat(values, counts).astype(np.bool)

    data = data.reshape(dims)
    if fix_coords:
        # xzy to xyz TODO the right thing
        data = np.transpose(data, (0, 2, 1))
        axis_order = 'xyz'
    else:
        axis_order = 'xzy'
    return 1*data


def load_voxels(path: str, tensor=False):
    if path.endswith(".npy"):
        vxls = np.load(path)
    elif path.endswith(".mat"):
        vxls = scipy.io.loadmat(path)['voxel']
    else:
        assert path.endswith(".binvox")
        with open(path, 'rb') as binvox_file:
            vxls = _read_as_3d_array(binvox_file)
    if tensor:
        return torch.from_numpy(vxls)
    return vxls


def load_mesh(filename: str, tensor=False) -> Mesh:
    triangles = []
    vertices = []
    filename = filename.replace(".binvox", ".obj")
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
        vertices = torch.from_numpy(vertices).float()
        triangles = torch.from_numpy(triangles).long()

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
