import torch
from utils import load_mesh, save_mesh, save_voxels, load_voxels
import os
import numpy as np


def test_save_load_mesh():
    m = load_mesh(os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "teapot.obj"))
    save_mesh(*m, os.path.join(os.path.dirname(os.path.realpath(__file__)), "test"))
    m2 = load_mesh(os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "test.obj"))
    os.remove(os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "test.obj"))

    assert np.allclose(m.vertices, m2.vertices)
    assert np.allclose(m.faces, m2.faces)


def test_save_load_voxels():
    voxels = np.random.randint(0, 2, size=9).reshape(3, 3)

    save_voxels(torch.from_numpy(voxels), "test")

    vxls = load_voxels("test.npy")
    os.remove("test.npy")

    assert np.allclose(voxels, vxls)
