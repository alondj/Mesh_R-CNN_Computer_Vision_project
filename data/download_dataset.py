import urllib.request
import zipfile
import os
import shutil
import tarfile
from pathlib import Path

import torch
from model.layers import Cubify
from .read_binvox import read_as_3d_array
from utils import save_mesh, normalize_mesh

"""
the dataset should be built like that:
-dataset
----shapeNet
-------ShapeNet_pointclouds
-------ShapeNetRendering
-------ShapeNetVox32
-------ShapeNetMeshes
----pix3d
-------img
-------mask
-------model
-------pix3d.json
"""


def download_pix3d(download_path):
    """
    pix3d img,voxel,masks http://pix3d.csail.mit.edu/data/pix3d.zip
    pix3d pointclouds https://drive.google.com/file/d/1RZakyBu9lPbG85SyconBn4sR8r2faInV/view
    """
    os.mkdir(f"{download_path}/dataset")
    url = "http://pix3d.csail.mit.edu/data/pix3d.zip"
    zip_download_path = f"{download_path}/dataset/pix3d.zip"
    print("downloading pix3d img zip...")
    urllib.request.urlretrieve(url, zip_download_path)
    print("unzipping pix3d.zip")
    with zipfile.ZipFile(zip_download_path, 'r') as zip_ref:
        zip_ref.extractall(f"{download_path}/dataset/pix3d")
    print("deleting the zip file")
    os.remove(zip_download_path)
    print("finished pix3d")


def download_shapenet(download_path):
    """
    shapeNet imgs http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
    shapeNet voxel http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz
    shapeNet pointclouds https://drive.google.com/open?id=1cfoe521iTgcB_7-g_98GYAqO553W8Y0g
    """

    print("downloading shapeNet img zip")
    os.mkdir(f"{download_path}/dataset/shapeNet")
    url = "http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz"
    zip_download_path = f"{download_path}/dataset/shapeNet/ShapeNetRendering.tgz"
    urllib.request.urlretrieve(url, zip_download_path)
    print("unzipping")
    tf = tarfile.open(
        f"{download_path}/dataset/shapeNet/ShapeNetRendering.tgz")
    tf.extractall(f"{download_path}/dataset/shapeNet")
    print("deleting the zip file")
    os.remove(zip_download_path)

    print("downloading shapNet voxel zip")
    url = "http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz"
    zip_download_path = f"{download_path}/dataset/shapeNet/ShapeNetVox32.tgz"
    urllib.request.urlretrieve(url, zip_download_path)
    print("unzipping")
    tf = tarfile.open(f"{download_path}/dataset/shapeNet/ShapeNetVox32.tgz")
    tf.extractall(f"{download_path}/dataset/shapeNet")
    print("deleting the zip file")
    os.remove(zip_download_path)
    print("finished shapeNet")


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def render_shapenet_meshes(download_path):
    # TODO we render voxels at their given size maybe we should upsample/downsample
    renderer = Cubify(threshold=0.5).to('cuda')

    pathlist = Path(download_path).glob('**/*.binvox')

    with torch.no_grad():
        # every time render 16 models
        for b in batch(pathlist, n=16):
            voxels = []
            paths = [str(p) for p in b]
            # read a batch
            for p in paths:
                with open(p, 'rb') as binvox_file:
                    v = torch.from_numpy(read_as_3d_array(binvox_file))
                    voxels.append(v)
            v_batch = torch.stack(voxels).to('cuda')

            # render and split again
            v_index, f_index, v_pos, _, faces = renderer(v_batch)

            vs = v_pos.split(v_index)
            fs = faces.split(f_index)

            # save the normalized meshes
            for v, f, p in zip(vs, fs, paths):
                # TODO v_path should be without file extension
                # TODO remove the shapenet pointClouds download and from dataset
                v_path = p.replace("ShapeNetVox32", "ShapeNetMeshes")
                v_path = v_path.replace(".binvox", "")
                Path(v_path).parent.mkdir(parents=True, exist_ok=True)
                save_mesh(normalize_mesh(v), f, v_path)


if __name__ == "__main__":
    download_pix3d("/home/benbanuz")
    download_shapenet("/home/benbanuz")
