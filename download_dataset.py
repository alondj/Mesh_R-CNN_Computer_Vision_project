import urllib.request
import zipfile
import os
import tarfile
from pathlib import Path
import json
import argparse
import torch
from model.layers import Cubify
from data.read_binvox import read_as_3d_array
from utils import save_mesh, normalize_mesh
"""
the dataset should be built like that:
-dataset
----shapeNet
-------ShapeNetRendering
-------ShapeNetVox32
-------ShapeNetMeshes
-------shapenet.json
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    renderer = Cubify(threshold=0.5).to(device)

    pathlist = list(Path(download_path).glob('**/*.binvox'))

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
            v_batch = torch.stack(voxels).float().to(device)

            # render and split again
            vs, v_index, faces, f_index, _ = renderer(v_batch)

            vs = vs.split(v_index)
            fs = faces.split(f_index)

            # save the normalized meshes
            for v, f, p in zip(vs, fs, paths):
                v_path = p.replace("ShapeNetVox32", "ShapeNetMeshes")
                v_path = v_path.replace(".binvox", "")
                Path(v_path).parent.mkdir(parents=True, exist_ok=True)
                save_mesh(normalize_mesh(v), f, v_path)


def get_shapenet_class_by_name(s: str):
    if s.find("02691156") != -1:
        return "airplane"
    if s.find("02828884") != -1:
        return "bench"
    if s.find("02933112") != -1:
        return "closet"
    if s.find("02958343") != -1:
        return "car"
    if s.find("03001627") != -1:
        return "chair"
    if s.find("03211117") != -1:
        return "tv"
    if s.find("03636649") != -1:
        return "lamp"
    if s.find("03691459") != -1:
        return "stereo"
    if s.find("04090263") != -1:
        return "gun"
    if s.find("04256520") != -1:
        return "sofa"
    if s.find("04379243") != -1:
        return "table"
    if s.find("04401088") != -1:
        return "phone"
    if s.find("04530566") != -1:
        return "ship"
    assert False, "no label found for shapenet should not happen"
    return -1


def create_shapenet_Json(directory_str):
    json_obj = []
    pathlist = Path(directory_str).glob('**/*.binvox')

    for path in pathlist:
        voxel_path = str(path)
        mesh_src = voxel_path.replace(
            "ShapeNetVox32", "ShapeNetMeshes")
        mesh_src.replace(".binvox", ".obj")
        img_path = voxel_path.replace("ShapeNetVox32", "ShapeNetRendering")
        img_path = img_path.replace("model.binvox", "rendering/00.png")
        category = get_shapenet_class_by_name(img_path)

        json_obj.append({"img": img_path, "category": category,
                         "voxel": voxel_path, "model": mesh_src})

    with open(f"{directory_str}/shapeNet/shapenet.json", "a+") as f:
        json.dump(json_obj, f)


def prepare_shapenet(download_path):
    download_shapenet(download_path)
    render_shapenet_meshes(download_path)
    create_shapenet_Json(download_path)


parser = argparse.ArgumentParser()
parser.add_argument('--download_path', type=str,
                    help='path to where the datasets will be stored')


if __name__ == "__main__":
    download_path = parser.parse_args().download_path
    download_pix3d(download_path)
    prepare_shapenet(download_path)
