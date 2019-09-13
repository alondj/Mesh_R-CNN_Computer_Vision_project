import urllib.request
import zipfile
import os
import shutil

"""
the dataset should be built like that:
-dataset
----shapeNet
-------ShapeNet_pointclouds
-------ShapeNetRendering
-------ShapeNetVox32
----pix3d
-------img
-------mask
-------model
-------pointclouds
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
        zip_ref.extractall(f"{download_path}/dataset")
    print("deleting the zip file")
    os.remove(zip_download_path)

    url = "https://drive.google.com/uc?authuser=0&id=1RZakyBu9lPbG85SyconBn4sR8r2faInV&export=download"
    zip_download_path = f"{download_path}/pointclouds.zip"
    print("downloading pix3d pointclouds zip")
    urllib.request.urlretrieve(url, zip_download_path)
    print("unzipping pix3d_pointclouds.zip")
    with zipfile.ZipFile(zip_download_path, 'r') as zip_ref:
        zip_ref.extractall(f"{download_path}/dataset/pix3d")
    print("deleting the zip file")
    os.remove(zip_download_path)
    src_dic = f"{download_path}/dataset/pix3d/data/pix3d/pointclouds"
    dest_dic = f"{download_path}/dataset/pix3d"
    shutil.move(src_dic, dest_dic)
    os.rmdir(f"{download_path}/dataset/pix3d/data/pix3d")
    os.rmdir(f"{download_path}/dataset/pix3d/data")
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
    zip_download_path = f"{download_path}/dataset/shapeNet/ShapeNetRendering.zip"
    urllib.request.urlretrieve(url, zip_download_path)
    print("unzipping")
    with zipfile.ZipFile(zip_download_path, 'r') as zip_ref:
        zip_ref.extractall(f"{download_path}/dataset/shapeNet")
    print("deleting the zip file")
    os.remove(zip_download_path)

    print("downloading shapNet voxel zip")
    url = "http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz"
    zip_download_path = f"{download_path}/dataset/shapeNet/ShapeNetVox32.zip"
    urllib.request.urlretrieve(url, zip_download_path)
    print("unzipping")
    with zipfile.ZipFile(zip_download_path, 'r') as zip_ref:
        zip_ref.extractall(f"{download_path}/dataset/shapeNet")
    print("deleting the zip file")
    os.remove(zip_download_path)

    print("downloading shapeNet pointclouds")
    url = "https://drive.google.com/uc?export=download&confirm=jbpW&id=1cfoe521iTgcB_7-g_98GYAqO553W8Y0g"
    zip_download_path = f"{download_path}/dataset/shapeNet/ShapeNet_pointclouds.zip"
    urllib.request.urlretrieve(url, zip_download_path)
    print("unzipping")
    with zipfile.ZipFile(zip_download_path, 'r') as zip_ref:
        zip_ref.extractall(f"{download_path}/dataset/shapeNet")
    print("deleting the zip file")
    os.remove(zip_download_path)
    print("finished shapeNet")


if __name__ == "__main__":
    download_pix3d("D:")
    download_shapenet()