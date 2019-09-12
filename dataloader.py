import json
import matplotlib.image as mpimg
from torch.utils.data import Dataset
import scipy.io as sci
import numpy as np
from pathlib import Path
import read_binvox


class pix3dDataset(Dataset):

    def __init__(self, dataset_path):
        with open(dataset_path) as json_file:
            dataset = json.load(json_file)
            self.models_vox_src = []
            self.imgs_src = []
            self.pointcloud = []

            for p in dataset:
                if p["img"].find("chair") != -1 or p["img"].find("sofa") != -1 or p["img"].find("table") != -1:
                    img_src = f"dataset/pix3d/{p['img']}"
                    model3d_src = f"dataset/pix3d/{p['voxel']}"

                    s = p['voxel']
                    beginning = s.find("voxel.mat")
                    s = s[0: beginning]
                    pointcloud_src = f"dataset/pix3d/pointclouds/{s}pcl_1024.npy"

                    self.imgs_src.append(img_src)
                    self.models_vox_src.append(model3d_src)
                    self.pointcloud.append(pointcloud_src)

    def __len__(self):
        return len(self.imgs_src)

    def __getitem__(self, idx):
        if type(idx) != slice:
            idx = slice(idx, idx + 1)
        if type(idx) == slice:
            img_src = self.imgs_src[idx]
            model_src = self.models_vox_src[idx]
            pointcloud_src = self.pointcloud[idx]

            imgs = []
            models = []
            clouds = []

            for img_s, model_s, pc_src in zip(img_src, model_src, pointcloud_src):
                imgs.append(mpimg.imread(img_s))
                models.append(sci.loadmat(model_s)['voxel'])
                clouds.append(np.load(pc_src))

            return imgs, models, clouds


class shapeNet_Dataset(Dataset):

    def __init__(self, directory_in_str, num_sampels=None):
        pathlist = Path(directory_in_str).glob('**/*.binvox')
        self.imgs_src = []
        self.models_vox_src = []
        self.pointcloud = []
        for i, path in enumerate(pathlist):
            if num_sampels is not None and i == num_sampels:
                break
            voxel_path = str(path)
            cloud_path = voxel_path.replace("ShapeNetVox32", "ShapeNet_pointclouds")
            cloud_path = cloud_path.replace("model.binvox", "pointcloud_1024.npy")
            img_path = voxel_path.replace("ShapeNetVox32", "ShapeNetRendering")
            img_path = img_path.replace("model.binvox", "rendering/00.png")

            self.models_vox_src.append(voxel_path)
            self.pointcloud.append(cloud_path)
            self.imgs_src.append(img_path)

    def __len__(self):
        return len(self.imgs_src)

    def __getitem__(self, idx):
        if type(idx) != slice:
            idx = slice(idx, idx + 1)
        if type(idx) == slice:
            img_src = self.imgs_src[idx]
            model_src = self.models_vox_src[idx]
            pointcloud_src = self.pointcloud[idx]

            imgs = []
            models = []
            clouds = []

            for img_s, model_s, pc_src in zip(img_src, model_src, pointcloud_src):
                imgs.append(mpimg.imread(img_s))
                clouds.append(np.load(pc_src))
                with open(model_s, 'rb') as binvox_file:
                    models.append(read_binvox.read_as_3d_array(binvox_file))

            return imgs, models, clouds


if __name__ == "__main__":
    pxd = pix3dDataset("dataset/pix3d/pix3d.json")
    #sdb = shapeNet_Dataset("../dataset/shapeNet/ShapeNetVox32", 9)
    imgs, models, clouds = pxd[0:3]
    print(imgs[0].shape)
