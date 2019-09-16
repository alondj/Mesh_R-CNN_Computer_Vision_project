import json
import matplotlib.image as mpimg
from torch.utils.data import Dataset
import scipy.io as sci
import numpy as np
from pathlib import Path
import data.read_binvox
import torch


class pix3dDataset(Dataset):

    def __init__(self, dataset_path, num_sampels=None):
        json_path = f"{dataset_path}/pix3d.json"
        with open(json_path) as json_file:
            dataset = json.load(json_file)
            self.models_vox_src = []
            self.imgs_src = []
            self.pointcloud = []
            self.masks = []
            self.bbox = []
            self.Class = []
            for i, p in enumerate(dataset):
                if num_sampels is not None and num_sampels == len(self.imgs_src):
                    break
                if p["img"].find("chair") != -1 or p["img"].find("sofa") != -1 or p["img"].find("table") != -1:
                    img_src = f"{dataset_path}/{p['img']}"
                    model3d_src = f"{dataset_path}/{p['voxel']}"
                    mask_src = f"{dataset_path}/{p['mask']}"

                    s = p['voxel']
                    beginning = s.find("voxel.mat")
                    s = s[0: beginning]
                    pointcloud_src = f"{dataset_path}/pointclouds/{s}pcl_1024.npy"

                    self.imgs_src.append(img_src)
                    self.models_vox_src.append(model3d_src)
                    self.pointcloud.append(pointcloud_src)
                    self.masks.append(mask_src)
                    self.bbox.append(torch.tensor(p['bbox']))
                    self.Class.append(p['category'])

    def __len__(self):
        return len(self.imgs_src)

    def __getitem__(self, idx):
        if type(idx) != slice:
            idx = slice(idx, idx + 1)
        if type(idx) == slice:
            img_src = self.imgs_src[idx]
            model_src = self.models_vox_src[idx]
            pointcloud_src = self.pointcloud[idx]
            masks_src = self.masks[idx]
            bbox = self.bbox[idx]
            label = self.Class[idx]

            imgs = []
            models = []
            clouds = []
            masks = []

            for img_s, model_s, pc_src, mask_s in zip(img_src, model_src, pointcloud_src, masks_src):
                imgs.append(torch.from_numpy(mpimg.imread(img_s)))
                models.append(torch.from_numpy(sci.loadmat(model_s)['voxel']))
                clouds.append(torch.from_numpy(np.load(pc_src)))
                masks.append(torch.from_numpy(mpimg.imread(mask_s)))

            targets = pix3DTargetList(masks, bbox, label)
            return imgs, models, clouds, targets


def get_class(s: str):
    if s.find("02691156") != -1:
        return "airplane"
    if s.find("02828884") != -1:
        return "bench"
    if s.find("02933112") != -1:
        return "drawer"
    if s.find("02958343") != -1:
        return "car"
    if s.find("03001627") != -1:
        return "chair"
    if s.find("03211117") != -1:
        return "TV"
    if s.find("03636649") != -1:
        return "lamp"
    if s.find("03691459") != -1:
        return "sterio"
    if s.find("04090263") != -1:
        return "gun"
    if s.find("04256520") != -1:
        return "sofa"
    if s.find("04379243") != -1:
        return "table"
    if s.find("04401088") != -1:
        return "phone"
    if s.find("04530566") != -1:
        return "boat"


class pix3DTarget():
    keys = ['boxes', 'masks', 'labels']

    def __init__(self, target: dict):
        if not(isinstance(target, dict) and all(k in target for k in self.keys)):
            raise ValueError(
                f"target must be a dictionary with keys {self.keys}")

        self.target = target

    def to(self, *args, **kwargs):
        for k in self.keys:
            self.target[k] = self.target[k].to(*args, **kwargs)
        return self

    def __getitem__(self, key):
        return self.target[key]

    def __setitem__(self, key, value):
        self.target[key] = value

    def __contains__(self, key):
        return key in self.target


class pix3DTargetList():
    def __init__(self, masks, boxes, labels):
        assert len(boxes) == len(labels)
        assert len(boxes) == len(masks)

        self.targets = []
        for mask, box, lb in zip(masks, boxes, labels):
            target = pix3DTarget({'masks': mask, 'boxes': box, 'labels': lb})
            self.targets.append(target)

    def to(self, *args, **kwargs):
        self.targets = [t.to(*args, **kwargs) for t in self.targets]
        return self

    def __getitem__(self, arg):
        return self.targets[arg]

    def __setitem__(self, key, value):
        self.targets[key] = value

    def __len__(self):
        return len(self.targets)


class shapeNet_Dataset(Dataset):

    def __init__(self, directory_in_str, num_sampels=None):
        pathlist = Path(directory_in_str).glob('**/*.binvox')
        self.imgs_src = []
        self.models_vox_src = []
        self.pointcloud = []
        self.label = []

        for i, path in enumerate(pathlist):
            if num_sampels is not None and i == num_sampels:
                break
            voxel_path = str(path)
            cloud_path = voxel_path.replace(
                "ShapeNetVox32", "ShapeNet_pointclouds")
            cloud_path = cloud_path.replace(
                "model.binvox", "pointcloud_1024.npy")
            img_path = voxel_path.replace("ShapeNetVox32", "ShapeNetRendering")
            img_path = img_path.replace("model.binvox", "rendering/00.png")

            self.models_vox_src.append(voxel_path)
            self.pointcloud.append(cloud_path)
            self.imgs_src.append(img_path)
            self.label.append(get_class(img_path))

    def __len__(self):
        return len(self.imgs_src)

    def __getitem__(self, idx):
        if type(idx) != slice:
            idx = slice(idx, idx + 1)
        if type(idx) == slice:
            img_src = self.imgs_src[idx]
            model_src = self.models_vox_src[idx]
            pointcloud_src = self.pointcloud[idx]
            label = self.label[idx]

            imgs = []
            models = []
            clouds = []

            for img_s, model_s, pc_src in zip(img_src, model_src, pointcloud_src):
                imgs.append(torch.from_numpy(mpimg.imread(img_s)))
                clouds.append(torch.from_numpy(np.load(pc_src)))
                with open(model_s, 'rb') as binvox_file:
                    models.append(torch.from_numpy(
                        data.read_binvox.read_as_3d_array(binvox_file)))

            return imgs, models, clouds, label


if __name__ == "__main__":
    pxd = pix3dDataset("../dataset/pix3d", 5)
    # sdb = shapeNet_Dataset("../dataset/shapeNet/ShapeNetVox32", 9)
    imgs, models, clouds, targets = pxd[0:1]
    print(models)
