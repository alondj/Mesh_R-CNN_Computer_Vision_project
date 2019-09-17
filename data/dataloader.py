import json
import matplotlib.image as mpimg
from torch.utils.data import Dataset
import scipy.io as sci
import numpy as np
from pathlib import Path
from data.read_binvox import read_as_3d_array
import torch
import PIL.Image
from torch import Tensor
from torch.nn.functional import adaptive_max_pool3d, interpolate


def fit_voxels_to_shape(voxels: Tensor, N):
    """
    up/downsample a BxVxVxV voxel grid to a BxNxNxN grid
    """
    assert voxels.ndim == 4, "expects batched input of shape BxVxVxV"

    M = voxels.shape[1]
    assert voxels.shape[1:] == torch.Size([M, M, M])

    if M > N:
        # downsample
        return adaptive_max_pool3d(voxels, N)
    elif M < N:
        # upsample
        return interpolate(voxels.unsqueeze(1), size=N).squeeze(1)

    return voxels


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

                    if p["img"].find("chair") != -1:
                        self.Class.append(torch.tensor(0))
                    if p["img"].find("sofa") != -1:
                        self.Class.append(torch.tensor(1))
                    if p["img"].find("table") != -1:
                        self.Class.append(torch.tensor(2))

    def __len__(self):
        return len(self.imgs_src)

    def __getitem__(self, idx):
        img_src = self.imgs_src[idx]
        model_src = self.models_vox_src[idx]
        pointcloud_src = self.pointcloud[idx]
        masks_src = self.masks[idx]
        bbox = self.bbox[idx]
        label = self.Class[idx]

        img = torch.from_numpy(mpimg.imread(img_src))
        model = torch.from_numpy(sci.loadmat(model_src)['voxel'])
        cloud = torch.from_numpy(np.load(pointcloud_src))
        mask = torch.from_numpy(mpimg.imread(masks_src))

        target = pix3DTarget({'masks': mask, 'boxes': bbox, 'labels': label})
        return img, model, cloud, target


def get_class(s: str):
    if s.find("02691156") != -1:
        return 0
    if s.find("02828884") != -1:
        return 1
    if s.find("02933112") != -1:
        return 2
    if s.find("02958343") != -1:
        return 3
    if s.find("03001627") != -1:
        return 4
    if s.find("03211117") != -1:
        return 5
    if s.find("03636649") != -1:
        return 6
    if s.find("03691459") != -1:
        return 7
    if s.find("04090263") != -1:
        return 8
    if s.find("04256520") != -1:
        return 9
    if s.find("04379243") != -1:
        return 10
    if s.find("04401088") != -1:
        return 11
    if s.find("04530566") != -1:
        return 12


class pix3DTarget():
    keys = ['boxes', 'masks', 'labels']

    def __init__(self, target: dict):
        if not (isinstance(target, dict) and all(k in target for k in self.keys)):
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
        img_src = self.imgs_src[idx]
        model_src = self.models_vox_src[idx]
        pc_src = self.pointcloud[idx]
        label = self.label[idx]

        rgba_image = PIL.Image.open(img_src)
        rgb_image = rgba_image.convert('RGB')
        img = torch.from_numpy(np.array(rgb_image))
        img = img.transpose(2, 0)
        img = img.type(torch.FloatTensor)

        cloud = torch.from_numpy(np.load(pc_src))
        with open(model_src, 'rb') as binvox_file:
            model = torch.from_numpy(read_as_3d_array(binvox_file))

        return img, model, cloud, torch.tensor(label)


if __name__ == "__main__":
    pxd = pix3dDataset("../dataset/pix3d", 5)
    # sdb = shapeNet_Dataset("../dataset/shapeNet/ShapeNetVox32", 9)
    imgs, models, clouds, targets = pxd[0]
    print(imgs.shape)
