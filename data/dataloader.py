import json
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from data.read_binvox import read_as_3d_array
import torch
import PIL.Image
from torch import Tensor
from torch.nn.functional import adaptive_max_pool3d, interpolate
from typing import List
from utils.save import Mesh, load_mesh, load_voxels


def fit_voxels_to_shape(voxels: Tensor, N):
    """
    up/downsample a BxVxVxV voxel grid to a BxNxNxN grid
    """
    assert voxels.ndim == 4, "expects batched input of shape BxVxVxV"
    dtype = voxels.dtype
    M = voxels.shape[1]
    assert voxels.shape[1:] == torch.Size([M, M, M])

    if M > N:
        # downsample
        return adaptive_max_pool3d(voxels.to(torch.float32), N).to(dtype)
    elif M < N:
        # upsample
        return interpolate(voxels.to(torch.float32).unsqueeze(1), size=N).squeeze(1).to(dtype)

    return voxels


class Batch():
    def __init__(self, images, voxels, num_voxels, meshes, targets):
        # fit voxels to shape BxVxVxV
        batched_models = torch.stack(voxels)
        batched_models = fit_voxels_to_shape(batched_models, num_voxels)

        # stack all meshes together ∑Vx3 ∑fx3
        batched_vertices = torch.cat([m.vertices for m in meshes])
        batched_faces = torch.cat([m.faces for m in meshes])
        batched_meshes = Mesh(batched_vertices, batched_faces)

        # create the index with which to separate the batched meshes
        mesh_index = [1 for _ in images]
        vertice_index = [m.vertices.shape[0] for m in meshes]
        face_index = [m.faces.shape[0] for m in meshes]

        self.images = images
        self.voxels = batched_models
        self.meshes = batched_meshes
        self.mesh_index = mesh_index
        self.vertice_index = vertice_index
        self.face_index = face_index
        self.targets = targets

    def to(self, *args, **kwargs):
        if not (self.images is None):
            if isinstance(self.images, (list, tuple)):
                self.images = type(self.images)(
                    [i.to(*args, **kwargs) for i in self.images])
            else:
                assert isinstance(self.images, Tensor)
                self.images = self.images.to(*args, **kwargs)
        if not (self.voxels is None):
            assert isinstance(self.voxels, torch.Tensor)
            self.voxels = self.voxels.to(*args, **kwargs)
        if not (self.meshes is None):
            assert isinstance(self.meshes, Mesh)
            assert self.face_index != None
            assert self.vertice_index != None
            assert self.mesh_index != None
            self.meshes = Mesh(self.meshes.vertices.to(*args, **kwargs),
                               self.meshes.faces.to(*args, **kwargs))
        if not (self.targets is None):
            self.targets = self.targets.to(*args, **kwargs)

        return self


class pix3dDataset(Dataset):

    def __init__(self, dataset_path, num_sampels=None):
        json_path = f"{dataset_path}/pix3d.json"
        with open(json_path) as json_file:
            dataset = json.load(json_file)
            self.models_vox_src = []
            self.imgs_src = []
            self.mesh_src = []
            self.masks = []
            self.bbox = []
            self.Class = []
            for p in dataset:
                if num_sampels is not None and num_sampels == len(self.imgs_src):
                    break
                img_src = f"{dataset_path}/{p['img']}"
                voxel_src = f"{dataset_path}/{p['voxel']}"
                mesh_src = f"{dataset_path}/{p['model']}"
                mask_src = f"{dataset_path}/{p['mask']}"

                self.mesh_src.append(mesh_src)
                self.imgs_src.append(img_src)
                self.models_vox_src.append(voxel_src)
                self.masks.append(mask_src)
                self.bbox.append(torch.Tensor(p['bbox']))
                self.Class.append(self.get_class(p['img']))

    def get_class(self, s: str):
        if s.find("bed") != -1:
            return 1
        if s.find("bookcase") != -1:
            return 2
        if s.find("chair") != -1:
            return 3
        if s.find("desk") != -1:
            return 4
        if s.find("misc") != -1:
            return 5
        if s.find("sofa") != -1:
            return 6
        if s.find("table") != -1:
            return 7
        if s.find("tool") != -1:
            return 8
        if s.find("wardrobe") != -1:
            return 9
        assert False, "no label found for pix3d should not happen"
        return -1

    def __len__(self):
        return len(self.imgs_src)

    def __getitem__(self, idx):
        img_src = self.imgs_src[idx]
        voxel_src = self.models_vox_src[idx]
        mesh_src = self.mesh_src[idx]
        masks_src = self.masks[idx]
        bbox = self.bbox[idx]
        label = torch.tensor(self.Class[idx])

        img = torch.from_numpy(mpimg.imread(img_src)).permute(2, 0, 1)
        model = load_voxels(voxel_src, tensor=True)
        mesh = load_mesh(mesh_src, tensor=True)
        mask = torch.from_numpy(mpimg.imread(masks_src))

        target = pix3DTarget({'masks': mask, 'boxes': bbox, 'labels': label})
        return img, model, mesh, target


class pix3DTarget():
    keys = ['boxes', 'masks', 'labels']

    def __init__(self, target: dict):
        if not(isinstance(target, dict) and all(k in target for k in self.keys)):
            raise ValueError(
                f"target must be a dictionary with keys {self.keys}")

        self.target = target

    def to(self, *args, **kwargs):
        target = {}
        for k in self.keys:
            target[k] = self.target[k].to(*args, **kwargs)
        return pix3DTarget(target)

    def __getitem__(self, key):
        return self.target[key]

    def __setitem__(self, key, value):
        self.target[key] = value

    def __contains__(self, key):
        return key in self.target


class pix3DTargetList():
    def __init__(self, pix3d_targets: List[pix3DTarget]):
        self.targets = pix3d_targets

    def to(self, *args, **kwargs):
        targets = [t.to(*args, **kwargs) for t in self.targets]
        return pix3DTargetList(targets)

    def __getitem__(self, arg):
        return self.targets[arg]

    def __setitem__(self, key, value):
        self.targets[key] = value

    def __len__(self):
        return len(self.targets)


def preparte_pix3dBatch(num_voxels: int):
    def batch_input(samples: List) -> Batch:
        images, voxel_gts, meshes, targets = zip(*samples)
        backbone_targets = pix3DTargetList(targets)

        return Batch(images=images, voxels=voxel_gts,
                     num_voxels=num_voxels, meshes=meshes,
                     targets=backbone_targets)

    return batch_input


def pix3dDataLoader(dataset: Dataset, batch_size: int, num_voxels: int, num_workers: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers,
                      collate_fn=preparte_pix3dBatch(num_voxels))


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
            self.label.append(self.get_class(img_path))

    def get_class(self, s: str):
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
        assert False, "no label found for shapenet should not happen"
        return -1

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

        return img, model, cloud, label


def preparte_shapeNetBatch(num_voxels: int):
    def batch_input(samples: List) -> Batch:
        images, voxel_gts, meshes, targets = zip(*samples)
        # batch images BxCxHxW
        images = torch.stack(images)
        targets = torch.Tensor(targets)

        return Batch(images=images, voxels=voxel_gts,
                     num_voxels=num_voxels, meshes=meshes,
                     targets=targets)

    return batch_input


def shapenetDataLoader(dataset: Dataset, batch_size: int, num_voxels: int, num_workers: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers,
                      collate_fn=preparte_shapeNetBatch(num_voxels))
