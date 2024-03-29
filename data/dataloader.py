import json
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import torch
import PIL.Image
from typing import List
from utils import Mesh, load_mesh, load_voxels, resample_voxels


class Batch():
    def __init__(self, images, voxels, num_voxels, meshes, backbone_targets):
        if isinstance(voxels, torch.Tensor):
            batched_models = voxels
        else:
            batched_models = torch.stack(voxels)

        if batched_models.shape[1:] != torch.Size([num_voxels]*3):
            batched_models = resample_voxels(batched_models, num_voxels)

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
        self.backbone_targets = backbone_targets

    def to(self, *args, **kwargs):
        if not (self.images is None):
            if isinstance(self.images, (list, tuple)):
                self.images = type(self.images)(
                    [i.to(*args, **kwargs) for i in self.images])
            else:
                assert isinstance(self.images, torch.Tensor)
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
        if not (self.backbone_targets is None):
            self.backbone_targets = self.backbone_targets.to(*args, **kwargs)

        return self

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self[idx:idx+1]

        images = self.images[idx]
        voxels = self.voxels[idx]
        backbone_targets = self.backbone_targets[idx]
        num_voxels = voxels.shape[1]

        meshes = [Mesh(v, f) for v, f in zip(self.meshes.vertices.split(self.vertice_index)[idx],
                                             self.meshes.faces.split(self.face_index)[idx])]

        return Batch(images, voxels, num_voxels, meshes, backbone_targets)

    def __len__(self):
        return len(self.images)


class pix3dDataset(Dataset):
    category_idx = {"bed": 1,
                    "bookcase": 2,
                    "chair": 3,
                    "desk": 4,
                    "misc": 5,
                    "sofa": 6,
                    "table": 7,
                    "tool": 8,
                    "wardrobe": 9}

    def __init__(self, dataset_path, classes=None):
        json_path = f"{dataset_path}/pix3d.json"
        with open(json_path) as json_file:
            dataset = json.load(json_file)
            self.voxels_src = []
            self.imgs_src = []
            self.mesh_src = []
            self.masks = []
            self.bbox = []
            self.Class = []
            for p in dataset:
                if classes is not None and p['category'] not in classes:
                    continue
                img_src = f"{dataset_path}/{p['img']}"
                voxel_src = f"{dataset_path}/{p['voxel']}"
                mesh_src = f"{dataset_path}/{p['model']}"
                mask_src = f"{dataset_path}/{p['mask']}"
                label = p['category']

                # only rgb images with 3 channles
                try:
                    img = torch.from_numpy(mpimg.imread(img_src))
                    if img.ndim != 3 or img.shape[2] != 3:
                        continue
                except Exception as _:
                    continue
                self.mesh_src.append(mesh_src)
                self.imgs_src.append(img_src)
                self.voxels_src.append(voxel_src)
                self.masks.append(mask_src)
                self.bbox.append(torch.Tensor(p['bbox']).unsqueeze(0))
                self.Class.append(self.get_class(label))

    def get_class(self, s: str):
        idx = self.category_idx.get(s, -1)
        assert idx != -1, "no label found for pix3d should not happen"
        return idx

    def __len__(self):
        return len(self.imgs_src)

    def __getitem__(self, idx):
        img_src = self.imgs_src[idx]
        voxel_src = self.voxels_src[idx]
        mesh_src = self.mesh_src[idx]
        masks_src = self.masks[idx]
        bbox = self.bbox[idx]
        label = torch.tensor(self.Class[idx]).unsqueeze(0)
        img = torch.from_numpy(mpimg.imread(img_src))
        img = img.permute(2, 1, 0)
        img = img.type(torch.FloatTensor)
        # normalize to 0-1
        if img.max() > 1.:
            img = img / 255.
        model = load_voxels(voxel_src, tensor=True)
        mesh = load_mesh(mesh_src, tensor=True)
        mask = torch.from_numpy(mpimg.imread(
            masks_src)).unsqueeze(0).permute(0, 2, 1)
        target = pix3DTarget({'masks': mask, 'boxes': bbox, 'labels': label})
        return img, model, mesh, target


class pix3DTarget():
    keys = ['boxes', 'masks', 'labels']

    def __init__(self, target: dict):
        if not (isinstance(target, dict) and all(k in target for k in self.keys)):
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
    def __init__(self, pix3d_backbone_targets: List[pix3DTarget]):
        self.backbone_targets = pix3d_backbone_targets

    def to(self, *args, **kwargs):
        backbone_targets = [t.to(*args, **kwargs)
                            for t in self.backbone_targets]
        return pix3DTargetList(backbone_targets)

    def __getitem__(self, arg):
        if isinstance(arg, slice):
            return pix3DTargetList(self.backbone_targets[arg])
        return self.backbone_targets[arg]

    def __setitem__(self, key, value):
        self.backbone_targets[key] = value

    def __len__(self):
        return len(self.backbone_targets)


def preparte_pix3dBatch(num_voxels: int):
    def batch_input(samples: List) -> Batch:
        images, voxel_gts, meshes, backbone_targets = zip(*samples)
        pix3d_backbone_targets = pix3DTargetList(backbone_targets)

        return Batch(images=images, voxels=voxel_gts,
                     num_voxels=num_voxels, meshes=meshes,
                     backbone_targets=pix3d_backbone_targets)

    return batch_input


class shapeNet_Dataset(Dataset):
    category_idx = {"airplane": 0,
                    "bench": 1,
                    "closet": 2,
                    "car": 3,
                    "chair": 4,
                    "tv": 5,
                    "lamp": 6,
                    "stereo": 7,
                    "gun": 8,
                    "sofa": 9,
                    "table": 10,
                    "phone": 11,
                    "ship": 12}

    def __init__(self, dataset_path, classes=None):
        json_path = f"{dataset_path}/shapenet.json"
        with open(json_path) as json_file:
            dataset = json.load(json_file)
            self.voxels_src = []
            self.imgs_src = []
            self.mesh_src = []
            self.label = []
            for p in dataset:
                if classes is not None and p['category'] not in classes:
                    continue
                img_src = p['img']
                voxel_src = p['voxel']
                mesh_src = p['model']
                label = p['category']

                try:
                    rgba_image = PIL.Image.open(img_src)
                    rgb_image = rgba_image.convert('RGB')
                except Exception as _:
                    continue

                self.mesh_src.append(mesh_src)
                self.imgs_src.append(img_src)
                self.voxels_src.append(voxel_src)
                self.label.append(self.get_class(label))

    def __len__(self):
        return len(self.imgs_src)

    def __getitem__(self, idx):
        img_src = self.imgs_src[idx]
        voxel_src = self.voxels_src[idx]
        mesh_src = self.mesh_src[idx]
        label = self.label[idx]

        rgba_image = PIL.Image.open(img_src)
        rgb_image = rgba_image.convert('RGB')
        img = torch.from_numpy(np.array(rgb_image))
        img = img.transpose(2, 0)
        img = img.type(torch.FloatTensor)

        # normalize to 0-1
        if img.max() > 1.:
            img = img / 255.

        mesh = load_mesh(mesh_src, tensor=True)
        model = load_voxels(voxel_src, tensor=True)
        return img, model, mesh, float(label)

    def get_class(self, s: str):
        idx = self.category_idx.get(s, -1)
        assert idx != -1, "no label found for shapenet should not happen"
        return idx


def preparte_shapeNetBatch(num_voxels: int):
    def batch_input(samples: List) -> Batch:
        images, voxel_gts, meshes, backbone_targets = zip(*samples)
        # batch images BxCxHxW
        images = torch.stack(images)
        backbone_targets = torch.Tensor(backbone_targets)

        return Batch(images=images, voxels=voxel_gts,
                     num_voxels=num_voxels, meshes=meshes,
                     backbone_targets=backbone_targets)

    return batch_input


def dataLoader(dataset: Dataset, batch_size: int, num_voxels: int, num_workers: int, test=False, num_train_samples=None,
               train_ratio=None):
    assert (train_ratio is None) or (
        num_train_samples is None), "at most one of train_ration and num_train_samples can set"

    indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    if train_ratio is None and num_train_samples is None:
        train_ratio = 1.

    if train_ratio != None:
        assert 0 < train_ratio <= 1.
        num_train_samples = int(np.floor(len(dataset) * train_ratio))

    if num_train_samples != None:
        assert 0 < num_train_samples <= len(dataset)
        train_indices = indices[:num_train_samples]
        test_indices = indices[num_train_samples:]

    if test:
        sampler = SubsetRandomSampler(test_indices)
    else:
        sampler = SubsetRandomSampler(train_indices)

    if isinstance(dataset, pix3dDataset):
        batch_fn = preparte_pix3dBatch
    else:
        assert isinstance(dataset, shapeNet_Dataset)
        batch_fn = preparte_shapeNetBatch

    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, sampler=sampler,
                      collate_fn=batch_fn(num_voxels))


if __name__ == "__main__":
    ds = shapeNet_Dataset("../../dataset/shapeNet")
    img, model, label = ds[1]
    import matplotlib.pyplot as plt

    print(img.shape)
    img = img.transpose(2, 0)
    imgplot = plt.imshow(img)
    plt.show()
