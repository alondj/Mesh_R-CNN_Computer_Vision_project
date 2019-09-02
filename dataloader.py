import json
import matplotlib.image as mpimg
from torch.utils.data import Dataset
import torch


class pix3dDataset(Dataset):

    def __init__(self, dataset_path):
        with open(dataset_path) as json_file:
            dataset = json.load(json_file)
            self.models_src = []
            self.imgs_src = []

            for p in dataset:
                img_src = f"dataset/pix3d/{p['img']}"
                model3d_src = f"dataset/pix3d/{p['model']}"
                self.imgs_src.append(img_src)
                self.models_src.append(model3d_src)

    def __len__(self):
        return len(self.imgs_src)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_src = [self.imgs_src[idx]]
        model_src = [self.models_src[idx]]

        imgs = []
        models = []
        for img_s, model_s in zip(img_src, model_src):
            imgs.append(mpimg.imread(img_s))
            models.append(model_s)

        return imgs, models


if __name__ == "__main__":
    pxd = pix3dDataset("dataset/pix3d/pix3d.json")
    img, _ = pxd[0]
    print(img)
