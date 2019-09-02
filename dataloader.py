import json
import matplotlib.image as mpimg
from torch.utils.data import Dataset
import torch


class pix3dDataset(Dataset):

    def __init__(self, dataset_path):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        with open(dataset_path) as json_file:
            dataset = json.load(json_file)
            self.models = []
            self.imgs = []

            for p in dataset:
                img_src = f"dataset/pix3d/{p['img']}"
                img = mpimg.imread(img_src)
                model3d_src = p["model"]

                self.imgs.append(img)
                self.models.append(model3d_src)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.imgs[idx], self.models[idx]


if __name__ == "__main__":
    pxd = pix3dDataset("dataset/pix3d/pix3d.json")
