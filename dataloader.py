import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# returns 2 lists of imgs and 3d models where the i-th img matches the i-th 3d model
def get_img_and_model(dataset_path, num_sampels=None):
    with open(dataset_path) as json_file:
        dataset = json.load(json_file)
        models = []
        imgs = []

        for i, p in enumerate(dataset):
            if i == num_sampels and num_sampels is not None:
                break
            img_src = f"dataset/pix3d/{p['img']}"
            img = mpimg.imread(img_src)
            model3d_src = p["model"]

            imgs.append(img)
            models.append(model3d_src)
        return imgs, models


if __name__ == "__main__":
    imgs, models = get_img_and_model("dataset/pix3d/pix3d.json", 3)
    print(models[1])
