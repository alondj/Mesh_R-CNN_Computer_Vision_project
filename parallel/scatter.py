import numpy as np
import torch


def split_to_n(l, n):
    sizes = np.full(n, len(l) // n)
    sizes[:len(l) % n] += 1
    ends = np.cumsum(sizes)

    return[l[ends[i]-sizes[i]:ends[i]] for i in range(len(sizes))]


def custom_scatter(images, targets=None, target_gpus=None, pix3d=False):
    if targets is not None:
        assert len(images) == len(targets)
    if target_gpus is None:
        target_gpus = list(range(torch.cuda.device_count()))
    divided_imgs = split_to_n(images, n=len(target_gpus))
    if targets is not None:
        divided_trgts = split_to_n(targets, n=len(target_gpus))
    else:
        divided_trgts = [None for _ in target_gpus]

    data = []
    for imgs, trgts, device in zip(divided_imgs, divided_trgts, target_gpus):
        if pix3d:
            imgs = [img.to(device) for img in imgs]
        else:
            imgs = imgs.to(device)

        if trgts is not None:
            trgts = trgts.to(device)

        data.append((imgs, trgts))
    return data
