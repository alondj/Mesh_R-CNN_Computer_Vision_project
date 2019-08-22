import torch
import torch.nn as nn


class GraphConv(nn.Module):
    # f′i = ReLU(W0xfi +∑j∈N(i)W1xfj)
    pass


class ResGraphConv(nn.Module):
    # ResGraphConv(D1→D2)consists of two graph convolution layers (each preceeded by ReLU)
    # and an additive skip connection, with a linear projection if the input and output dimensions are different
    pass


class VertexRefine(nn.Module):
    # eplained in the article
    pass


class VertexAlign(nn.Module):
    # explained in the article http://openaccess.thecvf.com/content_ECCV_2018/papers/Nanyang_Wang_Pixel2Mesh_Generating_3D_ECCV_2018_paper.pdf
    # as perceptual feature pooling
    pass


class Cubify(nn.Module):
    # explained in the article
    pass


class VoxelBranch(nn.Sequential):
    # explained in the article
    def __init__(self, in_channels, out_channels):
        super(VoxelBranch, self).__init__(
            # N x in_channels x V/2 x V/2
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            # N x 256 x V/2 x V/2
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # N x 256 x V/2 x V/2
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            # N x 256 x V x V
            nn.Conv2d(256, out_channels, kernel_size=1)
            # N x out_channels x V x V
        )
