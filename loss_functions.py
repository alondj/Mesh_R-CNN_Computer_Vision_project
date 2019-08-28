import torch
import torch.nn as nn
from torch import Tensor

# TODO maybe later we will use custom kernels for all of these ops
# losses of of mesh prediction network


def voxel_loss():
    # minimize binary cross entropy between predicted voxel occupancy probabilities and true voxel occupancies.
    pass


# mesh losses

def mesh_sampling():
    # described originally here https://arxiv.org/pdf/1901.11461.pdf
    # given a mesh sample a point cloud to be used in the loss functions
    pass


class ChamferDistance(nn.Module):
    # compute the chamder distance between point clouds
    # basically we sum the distance of each point in cloud A to it's nearest neighbour in cload B and vice versa
    # https://github.com/chrdiller/pyTorchChamferDistance cuda extention
    # https://github.com/345ishaan/DenseLidarNet/blob/master/code/chamfer_loss.py
    def __init__(self):
        super(ChamferDistance, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds: Tensor, gts: Tensor) -> Tensor:
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)

        return loss_1 + loss_2

    def batch_pairwise_dist(self, x: Tensor, y: Tensor):
        num_points_x, num_points_y = x.shape[1], y.shape[1]
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x).type(x.dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(y.dtype)
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(
            1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2*zz)
        return P


def normal_distance_between_point_clouds():
    pass


def edge_loss(vertex_positions: Tensor, vertex_adjacency: Tensor):
    ''' compute the edge loss as denoted by L(V,E) =1/|E| * ∑(v,v′)∈E ‖v−v′‖^2\n
        vertex_positions can be many vertices stacked along the vertix dimention\n
        vertex_adjacency can be many adjacency matrices stacked together in block diagonal format
    '''

    # xx[i,j] = pi dot pj
    xx = vertex_positions.mm(vertex_positions.t())

    rx = torch.diagonal(xx).expand_as(xx.t())

    dist_matrix = rx.t() + rx - 2*xx

    # we mask only (v,v′)∈E
    masked_dist_matrix = torch.mul(vertex_adjacency, dist_matrix)

    # normalize by the number of edges
    normalize_factor = torch.nonzero(masked_dist_matrix).shape[0]

    # we count each edge twice so when we normalize it cancels out 2*s /2|E|
    return masked_dist_matrix.sum() / normalize_factor


# losses of the backbone network


def mask_loss():
    pass


def box_loss():
    pass
