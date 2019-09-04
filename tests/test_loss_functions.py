import torch
from loss_functions import batched_point2point_distance, total_edge_length
from utils import dummy


def test_p2p_distance():
    a = torch.arange(15).float().reshape(5, 3)

    p2p = batched_point2point_distance(a).squeeze()

    assert p2p.shape == torch.Size([5, 5])

    expected = torch.Tensor([[0, 9*3, 36*3, 81*3, 144*3],
                             [9*3, 0, 9*3, 36*3, 81*3],
                             [36*3, 9*3, 0, 9*3, 36*3],
                             [81*3, 36*3, 9*3, 0, 9*3],
                             [144*3, 81*3, 36*3, 9*3, 0]])

    # check l2 norm
    assert torch.allclose(
        expected, batched_point2point_distance(a, a).squeeze())

    assert torch.allclose(expected, p2p)

    p2p = batched_point2point_distance(a, a+1).squeeze()
    # check shape
    assert p2p.shape == torch.Size([5, 5])

    # check distance between 2 clouds
    b = torch.arange(9).float().reshape(3, 3)
    a_b = batched_point2point_distance(a, b)
    b_a = batched_point2point_distance(b, a)

    assert torch.allclose(a_b.transpose(-1, -2), b_a)

    # check batched distance
    a = torch.randn(10, 20, 3)
    b = torch.randn(10, 40, 3)

    a_b = batched_point2point_distance(a, b)
    b_a = batched_point2point_distance(b, a)
    aa = batched_point2point_distance(a)

    assert aa.shape == torch.Size([10, 20, 20])
    assert a_b.shape == torch.Size([10, 20, 40])
    assert b_a.shape == torch.Size([10, 40, 20])
    assert torch.allclose(a_b.transpose(-1, -2), b_a)


def test_edge_length():
    pos = dummy(1, 10, 3).squeeze()
    p2p = batched_point2point_distance(pos).squeeze()

    adj = torch.zeros(10, 10)
    adj[[0, 1, 2], [1, 0, 1]] = 1
    adj[1, 2] = 1

    expected = (p2p[0, 1]+p2p[1, 0])/2

    assert torch.allclose(expected, total_edge_length(p2p, adj))