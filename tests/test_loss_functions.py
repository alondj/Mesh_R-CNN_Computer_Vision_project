import torch
from loss_functions import batched_point2point_distance, total_edge_length, batched_chamfer_distance, \
    face_sampling_probabilities_by_surface_area, mesh_sampling
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


def test_chamfer_distance():
    pt0 = dummy(1, 10, 3)
    pt1 = dummy(1, 7, 3)+1

    p2p = batched_point2point_distance(pt0, pt1)
    l0, idx0, l1, idx1 = batched_chamfer_distance(p2p)

    assert idx0.shape == torch.Size([1, 10])
    assert idx1.shape == torch.Size([1, 7])
    assert l0.item() == 300
    assert l1.item() == 21

    pt0 = dummy(1, 10, 3).expand(2, 10, 3)
    pt1 = (dummy(1, 7, 3)+1).expand(2, 7, 3)
    p2p = batched_point2point_distance(pt0, pt1)

    l0, idx0, l1, idx1 = batched_chamfer_distance(p2p)
    assert idx0.shape == torch.Size([2, 10])
    assert idx1.shape == torch.Size([2, 7])
    assert l0.item() == 600
    assert l1.item() == 42


def test_face_probas():
    pos = torch.Tensor([[0, 0, 0],
                        [1, 0, 0],
                        [1, 1, 1],  # 2
                        [0, 0, 2],
                        [0, 2, 0],
                        [0, 1, 5],  # 5
                        [2, 2, 2],
                        [2, 7, 0],
                        [2, 3, 5],  # 8
                        [2, 7, 8],
                        [0, 3, 2]])
    faces = torch.LongTensor([[1, 2, 8],
                              [3, 4, 5],
                              [0, 1, 7],
                              [6, 9, 10],
                              ])

    probas = face_sampling_probabilities_by_surface_area(pos, faces)

    areas = torch.Tensor([1.22474, 4., 3.5, 8.3666])
    expected_probas = areas / areas.sum()

    assert probas.size(0) == faces.shape[0]
    assert torch.allclose(expected_probas, probas)


def test_sampling():
    pos = torch.Tensor([[0, 0, 0],
                        [1, 0, 0],
                        [1, 1, 1],  # 2
                        [0, 0, 2],
                        [0, 2, 0],
                        [0, 1, 5],  # 5
                        [2, 2, 2],
                        [2, 7, 0],
                        [2, 3, 5],  # 8
                        [2, 7, 8],
                        [0, 3, 2]])
    faces = torch.LongTensor([[1, 2, 8],
                              [3, 4, 5],
                              [0, 1, 7],
                              [6, 9, 10],
                              ])

    pt = mesh_sampling(pos, faces, num_points=2000)

    assert pt.shape == torch.Size([2000, 3])
