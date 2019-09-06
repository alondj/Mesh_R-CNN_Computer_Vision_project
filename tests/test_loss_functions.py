import pytest
import torch
from loss_functions import batched_point2point_distance, total_edge_length, batched_chamfer_distance, \
    surface_areas, mesh_sampling
from utils import dummy

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')


@pytest.mark.parametrize('device', devices)
def test_p2p_distance(device):
    a = torch.arange(15).float().reshape(5, 3).to(device)

    p2p = batched_point2point_distance(a).squeeze()

    assert p2p.shape == torch.Size([5, 5])

    expected = torch.Tensor([[0, 9*3, 36*3, 81*3, 144*3],
                             [9*3, 0, 9*3, 36*3, 81*3],
                             [36*3, 9*3, 0, 9*3, 36*3],
                             [81*3, 36*3, 9*3, 0, 9*3],
                             [144*3, 81*3, 36*3, 9*3, 0]]).to(device)

    # check l2 norm
    assert torch.allclose(expected,
                          batched_point2point_distance(a, a).squeeze())

    assert torch.allclose(expected, p2p, rtol=1e-5)

    p2p = batched_point2point_distance(a, a+1).squeeze()
    # check shape
    assert p2p.shape == torch.Size([5, 5])

    # check distance between 2 clouds
    b = torch.arange(9).float().reshape(3, 3).to(device)
    a_b = batched_point2point_distance(a, b)
    b_a = batched_point2point_distance(b, a)

    assert torch.allclose(a_b.transpose(-1, -2), b_a)

    # check batched distance
    a = torch.randn(10, 20, 3).to(device)
    b = torch.randn(10, 40, 3).to(device)

    a_b = batched_point2point_distance(a, b)
    b_a = batched_point2point_distance(b, a)
    aa = batched_point2point_distance(a)

    assert aa.shape == torch.Size([10, 20, 20])
    assert a_b.shape == torch.Size([10, 20, 40])
    assert b_a.shape == torch.Size([10, 40, 20])
    assert torch.allclose(a_b.transpose(-1, -2), b_a)


@pytest.mark.parametrize('device', devices)
def test_edge_length(device):
    pos = dummy(1, 10, 3).squeeze().to(device)
    p2p = batched_point2point_distance(pos).squeeze().to(device)

    adj = torch.zeros(10, 10).to(device)
    adj[[0, 1, 2], [1, 0, 1]] = 1
    adj[1, 2] = 1

    edg_index = adj.nonzero()
    edg_index = torch.stack([edg_index[:, 0], edg_index[:, 1]])

    expected = (p2p[0, 1]+p2p[1, 0])/2

    assert torch.allclose(expected, total_edge_length(p2p, edg_index))


@pytest.mark.parametrize('device', devices)
def test_chamfer_distance(device):
    pt0 = dummy(1, 10, 3).to(device)
    pt1 = dummy(1, 7, 3).to(device)+1

    p2p = batched_point2point_distance(pt0, pt1)
    l0, idx0, l1, idx1 = batched_chamfer_distance(p2p)

    assert idx0.shape == torch.Size([1, 10])
    assert idx1.shape == torch.Size([1, 7])
    assert l0.item() == 300
    assert l1.item() == 21

    pt0 = dummy(1, 10, 3).expand(2, 10, 3).to(device)
    pt1 = (dummy(1, 7, 3)+1).expand(2, 7, 3).to(device)
    p2p = batched_point2point_distance(pt0, pt1)

    l0, idx0, l1, idx1 = batched_chamfer_distance(p2p)
    assert idx0.shape == torch.Size([2, 10])
    assert idx1.shape == torch.Size([2, 7])
    assert l0.item() == 600
    assert l1.item() == 42


@pytest.mark.parametrize('device', devices)
def test_face_probas(device):
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
                        [0, 3, 2]]).to(device)
    faces = torch.LongTensor([[1, 2, 8],
                              [3, 4, 5],
                              [0, 1, 7],
                              [6, 9, 10],
                              ]).to(device)

    surface_area = surface_areas(pos, faces)
    probas = surface_area/surface_area.sum()

    areas = torch.Tensor([1.22474, 4., 3.5, 8.3666]).to(device)
    expected_probas = areas / areas.sum()

    assert probas.size(0) == faces.shape[0]
    assert torch.allclose(expected_probas, probas)


@pytest.mark.parametrize('device', devices)
def test_sampling(device):
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
                        [0, 3, 2]]).to(device)
    faces = torch.LongTensor([[1, 2, 8],
                              [3, 4, 5],
                              [0, 1, 7],
                              [6, 9, 10],
                              ]).to(device)

    pt = mesh_sampling(pos, faces, num_points=2000)

    assert pt.shape == torch.Size([2000, 3])
