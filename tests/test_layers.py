import pytest
import torch
import torch.nn as nn
from model.layers import Cubify, GraphConv, ResGraphConv, VoxelBranch,\
    VertexAlign, ResVertixRefineShapenet, VertixRefineShapeNet, VertixRefinePix3D

from model.models import pretrained_ResNet50
from model.utils import aggregate_neighbours

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')


@pytest.mark.parametrize('device', devices)
def test_aggregate(device):
    a = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).to(device)
    edge_index = torch.LongTensor([[0, 0, 1, 2],
                                   [1, 2, 1, 0]]).to(device)

    out = aggregate_neighbours(edge_index, a)
    expected = torch.Tensor([[11., 13., 15.],
                             [4., 5., 6.],
                             [1., 2., 3.]]).to(device)

    assert torch.allclose(expected, out)


@pytest.mark.parametrize('device', devices)
def tesst_cubify(device):
    cube = Cubify(0.5).to(device)

    inp = torch.randn(1, 48, 48, 48).to(device)
    _ = cube(inp)


@pytest.mark.parametrize('device', devices)
def test_align(device):
    backbone = pretrained_ResNet50(
        loss_function=None, pretrained=False).to(device).eval()
    # check multiple graphs with multiple feature maps sizes
    img = torch.randn(2, 3, 137, 137).to(device)
    _, f_maps = backbone(img)
    align = VertexAlign()
    pos = torch.randint(0, 137, (100, 3)).float().to(device)
    vert_per_m = [49, 51]
    c = align(f_maps, pos, vert_per_m, [i.shape[1:] for i in img], [1, 1])
    assert c.shape == torch.Size([100, 3840])

    # check multiple graphs with one feature_map size
    f_map = torch.randn(2, 256, 224, 224).to(device)
    align = VertexAlign()
    c = align([f_map], pos, vert_per_m, [(224, 224), (224, 224)], [1, 1])
    assert c.shape == torch.Size([100, 256])


@pytest.mark.parametrize('device', devices)
def test_graphConv(device):
    conv = GraphConv(3, 6).to(device)
    conv.w0 = nn.Parameter(torch.ones(*conv.w0.shape).to(device))
    conv.w1 = nn.Parameter(torch.ones(*conv.w1.shape).to(device))

    in_f = torch.arange(9).reshape(3, 3).float().to(device)

    adj = torch.Tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).to(device)

    edge_index = adj.nonzero()
    edge_index = torch.stack([edge_index[:, 0], edge_index[:, 1]])
    assert edge_index.shape == torch.Size([2, 4])
    out_f = conv(in_f, edge_index)

    assert out_f.shape == torch.Size([3, 6])
    assert torch.allclose(out_f, torch.Tensor(
        [15, 36, 33]).view(3, 1).expand(3, 6).to(device))


@pytest.mark.parametrize('device', devices)
def test_resGraphConv(device):
    # without projection
    conv = ResGraphConv(3, 3).to(device)

    in_f = torch.arange(9).reshape(3, 3).float().to(device)

    adj = torch.Tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).to(device)

    edge_index = adj.nonzero()
    edge_index = torch.stack([edge_index[:, 0], edge_index[:, 1]])
    assert edge_index.shape == torch.Size([2, 4])
    out_f = conv(in_f, edge_index)

    assert out_f.shape == torch.Size([3, 3])

    # with projection
    conv = ResGraphConv(3, 10).to(device)

    in_f = torch.arange(9).reshape(3, 3).float().to(device)

    adj = torch.Tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).to(device)
    edge_index = adj.nonzero()
    edge_index = torch.stack([edge_index[:, 0], edge_index[:, 1]])
    assert edge_index.shape == torch.Size([2, 4])
    out_f = conv(in_f, edge_index)

    assert out_f.shape == torch.Size([3, 10])


@pytest.mark.parametrize('device', devices)
def test_voxelBranch(device):
    branch = VoxelBranch(10, 22).to(device)
    inp = torch.randn(2, 10, 64, 64).to(device)

    out = branch(inp)

    assert out.shape == torch.Size([2, 22, 128, 128])


@pytest.mark.parametrize('device', devices)
def test_ShapeNetFeatureExtractor(device):
    filters = 64
    backbone = pretrained_ResNet50(
        loss_function=None, pretrained=False).to(device).eval()

    H = 64
    x = torch.randn(2, 3, H, H).to(device)

    _, outs = backbone(x)

    assert len(outs) == 4

    for i, out in enumerate(outs):
        mul = 2**(i + 2)
        b, c, h, w = 2, mul * filters, H // mul, H // mul

        assert out.shape == torch.Size([b, c, h, w])


@pytest.mark.parametrize('device', devices)
def test_resVertixRefineShapenet(device):
    refine0 = ResVertixRefineShapenet(alignment_size=256,
                                      use_input_features=False).to(device)

    vertices_per_sample = [49, 51]
    vertex_adjacency = torch.zeros(100, 100).to(device)
    img_feature_maps = torch.randn(2, 256, 224, 224).to(device)

    # circle adjacency
    for i in range(49):
        vertex_adjacency[i, (i + 1) % 49] = 1
        vertex_adjacency[(i - 1) % 49, i] = 1
    for i in range(49, 100):
        vertex_adjacency[i, 49 + (i + 1) % 49] = 1
        vertex_adjacency[49 + (i + 1) % 49, i] = 1

    edge_index = vertex_adjacency.nonzero()
    edge_index = torch.stack([edge_index[:, 0], edge_index[:, 1]])

    vertex_positions = torch.randn(100, 3).to(device)
    sizes = [(132, 132), (132, 164)]
    new_pos, new_featues = refine0(vertices_per_sample, [img_feature_maps], edge_index,
                                   vertex_positions, sizes, vertex_features=None)

    assert new_pos.shape == torch.Size([100, 3])
    assert new_featues.shape == torch.Size([100, 128])

    refine1 = ResVertixRefineShapenet(alignment_size=256,
                                      use_input_features=True).to(device)

    new_pos, new_new_features = refine1(vertices_per_sample, [img_feature_maps], edge_index,
                                        vertex_positions, sizes, vertex_features=new_featues)
    assert new_pos.shape == torch.Size([100, 3])
    assert new_new_features.shape == torch.Size([100, 128])


@pytest.mark.parametrize('device', devices)
def test_vertixRefineShapenet(device):
    refine0 = VertixRefineShapeNet(alignment_size=256,
                                   use_input_features=False).to(device)

    vertices_per_sample = [49, 51]
    vertex_adjacency = torch.zeros(100, 100).to(device)
    img_feature_maps = torch.randn(2, 256, 224, 224).to(device)

    # circle adjacency
    for i in range(49):
        vertex_adjacency[i, (i + 1) % 49] = 1
        vertex_adjacency[(i - 1) % 49, i] = 1
    for i in range(49, 100):
        vertex_adjacency[i, 49 + (i + 1) % 49] = 1
        vertex_adjacency[49 + (i + 1) % 49, i] = 1

    edge_index = vertex_adjacency.nonzero()
    edge_index = torch.stack([edge_index[:, 0], edge_index[:, 1]])

    vertex_positions = torch.randn(100, 3).to(device)
    sizes = [(224, 136), (134, 122)]
    new_pos, new_featues = refine0(vertices_per_sample, [img_feature_maps], edge_index,
                                   vertex_positions, sizes, vertex_features=None)

    assert new_pos.shape == torch.Size([100, 3])
    assert new_featues.shape == torch.Size([100, 128])

    refine1 = VertixRefineShapeNet(
        alignment_size=256, use_input_features=True).to(device)

    new_pos, new_new_features = refine1(vertices_per_sample, [img_feature_maps], edge_index,
                                        vertex_positions, sizes, vertex_features=new_featues)
    assert new_pos.shape == torch.Size([100, 3])
    assert new_new_features.shape == torch.Size([100, 128])


@pytest.mark.parametrize('device', devices)
def test_vertixRefinePix3D(device):
    refine0 = VertixRefinePix3D(alignment_size=256,
                                use_input_features=False).to(device)

    vertices_per_sample = [49, 51]
    vertex_adjacency = torch.zeros(100, 100).to(device)
    img_feature_maps = torch.randn(2, 256, 224, 224).to(device)
    sizes = [(224, 136), (134, 122)]
    # circle adjacency
    for i in range(49):
        vertex_adjacency[i, (i + 1) % 49] = 1
        vertex_adjacency[(i - 1) % 49, i] = 1
    for i in range(49, 100):
        vertex_adjacency[i, 49 + (i + 1) % 49] = 1
        vertex_adjacency[49 + (i + 1) % 49, i] = 1

    edge_index = vertex_adjacency.nonzero()
    edge_index = torch.stack([edge_index[:, 0], edge_index[:, 1]])

    vertex_positions = torch.randn(100, 3).to(device)

    new_pos, new_featues = refine0(vertices_per_sample, img_feature_maps, edge_index,
                                   vertex_positions, sizes, vertex_features=None)

    assert new_pos.shape == torch.Size([100, 3])
    assert new_featues.shape == torch.Size([100, 128])

    refine1 = VertixRefinePix3D(alignment_size=256,
                                use_input_features=True).to(device)

    new_pos, new_new_features = refine1(vertices_per_sample, img_feature_maps, edge_index,
                                        vertex_positions, sizes, vertex_features=new_featues)
    assert new_pos.shape == torch.Size([100, 3])
    assert new_new_features.shape == torch.Size([100, 128])
