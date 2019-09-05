
import torch
import torch.nn as nn
from layers import Cubify, FCN, GraphConv, ResGraphConv, VoxelBranch,\
    VertexAlign, ResVertixRefineShapenet, VertixRefineShapeNet, VertixRefinePix3D

from utils import aggregate_neighbours


def test_aggregate():
    a = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    edge_index = torch.LongTensor([[0, 0, 1, 2],
                                   [1, 2, 1, 0]])

    out = aggregate_neighbours(edge_index, a)
    expected = torch.Tensor([[11., 13., 15.],
                             [4.,  5.,  6.],
                             [1.,  2.,  3.]])

    assert torch.allclose(expected, out)


def tesst_cubify():
    cube = Cubify(0.5).to('cuda')

    inp = torch.randn(1, 48, 48, 48).to('cuda:0')
    _ = cube(inp)


def test_align():
    feature_extractor = FCN(3)
    # check multiple graphs with multiple feature maps sizes
    img = torch.randn(2, 3, 137, 137)
    f_maps = feature_extractor(img)
    align = VertexAlign((137, 137))
    pos = torch.randint(0, 137, (100, 3)).float()
    vert_per_m = [49, 51]
    c = align(f_maps, pos, vert_per_m)
    assert c.shape == torch.Size([100, 3840])

    # check multiple graphs with one feature_map size
    f_map = torch.randn(2, 256, 224, 224)
    align = VertexAlign((224, 224))
    c = align([f_map], pos, vert_per_m)
    assert c.shape == torch.Size([100, 256])


def test_graphConv():
    conv = GraphConv(3, 6)
    conv.w0 = nn.Parameter(torch.ones(*conv.w0.shape))
    conv.w1 = nn.Parameter(torch.ones(*conv.w1.shape))

    in_f = torch.arange(9).reshape(3, 3).float()

    adj = torch.Tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    edge_index = adj.nonzero()
    edge_index = torch.stack([edge_index[:, 0], edge_index[:, 1]])
    assert edge_index.shape == torch.Size([2, 4])
    out_f = conv(in_f, edge_index)

    assert out_f.shape == torch.Size([3, 6])
    assert torch.allclose(out_f, torch.Tensor(
        [15, 36, 33]).view(3, 1).expand(3, 6))


def test_resGraphConv():
    # without projection
    conv = ResGraphConv(3, 3)

    in_f = torch.arange(9).reshape(3, 3).float()

    adj = torch.Tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    edge_index = adj.nonzero()
    edge_index = torch.stack([edge_index[:, 0], edge_index[:, 1]])
    assert edge_index.shape == torch.Size([2, 4])
    out_f = conv(in_f, edge_index)

    assert out_f.shape == torch.Size([3, 3])

    # with projection
    conv = ResGraphConv(3, 10)

    in_f = torch.arange(9).reshape(3, 3).float()

    adj = torch.Tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    edge_index = adj.nonzero()
    edge_index = torch.stack([edge_index[:, 0], edge_index[:, 1]])
    assert edge_index.shape == torch.Size([2, 4])
    out_f = conv(in_f, edge_index)

    assert out_f.shape == torch.Size([3, 10])


def test_voxelBranch():
    branch = VoxelBranch(10, 22)
    inp = torch.randn(2, 10, 64, 64)

    out = branch(inp)

    assert out.shape == torch.Size([2, 22, 128, 128])


def test_FCN():
    filters = 32
    fcn = FCN(3, filters=filters)

    H = 64
    x = torch.randn(2, 3, H, H)

    outs = fcn(x)

    assert len(outs) == 4

    for i, out in enumerate(outs):
        mul = 2**(i+2)
        b, c, h, w = 2, mul*filters, H//mul, H//mul

        assert out.shape == torch.Size([b, c, h, w])


def test_resVertixRefineShapenet():
    refine0 = ResVertixRefineShapenet(
        (224, 224), alignment_size=256, use_input_features=False)

    vertices_per_sample = [49, 51]
    vertex_adjacency = torch.zeros(100, 100)
    img_feature_maps = torch.randn(2, 256, 224, 224)

    # circle adjacency
    for i in range(49):
        vertex_adjacency[i, (i+1) % 49] = 1
        vertex_adjacency[(i-1) % 49, i] = 1
    for i in range(49, 100):
        vertex_adjacency[i, 49+(i+1) % 49] = 1
        vertex_adjacency[49 + (i+1) % 49, i] = 1

    edge_index = vertex_adjacency.nonzero()
    edge_index = torch.stack([edge_index[:, 0], edge_index[:, 1]])

    vertex_positions = torch.randn(100, 3)

    new_pos, new_featues = refine0(vertices_per_sample, [img_feature_maps], edge_index,
                                   vertex_positions, vertex_features=None)

    assert new_pos.shape == torch.Size([100, 3])
    assert new_featues.shape == torch.Size([100, 128])

    refine1 = ResVertixRefineShapenet(
        (224, 224), alignment_size=256, use_input_features=True)

    new_pos, new_new_features = refine1(vertices_per_sample, [img_feature_maps], edge_index,
                                        vertex_positions, vertex_features=new_featues)
    assert new_pos.shape == torch.Size([100, 3])
    assert new_new_features.shape == torch.Size([100, 128])


def test_vertixRefineShapenet():
    refine0 = VertixRefineShapeNet(
        (224, 224), alignment_size=256, use_input_features=False)

    vertices_per_sample = [49, 51]
    vertex_adjacency = torch.zeros(100, 100)
    img_feature_maps = torch.randn(2, 256, 224, 224)

    # circle adjacency
    for i in range(49):
        vertex_adjacency[i, (i+1) % 49] = 1
        vertex_adjacency[(i-1) % 49, i] = 1
    for i in range(49, 100):
        vertex_adjacency[i, 49+(i+1) % 49] = 1
        vertex_adjacency[49 + (i+1) % 49, i] = 1

    edge_index = vertex_adjacency.nonzero()
    edge_index = torch.stack([edge_index[:, 0], edge_index[:, 1]])

    vertex_positions = torch.randn(100, 3)

    new_pos, new_featues = refine0(vertices_per_sample, [img_feature_maps], edge_index,
                                   vertex_positions, vertex_features=None)

    assert new_pos.shape == torch.Size([100, 3])
    assert new_featues.shape == torch.Size([100, 128])

    refine1 = VertixRefineShapeNet(
        (224, 224), alignment_size=256, use_input_features=True)

    new_pos, new_new_features = refine1(vertices_per_sample, [img_feature_maps], edge_index,
                                        vertex_positions, vertex_features=new_featues)
    assert new_pos.shape == torch.Size([100, 3])
    assert new_new_features.shape == torch.Size([100, 128])


def test_vertixRefinePix3D():
    refine0 = VertixRefinePix3D(
        (224, 224), alignment_size=256, use_input_features=False)

    vertices_per_sample = [49, 51]
    vertex_adjacency = torch.zeros(100, 100)
    img_feature_maps = torch.randn(2, 256, 224, 224)

    # circle adjacency
    for i in range(49):
        vertex_adjacency[i, (i+1) % 49] = 1
        vertex_adjacency[(i-1) % 49, i] = 1
    for i in range(49, 100):
        vertex_adjacency[i, 49+(i+1) % 49] = 1
        vertex_adjacency[49 + (i+1) % 49, i] = 1

    edge_index = vertex_adjacency.nonzero()
    edge_index = torch.stack([edge_index[:, 0], edge_index[:, 1]])

    vertex_positions = torch.randn(100, 3)

    new_pos, new_featues = refine0(vertices_per_sample, img_feature_maps, edge_index,
                                   vertex_positions, vertex_features=None)

    assert new_pos.shape == torch.Size([100, 3])
    assert new_featues.shape == torch.Size([100, 128])

    refine1 = VertixRefinePix3D(
        (224, 224), alignment_size=256, use_input_features=True)

    new_pos, new_new_features = refine1(vertices_per_sample, img_feature_maps, edge_index,
                                        vertex_positions, vertex_features=new_featues)
    assert new_pos.shape == torch.Size([100, 3])
    assert new_new_features.shape == torch.Size([100, 128])
