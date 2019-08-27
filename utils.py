from typing import List, Tuple
import torch
from torch import Tensor


def conv_output(h: int, w: int, kernel: int = 3, padding: int = 0, dilation: int = 1, stride: int = 1):
    '''calculates the feature map height and width given the convolution parameters
    '''
    kh, kw = _tuple(kernel)
    ph, pw = _tuple(padding)
    dh, dw = _tuple(dilation)
    sh, sw = _tuple(stride)
    return _dim(h, k=kh, p=ph, s=sh, d=dh), _dim(w, k=kw, p=pw, s=sw, d=dw)


def _dim(h: int, k: int, p: int, s: int, d: int):
    return int((h+2*p-d*(k-1)-1)/s) + 1


def convT_output(h: int, w: int, kernel: int = 3, padding: int = 0, dilation: int = 1, stride: int = 1, output_padding: int = 0):
    '''calculates the feature map height and width given the transposed convolution parameters
    '''
    kh, kw = _tuple(kernel)
    ph, pw = _tuple(padding)
    dh, dw = _tuple(dilation)
    sh, sw = _tuple(stride)
    pouth, poutw = _tuple(output_padding)
    return _dimT(h, k=kh, p=ph, s=sh, d=dh, pout=pouth), _dimT(w, k=kw, p=pw, s=sw, d=dw, pout=poutw)


def _dimT(h: int, k: int, p: int, s: int, d: int, pout: int):
    return (h-1)*s-2*p+d*(k-1)+pout+1


def _tuple(n):
    if isinstance(n, tuple):
        assert len(n) == 2
        return n
    return n, n


def to_block_diagonal(matrices, sparse=False) -> Tensor:
    ''' given multiple matrices of irregular shapes return one matrix which contains them all on the diagonal\n
        if requested the block matirx will be a sparse instead of dense
    '''
    ms = torch.LongTensor([m.shape[0] for m in matrices])
    ns = torch.LongTensor([m.shape[1] for m in matrices])

    # we do not transpose or reorder in order to have a smaller matrix
    # maybe later if necessary
    M = torch.sum(ms)
    columns = ns+torch.cumsum(ms, 0)-ms
    N = torch.max(columns)

    device = matrices[0].device
    dtype = matrices[0].dtype

    if not sparse:
        # TODO vectorize
        dense = torch.zeros(M, N, device=device, dtype=dtype)
        for idx, m in enumerate(matrices):
            st_r = torch.sum(ms[:idx])
            c_end = columns[idx]
            dense[st_r:st_r+m.shape[0], st_r:c_end] = m
        return dense
    else:
        i_coords = []
        j_coords = []
        data = []
        # occupied indices in the block matrix
        for idx, m in enumerate(matrices):
            st_r = torch.sum(ms[:idx])
            c_end = columns[idx]
            data.append(m.flatten())
            for i in range(st_r, st_r+m.shape[0]):
                for j in range(st_r, c_end):
                    i_coords.append(i)
                    j_coords.append(j)

        data = torch.cat(data)
        return torch.sparse.FloatTensor(torch.LongTensor([i_coords, j_coords]), data, torch.Size([M, N]))


def from_block_diagonal(M: Tensor, shapes) -> List[Tensor]:
    ''' given a block diagonal matrix sparse or dense extracts the matrices denoted by the given shapes\n

        for eg. given m1,m2 with shapes s1 and s2\n
                M=to_block_diagonal(m1,m2)
                a,b=from_block_diagonal(M,[s1,s2])
                a and b will have the same values as m1,m2
    '''
    sum_rows = 0
    ms = []

    if not isinstance(shapes, list):
        shapes = [shapes]

    if M.is_sparse:
        M = M.to_dense()

    # TODO vectorize
    for shape in shapes:
        m = M[sum_rows:sum_rows+shape[0], sum_rows:sum_rows+shape[1]]
        ms.append(m)
        assert m.dtype == M.dtype
        assert m.device == M.device
        sum_rows += shape[0]

    return ms


if __name__ == "__main__":
    n_features = 10
    m1 = torch.arange(25).reshape(5, 5)
    m2 = torch.arange(6*6).reshape(6, 6)
    m3 = torch.arange(5*n_features).reshape(5, n_features)
    m4 = torch.arange(6*n_features).reshape(6, n_features)
    m5 = torch.arange(n_features*n_features).reshape(n_features, n_features)

    # M = to_block_diagonal([m1, m2], sparse=False)
    # N = to_block_diagonal([m3, m4], sparse=False)
    # sM = to_block_diagonal([m1, m2], sparse=True)
    # sN = to_block_diagonal([m3, m4], sparse=True)
    # print(M.shape, N.shape)
    # print(sM.shape, sN.shape)

    stacked_ms = torch.arange(3*5*5).reshape(3, 5, 5)

    # TODO this is not ok we should not have the zero column in the middle
    stacked = to_block_diagonal(
        [torch.ones(3, 2).float(), torch.ones(3, 2).float()])

    print(stacked)
