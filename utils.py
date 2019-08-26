from typing import List
import torch
from torch import Tensor


def conv_output(h: int, w: int, kernel: int = 3, padding: int = 0, dilation: int = 1, stride: int = 1):
    kh, kw = _tuple(kernel)
    ph, pw = _tuple(padding)
    dh, dw = _tuple(dilation)
    sh, sw = _tuple(stride)
    return _dim(h, k=kh, p=ph, s=sh, d=dh), _dim(w, k=kw, p=pw, s=sw, d=dw)


def _dim(h: int, k: int, p: int, s: int, d: int):
    return int((h+2*p-d*(k-1)-1)/s) + 1


def convT_output(h: int, w: int, kernel: int = 3, padding: int = 0, dilation: int = 1, stride: int = 1, output_padding: int = 0):
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


def block_matrices(*matrices) -> Tensor:
    # given multiple matrices of irregular shapes return one squre matrix which contains them all
    ms = torch.LongTensor([m.shape[0] for m in matrices])
    ns = torch.LongTensor([m.shape[1] for m in matrices])

    # we do not transpose or reorder in order to have a smaller matrix
    # maybe later if necessary
    rows = torch.sum(ms)
    columns = ns+torch.cumsum(ms, 0)-ms

    N = torch.max(rows, torch.max(columns))

    device = matrices[0].device
    dtype = matrices[0].dtype

    res = torch.zeros(N, N, device=device, dtype=dtype)

    # TODO vectorize
    for idx, m in enumerate(matrices):
        st_r = torch.sum(ms[:idx])
        c_end = columns[idx]

        res[st_r:st_r+m.shape[0], st_r:c_end] = m

    return res


def unblock_matrices(M: Tensor, shapes: list) -> List[Tensor]:
    sum_rows = 0
    ms = []

    # TODO vectorize
    for shape in shapes:
        m = M[sum_rows:sum_rows+shape[0], sum_rows:sum_rows+shape[1]]
        ms.append(m)
        assert m.dtype == M.dtype
        assert m.device == M.device
        sum_rows += shape[0]

    return ms


if __name__ == "__main__":
    n_features = 3
    m1 = torch.arange(10).reshape(2, 5)
    m2 = torch.arange(9).reshape(3, 3)
    m3 = torch.arange(6).reshape(2, 3)
    m4 = torch.arange(8).reshape(2, 4)

    M = block_matrices(m1, m2, m3, m4).to("cuda:0").to(torch.float32)
    print(M)

    ms = unblock_matrices(M, [(2, 5), (3, 3), (2, 3), (2, 4)])
