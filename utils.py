def conv_output(h, w, kernel=3, padding=0, dilation=1, stride=1):
    kh, kw = _tuple(kernel)
    ph, pw = _tuple(padding)
    dh, dw = _tuple(dilation)
    sh, sw = _tuple(stride)
    return _dim(h, k=kh, p=ph, s=sh, d=dh), _dim(w, k=kw, p=pw, s=sw, d=dw)


def _dim(h, k, p, s, d):
    return int((h+2*p-d*(k-1)-1)/s) + 1


def convT_output(h, w, kernel=3, padding=0, dilation=1, stride=1, output_padding=0):
    kh, kw = _tuple(kernel)
    ph, pw = _tuple(padding)
    dh, dw = _tuple(dilation)
    sh, sw = _tuple(stride)
    pouth, poutw = _tuple(output_padding)
    return _dimT(h, k=kh, p=ph, s=sh, d=dh, pout=pouth), _dimT(w, k=kw, p=pw, s=sw, d=dw, pout=poutw)


def _dimT(h, k, p, s, d, pout):
    return (h-1)*s-2*p+d*(k-1)+pout+1


def _tuple(n):
    if isinstance(n, tuple):
        assert len(n) == 2
        return n
    return n, n
