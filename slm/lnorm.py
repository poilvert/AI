#!/usr/bin/env python

"""Local Contrast Divisive Normalization operation (lnorm)
"""

__all__ = ['lnorm']

import numpy as np
from skimage.util.shape import view_as_windows

# -- small value below which the local normalization is not performed
DEFAULT_EPS = 1e-4

def lnorm(
        arr_in,
        neighborhood_shape=(3, 3),
        remove_mean=True,
        normalize=True,
        eps=DEFAULT_EPS
    ):
    """
    Local Contrast Divisive Normalization operation.

    Input
    =====
    arr_in
        4D tensor of shape [ni, h, w, d]

        ni being the number of images
        h being images "height"
        w being images "width"
        d being images "depth"

    Returns
    =======
    arr_out
        4D tensor of shape [ni, h-nh+1, w-nw+1, d]

        nh being the neighborhood height
        nw being the neighborhood width

    Warning
    =======
    all operations are performed in single precision floating point arithmetic.
    """
    assert arr_in.ndim == 4
    assert len(neighborhood_shape) == 2
    assert isinstance(remove_mean, bool)
    assert isinstance(normalize, bool)
    assert eps <= 1.

    narr = np.ascontiguousarray(arr_in.copy(), dtype='f')

    ni, h, w, d = narr.shape
    nh, nw = neighborhood_shape

    assert 1 <= nh <= h
    assert 1 <= nw <= w

    oh, ow = h-nh+1, w-nw+1

    assert oh >= 1
    assert ow >= 1

    arr_out = np.empty((ni, oh, ow, d), dtype='f')

    for i in xrange(ni):
        arr_out[i, ...] = lnorm_3D(
                narr[i, ...],
                neighborhood_shape=neighborhood_shape,
                remove_mean=remove_mean,
                normalize=normalize,
                eps=eps
                )

    return arr_out

def lnorm_3D(
        arr_in,
        neighborhood_shape=(3, 3),
        remove_mean=True,
        normalize=True,
        eps=DEFAULT_EPS
    ):
    """
    Local Contrast Divisive Normalization on a single 3D image

    Warning
    =======
    No checks are made on the input arguments. The code assumes that this
    function has been called by ``lnorm``.
    """

    h, w, d = arr_in.shape
    nh, nw = neighborhood_shape

    oh, ow = h-nh+1, w-nw+1
    nb_size = nh*nw*d

    start_h, stop_h = nh / 2, h - (nh - 1) / 2
    start_w, stop_w = nw / 2, w - (nw - 1) / 2
    arr_out = arr_in[start_h:stop_h, start_w:stop_w, :]

    if (not remove_mean) and (not normalize):
        return arr_out

    arr = view_as_windows(arr_in, (nh, nw, d)).reshape(oh, ow, nb_size)
    arr_sum = arr.sum(-1)
    arr_ssq = (arr ** 2).sum(-1)

    if remove_mean:
        arr_sqn = arr_ssq - (arr_sum ** 2) / nb_size
        arr_out -= arr_sum[..., None] / nb_size
    else:
        arr_sqn = arr_ssq
    np.putmask(arr_sqn, arr_sqn < eps, 1.)

    if normalize:
        arr_out /= np.sqrt(arr_sqn)[..., None]

    return arr_out

try:
    lnorm = profile(lnorm)
except NameError:
    pass
