#!/usr/bin/env python

"""Local Pooling operation (lpool)
"""

__all__ = ['lpool']

import numpy as np
from skimage.util.shape import view_as_windows

def lpool(
        arr_in,
        neighborhood_shape=(3, 3),
        stride=2,
        order=2,
    ):
    """
    Local Pooling operation
    """
    assert arr_in.ndim == 4
    assert len(neighborhood_shape) == 2
    assert isinstance(stride, int)
    assert stride >= 1
    assert isinstance(order, float)
    assert order >= 1

    narr = np.ascontiguousarray(arr_in.copy(), dtype='f')

    ni, h, w, d = arr_in.shape
    nh, nw = neighborhood_shape

    assert 1 <= nh <= h
    assert 1 <= nw <= w

    oh, ow = h-nh+1, w-nw+1

    assert oh >= 1
    assert ow >= 1

    arr_out = np.empty((ni, oh, ow, d), dtype='f')

    for i in xrange(ni):
        arr_out[i, ...] = lpool_3D(
                narr[i, ...],
                neighborhood_shape=neighborhood_shape,
                order=order,
                )

    return arr_out[:, ::stride, ::stride, :]

def lpool_3D(
        arr_in,
        neighborhood_shape=(3, 3),
        order=2,
    ):

    h, w, d = arr_in.shape
    nh, nw = neighborhood_shape

    oh, ow = h-nh+1, w-nw+1
    inverse_order = float(1. / order)

    arr_out = np.abs(arr_in) ** order
    arr_out = view_as_windows(arr_out, (nh, nw, 1)).reshape(oh, ow, d, nh*nw)
    arr_out = arr_out.sum(-1)
    arr_out = arr_out ** inverse_order

    return arr_out

try:
    lpool = profile(lpool)
except NameError:
    pass
