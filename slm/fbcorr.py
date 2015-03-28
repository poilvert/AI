#!/usr/bin/env python

"""Filter Bank Correlation operation (fbcorr)
"""

__all__ = ['fbcorr']

import numpy as np
from skimage.util.shape import view_as_windows
from mydot import dot

def fbcorr(
        arr_in,
        arr_fb,
    ):
    """
    Filter Bank Correlation operation
    """
    assert arr_in.ndim == 4
    assert arr_fb.ndim == 4

    narr = np.ascontiguousarray(arr_in.copy(), dtype='f')
    narr_fb = np.ascontiguousarray(arr_fb.copy(), dtype='f')

    ni, h, w, d = arr_in.shape
    fh, fw, fd, nf = arr_fb.shape

    assert 1 <= fh <= h
    assert 1 <= fw <= w
    assert d == fd

    oh, ow = h-fh+1, w-fw+1

    assert oh >= 1
    assert ow >= 1

    arr_out = np.empty((ni, oh, ow, nf), dtype='f')

    for i in xrange(ni):
        arr_out[i, ...] = fbcorr_3D(
                narr[i, ...],
                narr_fb
                )

    return arr_out

def fbcorr_3D(arr_in, arr_fb):
    """
    Warning
    =======
    No checks on inputs performed
    """

    h, w, d = arr_in.shape
    fh, fw, fd, nf = arr_fb.shape

    assert fd == d

    oh, ow = h-fh+1, w-fw+1

    arr_inr = view_as_windows(arr_in, (fh, fw, fd))
    arr_inm = np.ascontiguousarray(arr_inr.reshape(oh*ow, fh*fw*fd))

    arr_fbm = np.ascontiguousarray(arr_fb.reshape(fh*fw*fd, nf))

    arr_outm = dot(arr_inm, arr_fbm)
    arr_out = arr_outm.reshape(oh, ow, nf)

    return arr_out

try:
    fbcorr = profile(fbcorr)
except NameError:
    pass
