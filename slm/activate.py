#!/usr/bin/env python

"""Activation operation (activate)
"""

__all__ = ['activate', 'activateabs']

import numpy as np

def activate(arr_in, min_val=0., max_val=1.):
    """
    Activation function. This function emulates the ``saturation`` of neurons
    """
    assert arr_in.ndim == 4

    arr_out = np.ascontiguousarray(arr_in.copy(), dtype='f')
    np.putmask(arr_out, arr_out <= min_val, min_val)
    np.putmask(arr_out, arr_out >= max_val, max_val)

    return arr_out

def activateabs(arr_in, max_val=1.):
    """
    Activation in the form of a point-wise modulus operation and a clipping to a
    certain maximum value
    """
    assert arr_in.ndim == 4

    arr_out = np.ascontiguousarray(arr_in.copy(), dtype='f')
    arr_out = np.abs(arr_out)
    np.putmask(arr_out, arr_out >= max_val, max_val)

    return arr_out

try:
    activate = profile(activate)
except NameError:
    pass
