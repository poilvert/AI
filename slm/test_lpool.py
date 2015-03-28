#!/usr/bin/env python

import numpy as np
from numpy.testing import assert_allclose
from lpool import lpool

RTOL=1e-5
ATOL=1e-5


def test_constant_images_L1():
    arr = np.ones((10, 64, 64, 1), dtype='f')
    for i in range(10):
        arr[i, ...] *= i * (-1)**i
    out = lpool(arr, neighborhood_shape=(2, 2), order=1.)
    ref = np.ones((10, 63, 63, 1), dtype='f')[:, ::2, ::2, :]
    for i in range(10):
        ref[i, ...] *= 4*i
    assert_allclose(out, ref, rtol=RTOL, atol=ATOL)


def test_constant_images_L2():
    arr = np.ones((10, 64, 64, 1), dtype='f')
    for i in range(10):
        arr[i, ...] *= i * (-1)**i
    out = lpool(arr, neighborhood_shape=(2, 2), order=2.)
    ref = np.ones((10, 63, 63, 1), dtype='f')[:, ::2, ::2, :]
    for i in range(10):
        ref[i, ...] *= 2*i
    assert_allclose(out, ref, rtol=RTOL, atol=ATOL)


def test_constant_images_L10():
    arr = np.ones((10, 64, 64, 1), dtype='f')
    for i in range(10):
        arr[i, ...] *= i * (-1)**i
    out = lpool(arr, neighborhood_shape=(2, 2), order=10.)
    ref = np.ones((10, 63, 63, 1), dtype='f')[:, ::2, ::2, :]
    for i in range(10):
        ref[i, ...] *= float(4**(1./10.))*i
    assert_allclose(out, ref, rtol=RTOL, atol=ATOL)
