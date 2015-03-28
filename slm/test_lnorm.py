#!/usr/bin/env python

import numpy as np
from numpy.testing import assert_allclose
from lnorm import lnorm

RTOL=1e-5
ATOL=1e-5


def test_constant_input_no_mean_remove_no_normalize():
    val = 2.
    shape = (10, 20, 20, 3)
    arr = val * np.ones(shape, dtype='f')
    ref = val * np.ones((10, 18, 18, 3), dtype='f')
    out = lnorm(arr, remove_mean=False, normalize=False)
    assert_allclose(out, ref, rtol=RTOL, atol=ATOL)


def test_constant_input_mean_remove_no_normalize():
    val = 2.
    shape = (10, 20, 20, 3)
    arr = val * np.ones(shape, dtype='f')
    ref = 0. * np.ones((10, 18, 18, 3), dtype='f')
    out = lnorm(arr, remove_mean=True, normalize=False)
    assert_allclose(out, ref, rtol=RTOL, atol=ATOL)


def test_constant_input_mean_remove_normalize():
    val = 2.
    shape = (10, 20, 20, 3)
    arr = val * np.ones(shape, dtype='f')
    ref = 0. * np.ones((10, 18, 18, 3), dtype='f')
    out = lnorm(arr, remove_mean=True, normalize=True)
    assert_allclose(out, ref, rtol=RTOL, atol=ATOL)


def test_single_2D_image_gradient_in_x_remove_mean():
    shape = (1, 20, 20, 1)
    arr = np.arange(20*20).reshape(shape)
    ref = 10.5 * np.ones((1, 19, 19, 1), dtype='f')
    out = lnorm(arr, neighborhood_shape=(2, 2),
            remove_mean=True, normalize=False)
    assert_allclose(out, ref, rtol=RTOL, atol=ATOL)


def test_single_2D_image_gradient_in_y_remove_mean():
    shape = (1, 20, 20, 1)
    arr = np.arange(20*20).reshape(20, 20).T.reshape(shape)
    ref = 10.5 * np.ones((1, 19, 19, 1), dtype='f')
    out = lnorm(arr, neighborhood_shape=(2, 2),
            remove_mean=True, normalize=False)
    assert_allclose(out, ref, rtol=RTOL, atol=ATOL)
