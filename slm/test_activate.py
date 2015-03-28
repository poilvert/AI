#!/usr/bin/env python

import numpy as np
from numpy.testing import assert_allclose
from activate import activate, activateabs

RTOL=1e-5
ATOL=1e-5


def test_activate_upper_bounded():
    arr = np.random.randn(10, 100, 100, 32).astype('f')
    arr -= arr.max()
    out = activate(arr, min_val=-np.inf, max_val=1.0)
    assert_allclose(out, arr, rtol=RTOL, atol=ATOL)


def test_activate_lower_bounded():
    arr = np.random.randn(10, 100, 100, 32).astype('f')
    arr -= arr.min()
    out = activate(arr, min_val=0., max_val=np.inf)
    assert_allclose(out, arr, rtol=RTOL, atol=ATOL)


def test_activateabs_on_positive_array():
    arr = np.random.randn(10, 100, 100, 32).astype('f')
    arr -= arr.min()
    arr /= arr.max()
    out = activateabs(arr, max_val=1.)
    assert_allclose(out, arr, rtol=RTOL, atol=ATOL)


def test_activateabs_when_saturating():
    arr = np.random.randn(10, 100, 100, 32).astype('f')
    arr -= arr.min()
    arr += 1.
    out = activateabs(arr, max_val=1.)
    ref = np.ones(out.shape, dtype=out.dtype)
    assert_allclose(out, ref, rtol=RTOL, atol=ATOL)
