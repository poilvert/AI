#!/usr/bin/env python

# Authors: Nicolas Poilvert <nicolas.poilvert@gmail.com>
#          Nicolas Pinto <nicolas.pinto@gmail.com>
# License: BSD

__all__ = ['SlmModel', 'MallatSlmModel']

import numpy as np
from model_utilities import process_one_layer
from model_utilities import recursive_process
from model_utilities import join_features

BASIC_OPS = ['lnorm', 'fbcorr', 'activate', 'lpool']

DEFAULT_QUANTIZATION = False
DEFAULT_QUANTIZATION_N_BITS = 8

# -- SLM models
class SlmModel(object):

    def __init__(self,
            layers,
            quantization=DEFAULT_QUANTIZATION,
            quantization_n_bits=DEFAULT_QUANTIZATION_N_BITS
            ):
        assert len(layers) > 0
        for layer in layers:
            for op in layer:
                assert op["type"] in BASIC_OPS
        self.desc = layers
        self.quantization = quantization
        self.quantization_n_bits = quantization_n_bits

    def forward(self, X):
        assert X.ndim == 4
        Y = np.ascontiguousarray(X.copy(), dtype=np.float32)
        layers = self.desc
        for layer in layers:
            Y = process_one_layer(Y, layer,
                    quantization=self.quantization,
                    quantization_n_bits=self.quantization_n_bits
                    )
        return Y

# -- Mallat-like SLM models
class MallatSlmModel(object):

    def __init__(self,
            layers,
            quantization=DEFAULT_QUANTIZATION,
            quantization_n_bits=DEFAULT_QUANTIZATION_N_BITS
            ):
        assert len(layers) > 0
        for layer in layers:
            for op in layer:
                assert op["type"] in BASIC_OPS
        self.desc = layers
        self.quantization = quantization
        self.quantization_n_bits = quantization_n_bits

    def forward(self, X):
        assert X.ndim == 4
        Y = np.ascontiguousarray(X.copy(), dtype=np.float32)
        layers = self.desc
        Y_l = recursive_process(Y, layers,
                quantization=self.quantization,
                quantization_n_bits=self.quantization_n_bits
                )
        Y = join_features(Y_l)
        return Y

if __name__ == "__main__":

    from scipy.misc import lena
    from time import time

    n_repeats = 10
    model = [
        [
            {"type":"lnorm",
             "neighborhood_shape":(3,3),
             "remove_mean":True,
             "normalize":True},
            {"type":"fbcorr", "filter_bank":np.random.randn(5, 5, 1, 16)},
            {"type":"activate", "min_val":0., "max_val":1.},
            {"type":"lpool",
             "neighborhood_shape":(2,2),
             "order":2.,
             "stride":2}
        ],
        [
            {"type":"lnorm",
             "neighborhood_shape":(5,5),
             "remove_mean":True,
             "normalize":True},
            {"type":"fbcorr", "filter_bank":np.random.randn(3, 3, 16, 32)},
            {"type":"activate", "min_val":0., "max_val":1.},
            {"type":"lpool",
             "neighborhood_shape":(2,2),
             "order":2.,
             "stride":2}
        ],
    ]

    quantization = True
    quantization_n_bits = 8
    stride = 1

    arr_in = lena()[None, ::stride, ::stride, None]
    arr_in -= arr_in.mean()
    arr_in /= (arr_in.max() - arr_in.min())

    Mallatmodel = MallatSlmModel(
            model,
            quantization=quantization,
            quantization_n_bits=quantization_n_bits
            )
    Slmmodel = SlmModel(
            model,
            quantization=quantization,
            quantization_n_bits=quantization_n_bits
            )

    start = time()
    for _ in xrange(n_repeats):
        sarr_out = Slmmodel.forward(arr_in)
    stop = time()

    duration = 1000. * (stop - start) / n_repeats
    print 'SLM'
    print 'time to compute (ms): %6.2f' % duration
    print 'shape: %s' % str(sarr_out.shape)

    start = time()
    for _ in xrange(n_repeats):
        marr_out = Mallatmodel.forward(arr_in)
    stop = time()

    duration = 1000. * (stop - start) / n_repeats
    print 'Mallat'
    print 'time to compute (ms): %6.2f' % duration
    print 'shape: %s' % str(marr_out.shape)
