#!/usr/bin/env python

from lnorm import lnorm
from lpool import lpool
from fbcorr import fbcorr
from activate import activate
from resample import resample
import numpy as np
import numexpr as ne

DEFAULT_EPS = 1e-3

# -- Quantizing an array in N bits between its min and max
def quantize_array(arr, n_bits=8, eps=1e-4):
    arr_out = arr.copy().astype('f')
    arr_min, arr_max = arr_out.min(), arr_out.max()
    # -- the "if" statement here basically kills some noise
    if np.abs(arr_max - arr_min) < eps:
        arr_out[:] = float((arr_min + arr_max) / 2)
    else:
        scale = float(2. ** (n_bits) / (arr_max - arr_min))
        arr_out = ne.evaluate("scale * (arr_out - arr_min)")
        arr_out = np.around(arr_out)
        inverse_scale = float((arr_max - arr_min) / 2. ** (n_bits))
        arr_out = ne.evaluate("inverse_scale * arr_out + arr_min")
    return arr_out

# -- resampling utility for 4D tensors (resampling only along the last 3
# dimensions)
def resample4D(arr_in, out_shape):
    assert arr_in.ndim == 4
    narr = np.ascontiguousarray(arr_in.copy(), dtype=np.float32)
    ni, h, w, d = narr.shape
    nh, nw, nd = out_shape
    arr_out = np.empty((ni, nh, nw, nd), dtype=narr.dtype)
    for i in xrange(ni):
        arr_out[i, ...] = resample(narr[i, ...], out_shape)
    return arr_out

# -- computing the feature map for a list of basic SLM operations
def process_one_layer(
        X, layer,
        quantization=False,
        quantization_n_bits=8
        ):

    Y = np.ascontiguousarray(X, dtype=np.float32)

    for op in layer:

        if quantization:
            Y = quantize_array(Y, n_bits=quantization_n_bits)

        op_type = op["type"]

        if op_type == "lnorm":
            nb_shape = op["neighborhood_shape"]
            assert len(nb_shape) == 2
            remove_mean = op["remove_mean"]
            assert isinstance(remove_mean, bool)
            normalize = op["normalize"]
            assert isinstance(normalize, bool)
            Y = lnorm(Y,
                      neighborhood_shape=nb_shape,
                      remove_mean=remove_mean,
                      normalize=normalize,
                      eps=DEFAULT_EPS
                     )

        elif op_type == "lpool":
            nb_shape = op["neighborhood_shape"]
            assert len(nb_shape) == 2
            order = op["order"]
            assert isinstance(order, float)
            stride = op["stride"]
            assert isinstance(stride, int)
            Y = lpool(Y,
                      neighborhood_shape=nb_shape,
                      stride=stride,
                      order=order
                     )

        elif op_type == "fbcorr":
            fbank = op["filter_bank"]
            if quantization:
                fbank = quantize_array(fbank, n_bits=quantization_n_bits)
            assert fbank.ndim == 4
            Y = fbcorr(Y,
                       fbank
                      )

        elif op_type == "activate":
            min_val = op["min_val"]
            assert isinstance(min_val, float)
            max_val = op["max_val"]
            assert isinstance(max_val, float)
            Y = activate(Y,
                         min_val=min_val,
                         max_val=max_val
                        )

    return Y

# -- recursive extraction of features in a Mallat-like tree
def recursive_process(
        X, layers,
        quantization=False,
        quantization_n_bits=8
        ):
    assert len(layers) > 0
    if len(layers) == 1:
        Y_right = process_one_layer(
                X, layers[0],
                quantization=quantization,
                quantization_n_bits=quantization_n_bits,
                )
        h, w = Y_right.shape[1:3]
        d = X.shape[-1]
        Y_left = resample4D(X, (h, w, d))
        return [Y_left, Y_right]
    else:
        Y_right = process_one_layer(
                X, layers[0],
                quantization=quantization,
                quantization_n_bits=quantization_n_bits,
                )
        h, w = Y_right.shape[1:3]
        d = X.shape[-1]
        Y_left = resample4D(X, (h, w, d))
        return recursive_process(Y_left, layers[:-1]) + \
               recursive_process(Y_right, layers[1:])

# -- upsampling of feature maps obtained from a recursive extraction to a common
# tensor shape
def join_features(my_list):
    assert len(my_list) >= 1
    shapes = np.array([tensor.shape for tensor in my_list])
    max_h, max_w = shapes.max(axis=0)[1:3]
    new_list = []
    for tensor in my_list:
        h, w, d = tensor.shape[1:]
        if h != max_h or w != max_w:
            new_list += [resample4D(tensor, (max_h, max_w, d))]
        else:
            new_list += [tensor]
    return np.ascontiguousarray(np.concatenate(new_list, axis=-1))
