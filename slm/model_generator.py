#!/usr/bin/env python

# Authors: Nicolas Poilvert <nicolas.poilvert@gmail.com>
# License: BSD

__all__ = ['generate_random_slm_model']

import numpy as np
from numpy.linalg import norm
from numpy.linalg import svd
from mydot import dot
import random

DEFAULT_SEED = 42

def generate_filter_bank(
        nb,
        depth=16,
        n_filters=32,
        center_filters=True,
        normalize_filters=True,
        orthonormal_filters=True,
        seed=DEFAULT_SEED,
    ):

    # -- Lowdin-type orthonormalization
    def _Lowdin_orthonormalization(fbank):
        nh, nw, nd, nf = fbank.shape
        fbank_mat = fbank.reshape(nh*nw*nd, nf)
        U, s, V = svd(fbank_mat, full_matrices=False)
        Lowdin_mat = dot(U, V)
        assert Lowdin_mat.shape == fbank_mat.shape
        return Lowdin_mat.reshape(nh, nw, nd, nf)

    # -- seeding the numpy random number generator
    np.random.seed(seed)

    filter_bank = np.random.randn(nb, nb, depth, n_filters)
    if center_filters:
        for i in xrange(n_filters):
            filter_bank[..., i] -= filter_bank[..., i].mean()
    if normalize_filters:
        for i in xrange(n_filters):
            filter_bank[..., i] /= norm(filter_bank[..., i].ravel())
    if orthonormal_filters:
        filter_bank = _Lowdin_orthonormalization(filter_bank)

    return filter_bank

def generate_layer(
        lnorm_nb, remove_mean, normalize,
        fbcorr_nb, previous_depth, n_filters,
        center_filters, normalize_filters, orthonormal_filters,
        lpool_nb, order, stride,
        activate_min_val=0., activate_max_val=1.,
        seed=DEFAULT_SEED,
    ):

    assert isinstance(remove_mean, bool)
    assert isinstance(normalize, bool)

    params = [
        {
          "type": "lnorm",
          "neighborhood_shape": (int(lnorm_nb), int(lnorm_nb)),
          "remove_mean": remove_mean,
          "normalize": normalize,
        },
        {
          "type": "fbcorr",
          "filter_bank": generate_filter_bank(
                            fbcorr_nb,
                            depth=previous_depth,
                            n_filters=n_filters,
                            center_filters=center_filters,
                            normalize_filters=normalize_filters,
                            orthonormal_filters=orthonormal_filters,
                            seed=seed,
                        ),
          "center_filters": center_filters,
          "normalize_filters": normalize_filters,
          "orthonormal_filters": orthonormal_filters,
        },
        {
          "type": "activate",
          "min_val": activate_min_val,
          "max_val": activate_max_val,
        },
        {
          "type": "lpool",
          "neighborhood_shape": (lpool_nb, lpool_nb),
          "order": order,
          "stride": stride,
        }
    ]

    return params

def generate_random_layer(
        previous_depth,
        seed=DEFAULT_SEED,
        lnorm_nb_range=[3, 5, 7 ,9],
        remove_mean_range=[True, False], normalize_range=[True, False],
        fbcorr_nb_range=[3, 5, 7, 9], n_filters_range=[16, 32, 64, 128],
        center_filters=True, normalize_filters=True, orthonormal_filters=True,
        lpool_nb_range=[2], order_range=[1., 2., 10.], stride=2,
        activate_min_val=0., activate_max_val=1.,
    ):

    layer = generate_layer(
        random.choice(lnorm_nb_range),
        random.choice(remove_mean_range),
        random.choice(normalize_range),
        random.choice(fbcorr_nb_range),
        previous_depth,
        random.choice(n_filters_range),
        center_filters,
        normalize_filters,
        orthonormal_filters,
        random.choice(lpool_nb_range),
        random.choice(order_range),
        stride,
        activate_min_val=0.,
        activate_max_val=1.,
        seed=seed,
    )

    return layer

def generate_random_slm_model(
        n_layers,
        seed=DEFAULT_SEED,
        lnorm_nb_range=[3, 5, 7 ,9],
        remove_mean_range=[True, False], normalize_range=[True, False],
        fbcorr_nb_range=[3, 5, 7, 9], n_filters_range=[16, 32, 64, 128],
        center_filters=True, normalize_filters=True, orthonormal_filters=True,
        lpool_nb_range=[2], order_range=[1., 2., 10.], stride=2,
        activate_min_val=0., activate_max_val=1.,
        starting_depth=1
    ):

    assert n_layers >= 1
    depth = starting_depth
    model = []
    for _ in xrange(n_layers):
        layer = generate_random_layer(
                    depth,
                    seed=seed,
                    lnorm_nb_range=lnorm_nb_range,
                    remove_mean_range=remove_mean_range,
                    normalize_range=normalize_range,
                    fbcorr_nb_range=fbcorr_nb_range,
                    n_filters_range=n_filters_range,
                    center_filters=center_filters,
                    normalize_filters=normalize_filters,
                    orthonormal_filters=orthonormal_filters,
                    lpool_nb_range=lpool_nb_range,
                    order_range=order_range,
                    stride=stride,
                    activate_min_val=activate_min_val,
                    activate_max_val=activate_max_val,
                )
        depth = layer[1]["filter_bank"].shape[-1]
        model += [layer]

    return model
