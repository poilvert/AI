#!/usr/bin/env python

# Authors: Nicolas Poilvert <nicolas.poilvert@gmail.com>
# License: BSD

"""
A simple and dense Kernel Ridge Regressor. Built after Max Weilling's note on
Kernel Ridge Regression.

Cost function
=============

Given a vector of adjustable parameters w, a set of (xi, yi) training examples
(with xi a d-dimensional vector and yi a real number for all i) and a parameter
for the regularization cost l, the cost function that this code minimizes is the
following::

    C(w) = 1/2 sum[(yi - w.T * xi) ** 2] + 1/2 * l * ||w|| ** 2

where ``||w||`` denotes the Euclidean norm (i.e. 2-norm) of vector w, and
``w.T`` its transpose or row vector. In the case of a non-linear kernel, xi is
first transformed into a ``feature space`` and the resulting feature vector is
used instead of xi for all i.

Optimal leave-one-out cross validated regularization
====================================================

Given a kernel K (with fixed kernel parameters), the KRR can take advantage of
many optimizations.

    1. The first one is to be able to compute the kernel pseudo-inverse for any
    regularization parameter l by knowing the eigenvalue decomposition of the
    kernel. For this we have::

        K = Q * L * Q.T
        (K + l * I) ** (-1) = Q * (L + l * I) ** (-1) * Q.T

    where *L* is the diagonal matrix of eigenvalues of the kernel K. So once we
    know the eigendecomposition of K, we can compute the pseudo inverse matrices
    for all the regularization parameters with only matrix-matrix
    multiplications.

    2. The second optimization has to do with the fact that we can get access to
    the leave-one-out errors at pretty much no cost. Indeed, for a kernel matrix
    K, and a regularization parameter l, the vector of LOO errors (so for each i
    in this vector, error[i] is the prediction error on the i-th sample, when a
    KRR regressor has been trained on **all** the other samples) is given by the
    following analytical expression::

        error[i] = (y[i] - dot(K * (K + l * I) ** (-1), y)[i])
        / (1. - (K * (K + l * I) ** (-1))[i, i])

    where *dot* is the matrix-vector dot product and *y* is the vector of target
    values to regress (for the training part).
"""

__all__ = ['KernelRidgeRegressor']

import numpy as np
import numexpr as ne
from mydot import dot
from scipy.spatial.distance import cdist

# -- Default Kernel-related parameters
DEFAULT_KERNEL_TYPE = 'linear'
DEFAULT_KERNEL_TYPES = ['linear', 'laplace', 'gaussian']
DEFAULT_KERNEL_KWARGS = {'sigma': 1.}

# ------------
# -- Utilities
# ------------
def get_L1_matrix(A, B):

    assert A.ndim == 2
    assert B.ndim == 2
    assert A.shape[1] == B.shape[0]

    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)

    d_AB = cdist(A, B.T, 'cityblock')

    return d_AB


def get_L2_matrix(A, B):

    assert A.ndim == 2
    assert B.ndim == 2
    assert A.shape[1] == B.shape[0]

    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)

    d_AB = cdist(A, B.T, 'sqeuclidean')

    return d_AB


def get_kernel_matrix(
    left_mat, right_mat,
    kernel_type=DEFAULT_KERNEL_TYPE,
    kernel_kwargs=DEFAULT_KERNEL_KWARGS
    ):
    """A linear kernel returns the dot product matrix. The Laplace and Gaussian
    kernels are defined in equations (9) and (8) of Table 2 in the reference
    JCTC paper.
    """

    assert left_mat.ndim == 2
    assert right_mat.ndim == 2
    assert left_mat.shape[1] == right_mat.shape[0]
    assert kernel_type in DEFAULT_KERNEL_TYPES

    if kernel_type == 'linear':
        K = dot(left_mat, right_mat)

    elif kernel_type == 'laplace':
        one_over_sigma = 1. / float(kernel_kwargs['sigma'])
        scaling = -1. * one_over_sigma
        lr_diff = get_L1_matrix(left_mat, right_mat)
        K = ne.evaluate("exp(scaling * lr_diff)")

    elif kernel_type == 'gaussian':
        one_over_sigma_square = 1. / (float(kernel_kwargs['sigma']) ** 2)
        scaling = -0.5 * one_over_sigma_square
        lr_diff = get_L2_matrix(left_mat, right_mat)
        K = ne.evaluate("exp(scaling * lr_diff)")

    return K


# -------------
# -- Base Class
# -------------
class KernelRidgeRegressor(object):

    def __init__(self,
                 kernel_type=DEFAULT_KERNEL_TYPE,
                 kernel_kwargs=DEFAULT_KERNEL_KWARGS,
                 regularization=[1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7,
                     1e-6, 5e-6, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                     1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1],
                 ):

        assert kernel_type in DEFAULT_KERNEL_TYPES

        self.kernel_type = kernel_type
        self.kernel_kwargs = kernel_kwargs
        self.regularization = regularization

    def fit(self, X, y):
        """
        This method will take advantage of two things:

            1. It will perform an eigenvalue decomposition of the kernel matrix
            to easily compute the inverse of (K + l*I) for all values of the
            regularization l.
            2. For each regularization parameter, it will exploit known exact
            results to compute with no extra work the LOO cross validation
            score.

        So given some kernel parameters fixed (given at initialization), this
        method will auto-tune the regularization for best LOO cross validated
        score.
        """

        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.size
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()
        assert not np.isnan(y).any()
        assert not np.isinf(y).any()

        X = np.ascontiguousarray(X, dtype='d')
        y = np.ascontiguousarray(y, dtype='d')

        n_samples, n_features = X.shape
        self.n_samples = n_samples
        self.n_features = n_features

        K = get_kernel_matrix(
                X, X.T,
                kernel_type=self.kernel_type,
                kernel_kwargs=self.kernel_kwargs,
                )

        # -- We determined the optimal regularization here by minimizing the LOO
        # cross validation error on the whole training set X
        eigs, Q = np.linalg.eigh(K)

        residual_vec_l = []
        for l in self.regularization:
            Kl_pseudo_inverse = dot(Q, dot(np.diag(1./(eigs + l)), Q.T))
            Hl = dot(K, Kl_pseudo_inverse)
            residual_vec_l += [(1./(1. - np.diag(Hl))) * (y - dot(Hl, y))]

        best_regularization_idx = np.argmin(
                    [np.mean(residual_vec ** 2)
                     for residual_vec in residual_vec_l]
                )
        l_opt = self.regularization[best_regularization_idx]

        # -- we finally keep the best regularization parameter and use it
        opt_Kl_pseudo_inverse = dot(Q, dot(np.diag(1./(eigs + l_opt)), Q.T))
        z = dot(opt_Kl_pseudo_inverse, y)

        self.z = z
        self.X_trn = X
        self.loo_residuals = residual_vec_l[best_regularization_idx]

    def predict(self, X):

        assert X.ndim == 2
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()
        assert X.shape[1] == self.n_features

        X = np.ascontiguousarray(X, dtype='d')
        K_tst_trn = get_kernel_matrix(
                X, self.X_trn.T,
                kernel_type=self.kernel_type,
                kernel_kwargs=self.kernel_kwargs,
                )
        pred = dot(K_tst_trn, self.z)

        return pred
