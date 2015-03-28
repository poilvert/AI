#!/usr/bin/env python

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
# Licence: BSD

"""
Linear SVM with One-Vs-All multi-class support using Scikit-learn.
"""

__all__ = ['MultiOVALinearLibSVM']

import numpy as np
import sklearn.svm as svm
from mydot import dot

DEFAULT_C = 1e5
DEFAULT_EPS = 1e-3

class MultiOVALinearLibSVM(object):

    def __init__(self, C=DEFAULT_C):
        self.C = C

    def fit(self, data, lbls):

        assert data.ndim == 2
        assert lbls.ndim == 1
        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

        categories = np.unique(lbls)
        assert categories.size >= 2

        ntrain = len(lbls)

        assert data.shape[0] == ntrain

        data = data.copy()
        data.shape = ntrain, -1

        # -- Normalizing data (zero-mean, unit std)
        fmean = data.mean(0)
        fstd = data.std(0)
        np.putmask(fstd, fstd <= DEFAULT_EPS, 1)
        data -= fmean
        data /= fstd

        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

        # -- Computing Train-Train Kernel
        kernel_traintrain = dot(data, data.T)
        ktrace = kernel_traintrain.trace()
        ktrace = ktrace != 0 and ktrace or 1
        kernel_traintrain /= ktrace

        # -- Training
        cat_index = {}
        alphas = {}
        support_vectors = {}
        biases = {}
        clfs = {}
        for icat, cat in enumerate(categories):

            ltrain = np.zeros(len(lbls))
            ltrain[lbls != cat] = -1
            ltrain[lbls == cat] = +1

            clf = svm.SVC(kernel='precomputed', C=self.C)
            clf.fit(kernel_traintrain, ltrain)

            alphas[cat] = clf.dual_coef_
            support_vectors[cat] = clf.support_
            biases[cat] = clf.intercept_
            cat_index[cat] = icat
            clfs[cat] = clf

        self._train_data = data
        self._ktrace = ktrace
        self._fmean = fmean
        self._fstd = fstd
        self._support_vectors = support_vectors
        self._alphas = alphas
        self._biases = biases
        self._clfs = clfs

        self.categories = categories

    def transform(self, data):

        assert data.ndim == 2
        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

        ntest = len(data)

        data = data.copy()

        # -- Normalizing data (zero-mean, unit std)
        data.shape = ntest, -1
        data -= self._fmean
        data /= self._fstd

        assert not np.isnan(data).any()
        assert not np.isinf(data).any()

        # -- Computing Train-Test Kernel
        kernel_traintest = dot(self._train_data, data.T)

        assert not np.isnan(kernel_traintest).any()
        assert not np.isinf(kernel_traintest).any()

        kernel_traintest /= self._ktrace

        assert not np.isnan(kernel_traintest).any()
        assert not np.isinf(kernel_traintest).any()

        # -- Finding Matching Categories
        categories = self.categories
        clfs = self._clfs

        outputs = np.zeros((ntest, len(categories)), dtype='float32')

        for icat, cat in enumerate(categories):
            clf = clfs[cat]
            resps = clf.decision_function(kernel_traintest.T).ravel()
            outputs[:, icat] = resps

        return outputs

    def predict(self, data):

        cats = self.categories

        outputs = self.transform(data)
        preds = outputs.argmax(1)
        lbls = [cats[pred] for pred in preds]

        return lbls
