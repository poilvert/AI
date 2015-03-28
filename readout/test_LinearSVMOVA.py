#!/usr/bin/env python

# Authors: Nicolas Pinto <nicolas.pinto@gmail.com>
#          Nicolas Poilvert <nicolas.poilvert@gmail.com>
# Licence: BSD

from LinearSVMOVA import MultiOVALinearLibSVM
from sklearn.datasets import load_digits

def test_Linear_SVM_OVA_on_digits():

    # -- Loading the "Digits" dataset from sklearn
    ds = load_digits()
    X, y = ds.data, ds.target

    # -- split data and targets into "train" and "test"
    X_trn, X_tst = X[:1500], X[1500:]
    y_trn, y_tst = y[:1500], y[1500:]

    # -- classifier training
    clf = MultiOVALinearLibSVM()
    clf.fit(X_trn, y_trn)

    # -- predictions on the "test" set
    y_pred = clf.predict(X_tst)

    # -- Accuracy
    acc = 1. * (y_pred == y_tst).sum() / y_tst.size
    assert acc > 0.87
