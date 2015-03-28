#!/usr/bin/env python

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skdata import pubfig83

DEFAULT_USE_COLOR_SPACE = False

def get_X_y(
        img_shape=(128, 128),
        n_imgs_per_class=20,
        n_classes=5,
        use_color_space=DEFAULT_USE_COLOR_SPACE,
        seed=42
        ):

    # -- Seeding the Numpy random number generator
    np.random.seed(seed)

    # -- PubFig83 dataset object and its metadata
    ds = pubfig83.PubFig83()
    meta = ds.meta

    # -- Loading the dataset labels and unique classes
    labels = np.array([item['name'] for item in meta])
    classes = np.unique(labels)
    tot_imgs_per_class = np.array(
            [(labels == myclass).sum() for myclass in classes]
        )

    assert labels.ndim == 1
    assert tot_imgs_per_class.min() >= n_imgs_per_class
    assert tot_imgs_per_class.sum() == labels.size

    # -- Filenames associated to the labels
    filenames = np.array([item['filename'] for item in meta])

    assert filenames.ndim == 1
    assert filenames.size == labels.size

    # -- getting the indices of all the necessary labels on a per-class basis
    class_idx = np.random.permutation(classes.size)[:n_classes]
    sel_classes = classes[class_idx]
    class_indices_l = []
    for myclass in sel_classes:

        myclass_indices = (labels == myclass).nonzero()[0]

        assert myclass_indices.ndim == 1
        assert len(myclass_indices) >= n_imgs_per_class

        n_myclass = myclass_indices.size
        random_order = np.random.permutation(n_myclass)[:n_imgs_per_class]
        class_indices_l += [myclass_indices[random_order]]

    # -- Returning all input images and input labels (X, y)
    final_indices = np.concatenate(class_indices_l)
    y = labels[final_indices]

    assert y.ndim
    assert np.unique(final_indices).size == final_indices.size

    if not use_color_space:
        raw_imgs = np.array([
                np.array(
                resize(
                    imread(filenames[i], as_grey=True), img_shape
                    )
                )
                for i in final_indices
        ])
        X = raw_imgs[..., None]
    else:
        img_shape = img_shape + (3,)
        raw_imgs = np.array([
                np.array(
                resize(
                    imread(filenames[i], as_grey=False), img_shape
                    )
                )
                for i in final_indices
        ])
        X = raw_imgs

    assert X.ndim == 4
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()

    return np.ascontiguousarray(X), np.ascontiguousarray(y)
