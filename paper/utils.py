# coding=utf-8
from os.path import join
from matplotlib import offsetbox
import numpy as np
from os.path import isfile
from scipy.misc.pilutil import imread, imresize
from matplotlib import pyplot as plt
import os

__author__ = 'Michał Ciołczyk'


def read_image(name):
    return imread(name, True)


def greyscale_to_rgb(im):
    return np.array([im, im, im])


def load_dataset(dir_name, fifty=True):
    X = []
    i = 0
    for f in os.listdir(dir_name):
        if isfile(join(dir_name, f)):
            im = read_image(join(dir_name, f))
            X.append(im.ravel())
            i += 1
            if i >= 50 and fifty:
                break
    return np.array(X)


def plot_embedding(X, X_orig, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1])

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 12e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(
                    imresize(greyscale_to_rgb(X_orig[i].reshape((720, 1280))), (36, 64))
                ),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
