#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# JSM product code
#
# (C) Copyright 2017 Juxt SmartMandate Pvt Ltd
# All right reserved.
#
# This file is confidential and NOT open source.  Do not distribute.
#

"""

"""

import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def make_classification():
    rng = np.random.RandomState(12345)
    xx = rng.multivariate_normal([0.5, 0.5], [[0, 0.05], [0.05, 0]], size=(100,))
    yy = rng.multivariate_normal([-0.5, -0.5], [[0, 0.05], [0.05, 0]], size=(100,))
    X = np.r_[xx, yy]
    Y = np.ones((200, 1))
    Y[:100, :] = 0
    return X, Y


def perceptron_make_moons():
    X, y = make_moons(noise=0.04)
    X[y == 0, :] += 0.8
    y[y == 0] = -1
    y = y.reshape(-1, 1)
    return X, y


def draw_decision_boundary(weights, X, y, show=False):
    """Show the scatterplot of the data colored by the classes,
    draw the decision line based on the weights."""
    xx = np.linspace(X[:, 0].min(), X[:, 0].max())
    yy = - (weights[0] * xx + weights[2]) / weights[1]
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.plot(xx, yy)
    if show:
        plt.show()


def backprop_decision_boundary(predictor, X, y, show=False):
    y = np.argmax(y, axis=1)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = predictor(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.argmax(axis=1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    if show:
        plt.show()
