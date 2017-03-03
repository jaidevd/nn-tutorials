# coding: utf-8
import numpy as np
from sklearn.datasets import make_circles
from sklearn.preprocessing import OneHotEncoder
from theano import shared, function
import theano.tensor as T
from utils import backprop_decision_boundary

X, Y = make_circles(factor=0.1, noise=0.1)
Y = OneHotEncoder().fit_transform(Y.reshape(-1, 1)).toarray()

rng = np.random.RandomState(12345)
x, y = T.dmatrices('xy')
w1 = shared(rng.rand(2, 3), name="w1")
w2 = shared(rng.rand(3, 2), name="w2")
b1 = shared(rng.rand(1, 3), name="b1")
b2 = shared(rng.rand(1, 2), name="b2")

l1_activation = T.dot(x, w1) + b1.repeat(x.shape[0], axis=0)
l1_op = 1.0 / (1 + T.exp(-l1_activation))
l2_activation = T.dot(l1_op, w2) + b2.repeat(l1_op.shape[0], axis=0)
l2_op = 1.0 / (1 + T.exp(-l2_activation))
loss = 0.5 * T.sum((y - l2_op) ** 2)

get_loss = function([x, y], loss)
predict = function([x], l2_op)

backprop_decision_boundary(predict, X, Y, show=True)

gw1, gw2, gb1, gb2 = T.grad(loss, [w1, w2, b1, b2])
updates = [(w1, w1 - 0.2 * gw1), (w2, w2 - 0.2 * gw2), (b1, b1 - 0.2 * gb1),
           (b2, b2 - 0.2 * gb2)]
train = function([x, y], loss, updates=updates)

for i in range(500):
    train(X, Y)
backprop_decision_boundary(predict, X, Y, show=True)
