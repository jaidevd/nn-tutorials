# coding: utf-8
from theano import shared, function
import theano.tensor as T
import numpy as np
import utils

rng = np.random.RandomState(12345)

X, Y = utils.make_classification()
X = np.c_[X, np.ones((200, 1))]
x, y = T.dmatrices('xy')
w = rng.rand(3, 1)
w = shared(rng.rand(3, 1), name="w")

utils.draw_decision_boundary(w.get_value(), X, Y, True)

z = T.dot(x, w)
activation = 1.0 / (1 + T.exp(-z))
pLoss = 0.5 * T.sum((y - activation) ** 2)
gw = T.grad(pLoss, w)
train = function([x, y], pLoss, updates=[(w, w - 0.1 * gw)])

for i in range(100):
    train(X, Y)

utils.draw_decision_boundary(w.get_value(), X, Y, True)
