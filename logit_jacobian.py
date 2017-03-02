# coding: utf-8
import theano.tensor as T
from theano.gradient import jacobian
import utils
import numpy as np
from theano import function, shared

X, Y = utils.make_classification()
X = np.c_[X, np.ones((200, 1))]
x, y = T.dmatrices('xy')

rng = np.random.RandomState(12345)
w = rng.rand(3, 1)
w = shared(rng.rand(3, 1), name="w")
utils.draw_decision_boundary(w.get_value(), X, Y, True)

z = T.dot(x, w)
activation = 1.0 / (1 + T.exp(-z))
loss = 0.5 * T.sum((y - activation) ** 2)
gw = jacobian(loss, w)
train = function([x, y], loss, updates=[(w, w - 0.3 * gw)])

for i in range(100):
    train(X, Y)

utils.draw_decision_boundary(w.get_value(), X, Y, True)
