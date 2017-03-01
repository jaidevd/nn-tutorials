# coding: utf-8
from theano import shared, function
from theano.gradient import jacobian
import theano.tensor as T
import numpy as np
from utils import perceptron_make_moons, draw_decision_boundary

rng = np.random.RandomState(12345)

x = T.dmatrix('x')
y = T.dmatrix('y')
w = shared(rng.rand(3, 1), name="w")

X, Y = perceptron_make_moons()

X = np.c_[X, np.ones((100, 1))]
activation = T.dot(x, w)
pLoss = T.sum(T.max(T.concatenate((T.zeros((x.shape[0], 1)), - activation * y),
                                  axis=1), axis=1), axis=0) / x.shape[0]
get_loss = function([x, y], pLoss)
myjacobian = jacobian(pLoss, [w])
get_jacobian = function([x, y], myjacobian)

draw_decision_boundary(w.get_value().ravel(), X, Y, True)

# equivalent of train

for i in range(20):
    loss = get_loss(X, Y)
    _jac = get_jacobian(X, Y)[0]
    _w = w.get_value()
    print(loss, _jac.ravel())
    w.set_value(_w - _jac)

draw_decision_boundary(w.get_value().ravel(), X, Y, True)
