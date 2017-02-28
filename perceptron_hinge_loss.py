# coding: utf-8
from theano import shared, function
import theano.tensor as T
import numpy as np

rng = np.random.RandomState(12345)
x = T.dmatrix('x')
y = T.dmatrix('y')
w = shared(rng.rand(2, 1), name="w")
b = shared(rng.rand(), name="b")
activation = T.dot(x, w) + b
pLoss = T.sum(T.max(T.concatenate((T.zeros((x.shape[0], 1)), - activation * y),
                                  axis=1), axis=1), axis=0) / x.shape[0]
gw, gb = T.grad(pLoss, [w, b])
train = function([x, y], [pLoss], updates=[(w, w - gw), (b, b - gb)])
predict = function([x], [activation])

xx = rng.multivariate_normal([0.5, 0.5], [[0, 0.05], [0.05, 0]], size=(100,))
yy = rng.multivariate_normal([-0.5, -0.5], [[0, 0.05], [0.05, 0]], size=(100,))
X = np.r_[xx, yy]
Y = np.ones((200, 1))
Y[:100, :] = 0

predict(X)
for i in range(10):
    print(train(X, Y))
