# coding: utf-8
from theano import function
from theano.gradient import jacobian, hessian
import theano.tensor as T

# Jacobian
x = T.dscalar('x')
y = T.dscalar('y')
f = (x ** 2) * y + y
T.grad(f, [x, y])
del_f = jacobian(f, [x, y])
my_jacobian_function = function([x, y], del_f)

# Hessian
x = T.dvector('x')
f = (x[0] ** 2) * x[1] + x[1]
jf = f.sum()
jHess = hessian(jf, [x])
my_hessian_func = function([x], jHess)
