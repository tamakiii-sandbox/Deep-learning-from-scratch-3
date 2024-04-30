import pytest
import numpy as np
from steps.step05 import Square, Exp, Variable


def test():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)

    np.testing.assert_almost_equal(y.data, 1.64872, decimal=5)
    np.testing.assert_almost_equal(x.grad, 3.29744, decimal=5)
