import pytest
import numpy as np
from steps.step08 import Variable, Square, Exp, Square


def test():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    y.backward()

    np.testing.assert_almost_equal(x.grad, 3.29744, decimal=5)
