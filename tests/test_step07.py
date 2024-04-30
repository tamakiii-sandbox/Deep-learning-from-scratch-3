import pytest
import numpy as np
from steps.step07 import Variable, Square, Exp, Square


def test_procedural():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)

    C = y.creator
    b = C.input
    b.grad = C.backward(y.grad)

    B = b.creator
    a = B.input
    a.grad = B.backward(b.grad)

    A = a.creator
    x = A.input
    x.grad = A.backward(a.grad)

    np.testing.assert_almost_equal(x.grad, 3.29744, decimal=5)


def test_autopropagation():
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
