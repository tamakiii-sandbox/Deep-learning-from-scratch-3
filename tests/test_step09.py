import pytest
import numpy as np
from steps.step09 import square, exp, Variable


def test():
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))

    y.grad = np.array(1.0)
    y.backward()

    np.testing.assert_almost_equal(x.grad, 3.29744, decimal=5)


def test_only_backward():
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()

    np.testing.assert_almost_equal(x.grad, 3.29744, decimal=5)
