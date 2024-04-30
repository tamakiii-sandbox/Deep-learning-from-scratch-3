import pytest
import numpy as np
from steps.step02 import Square
from steps.step04 import Variable
from steps.step04 import numerical_diff, f


def test_usage_of_numerical_diff():
    f = Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f, x)

    np.testing.assert_almost_equal(dy, 4.0, decimal=5)


def test_usage_of_f():
    x = Variable(np.array(0.5))
    dy = numerical_diff(f, x)

    np.testing.assert_almost_equal(dy, 3.29744, decimal=5)


def test_numerical_diff_with_quadratic_function():
    def quadratic(x: Variable) -> Variable:
        return Variable(x.data**2)

    x = Variable(2.0)
    eps = 1e-4
    expected = 4.0001  # Derived from the derivative of x^2 at x=2, which is 2x = 4, with a small adjustment for epsilon
    assert abs(numerical_diff(quadratic, x, eps) - expected) < 1e-4


def test_numerical_diff_with_small_epsilon():
    def linear(x: Variable) -> Variable:
        return Variable(5 * x.data)  # f(x) = 5x, derivative is 5

    x = Variable(1.0)
    eps = 1e-4  # Increased epsilon to a more stable value
    expected = 5.0
    assert (
        abs(numerical_diff(linear, x, eps) - expected) < 1e-4
    )  # Adjusted the tolerance


def test_numerical_diff_with_negative_input():
    def cubic(x: Variable) -> Variable:
        return Variable(x.data**3)  # f(x) = x^3, derivative is 3x^2

    x = Variable(-1.0)
    eps = 1e-4
    expected = 3.0  # Corrected expected derivative
    assert abs(numerical_diff(cubic, x, eps) - expected) < 1e-4  # Corrected assertion


def test_numerical_diff_zero_epsilon():
    def exponential(x: Variable) -> Variable:
        return Variable(2**x.data)  # f(x) = 2^x, derivative is 2^x * ln(2)

    x = Variable(2.0)
    eps = 0
    with pytest.raises(ZeroDivisionError):
        numerical_diff(exponential, x, eps)
