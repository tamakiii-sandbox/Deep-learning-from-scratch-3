import pytest
from typing import Any
import numpy as np

from steps.step03 import Exp
from steps.step02 import Variable, Square


def test_exp_function():
    # Create a Variable instance with a scalar value
    x = Variable(np.array(2.0))  # np.exp(2) is about 7.389056
    exp_function = Exp()

    # Apply the Exp function
    result = exp_function(x)

    # Check that the result is correct
    expected = np.exp(2.0)
    np.testing.assert_almost_equal(
        result.data,
        expected,
        decimal=5,
        err_msg="Exp function did not compute correctly",
    )


def test_exp_function_with_negative_input():
    # Test with negative input
    x = Variable(np.array(-1.0))  # np.exp(-1) is about 0.367879
    exp_function = Exp()

    # Apply the Exp function
    result = exp_function(x)

    # Check that the result is correct
    expected = np.exp(-1.0)
    np.testing.assert_almost_equal(
        result.data,
        expected,
        decimal=5,
        err_msg="Exp function did not handle negative input correctly",
    )


# $y = (e^{x^2})^2$
def test_exponential_of_square_squared():
    # Define the operations
    A = Square()
    B = Exp()
    C = Square()

    # Create an input Variable
    x = Variable(np.array(0.5))

    # Apply the functions
    a = A(x)  # Square of x: x^2
    b = B(a)  # Exponential of a: e^(x^2)
    c = C(b)  # Square of b: (e^(x^2))^2

    # Calculate the expected result using numpy directly
    expected = (np.exp(0.5**2)) ** 2

    # Check if the computed result from the composite function matches the expected result
    np.testing.assert_almost_equal(
        c.data,
        expected,
        decimal=5,
        err_msg="Composite function (e^{x^2})^2 did not compute correctly",
    )
