import pytest
from typing import Any
import numpy as np
from steps.step01 import Variable
from steps.step02 import Function, Square


def test_square_function():
    # Setup: create a Variable object with a specific numpy array
    x = Variable(np.array(10))
    square_function = Square()

    # Action: apply the Square function
    result = square_function(x)

    # Assert: check if the output is as expected
    assert isinstance(result, Variable), "The result should be an instance of Variable"
    assert np.array_equal(result.data, np.array(100)), "The square of 10 should be 100"


def test_function_call_not_implemented():
    # Setup: create a Variable and an instance of the base Function class
    x = Variable(np.array(10))
    base_function = Function()

    # Assert: ensure that calling the base function raises NotImplementedError
    with pytest.raises(NotImplementedError):
        base_function(x)
