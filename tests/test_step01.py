import numpy as np
import pytest
from steps.step01 import Variable


def test_variable_initialization():
    data = np.array(1.0)
    var = Variable(data)
    assert np.array_equal(
        var.data, data
    ), "The data should be correctly assigned to var.data"


def test_variable_data_type():
    data = np.array(1.0)
    var = Variable(data)
    assert isinstance(
        var.data, np.ndarray
    ), "var.data should be an instance of numpy.ndarray"
