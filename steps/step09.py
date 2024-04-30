from __future__ import annotations
import numpy as np
from numpy import ndarray
from typing import Optional, Callable, NewType


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Variable:
    data: ndarray
    grad: Optional[ndarray] = None
    creator: Optional[Function] = None

    def __init__(self, data: ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))

        self.data: ndarray = data

    def set_creator(self, func: Function):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        assert isinstance(self.creator, Function)
        funcs: list[Function] = [self.creator]
        while funcs:
            f: Function = funcs.pop()
            x: Variable = f.input
            y: Variable = f.output
            assert y.grad is not None
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    input: Variable
    output: Variable

    def __call__(self, input: Variable) -> Variable:
        x: ndarray = input.data
        y: ndarray = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x: ndarray) -> ndarray:
        raise NotImplementedError

    def backward(self, gy: ndarray) -> ndarray:
        raise NotImplementedError


class Square(Function):
    def forward(self, x: ndarray) -> ndarray:
        return x**2

    def backward(self, gy: ndarray) -> ndarray:
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: ndarray) -> ndarray:
        return np.exp(x)

    def backward(self, gy: ndarray) -> ndarray:
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x: Variable):
    f = Square()
    return Square()(x)


def exp(x: Variable):
    f = Exp()
    return f(x)
