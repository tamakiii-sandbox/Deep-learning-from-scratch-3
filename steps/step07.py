import numpy as np
from numpy import ndarray
from typing import Optional, Callable

class Variable:
    data: ndarray
    grad: Optional[ndarray] = None
    creator: Optional[Callable[['Variable'], 'Variable']] = None

    def __init__(self, data: ndarray):
        self.data: ndarray = data

    def set_creator(self, func: Callable[['Variable'], 'Variable']):
        self.creator = func

class Function:
    input: Variable
    output: Variable

    def __call__(self, input: Variable) -> Variable:
        x: ndarray = input.data
        y: ndarray = self.forward(x)
        output = Variable(y)
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
        return x ** 2

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

def main():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    assert y.creator == C

if __name__ == "__main__":
    main()
