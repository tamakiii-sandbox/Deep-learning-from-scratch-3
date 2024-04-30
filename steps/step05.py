import numpy as np
from numpy import ndarray


class Variable:
    def __init__(self, data: ndarray):
        self.data = data
        self.grad = None  # 勾配（Gradient）


class Function:
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        return output

    def forward(self, x: ndarray):
        raise NotImplementedError()

    def backward(self, gy: ndarray):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: ndarray):
        y = x**2
        return y

    def backward(self, gy: ndarray):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: ndarray):
        y = np.exp(x)
        return y

    def backward(self, gy: ndarray):
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

    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)

    print(y.data)
    print(x.grad)


if __name__ == "__main__":
    main()
