import numpy as np
from numpy import ndarray
from steps.step01 import Variable


class Function:
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x**2


def main():
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(type(y))
    print(y.data)


if __name__ == "__main__":
    main()
