import numpy as np
from steps.step02 import Function, Square
from steps.step01 import Variable  # Make sure this import is correct


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


def main():
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    c = C(b)

    print(c.data)


if __name__ == "__main__":
    main()
