import numpy as np
from numpy import ndarray


class Variable:
    def __init__(self, data: ndarray):
        self.data = data


def main():
    data = np.array(1.0)
    x = Variable(data)
    print("x.data:", x.data)

    a = np.array(1)
    print("a.ndim:", a.ndim)

    b = np.array([1, 2, 3])
    print("b.ndim:", b.ndim)

    c = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    print("c.ndim:", c.ndim)


if __name__ == "__main__":
    main()
