from steps.step01 import Variable


def numerical_diff(f: callable, x: Variable, eps: float = 1e-4) -> float:
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)
