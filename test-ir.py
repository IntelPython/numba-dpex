import numpy as np
from numba import float64, vectorize


@vectorize([float64(float64, float64)])
def f(x, y):
    return x + y


a = np.ones(10)
b = np.ones(10)
c = f(a, b)
print(c)
