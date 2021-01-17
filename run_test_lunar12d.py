import numpy as np

from complex_functions.lunar12d import lunarlander_func
from run_test_synthetic import test

def black_box_function(**kvargs):
    x = np.array([item[1] for item in sorted(kvargs.items(), key=lambda z: z[0])])
    return lunarlander_func(x)

if __name__ == "__main__":
    dim = 12
    pbounds = dict(zip([str(i).zfill(3) for i in np.arange(dim)], [(0, 2)] * dim))
    test(black_box_function, pbounds, 1500, 50)
