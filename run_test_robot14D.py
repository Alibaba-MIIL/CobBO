import numpy as np

from complex_functions.push_function import PushReward
from run_test_synthetic import test

def black_box_function(**kvargs):
    x = np.array([item[1] for item in sorted(kvargs.items(), key=lambda z: z[0])])
    return f(x)

if __name__ == "__main__":
    f = PushReward()
    pbounds = dict(zip([str(i).zfill(2) for i in np.arange(f.dx)], zip(f.xmin, f.xmax)))
    test(black_box_function, pbounds, 10000, 150)


