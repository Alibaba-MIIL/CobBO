import numpy as np

from complex_functions.helper import ConstantOffsetFn, NormalizedInputFn
from complex_functions.rover_function import create_large_domain
from run_test_synthetic import test

def l2cost(x, point):
    return 10 * np.linalg.norm(x - point, 1)

def black_box_function(**kvargs):
    x = np.array([item[1] for item in sorted(kvargs.items(), key=lambda z: z[0])])
    return f(x)

if __name__ == "__main__":
    domain = create_large_domain(force_start=False,
                                 force_goal=False,
                                 start_miss_cost=l2cost,
                                 goal_miss_cost=l2cost)
    n_points = domain.traj.npoints

    raw_x_range = np.repeat(domain.s_range, n_points, axis=1)

    # Maximum value of f
    f_max = 5.0
    f = ConstantOffsetFn(domain, f_max)
    f = NormalizedInputFn(f, raw_x_range)
    x_range = f.get_range()
    dim = len(x_range[0])

    pbounds = dict(zip([str(i).zfill(3) for i in np.arange(dim)], zip(x_range[0], x_range[1])))

    test(black_box_function, pbounds, 10000, 200)
