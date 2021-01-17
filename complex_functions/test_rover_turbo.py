from turbo_1.turbo_1 import Turbo1
import numpy as np
from rover_function import create_large_domain

# Let's start by definying our function, bounds, and instanciating an optimization object.
#def black_box_function(x1, x2, x3, x4, x5, x6, x7):
#    return -(x1 -1) ** 2 - (x2 - 1) ** 2 - (x3 - 1) ** 2 - (x4 - 1) ** 2 - 0.1*(x5 - 1) ** 2 - 0.1*(x6 - 1) ** 2 - 0.1*(x7 - 1) ** 2

def l2cost(x, point):
    return 10 * np.linalg.norm(x - point, 1)


domain = create_large_domain(force_start=False,
                             force_goal=False,
                             start_miss_cost=l2cost,
                             goal_miss_cost=l2cost)
n_points = domain.traj.npoints

raw_x_range = np.repeat(domain.s_range, n_points, axis=1)

from helper import ConstantOffsetFn, NormalizedInputFn

# maximum value of f
f_max = 5.0
f = ConstantOffsetFn(domain, f_max)
f = NormalizedInputFn(f, raw_x_range)
x_range = f.get_range()
dim = len(x_range[0])

pbounds = dict(zip([str(i).zfill(3) for i in np.arange(dim)], zip(x_range[0], x_range[1])))



class Rover:
    def __init__(self, dim=1):
        self.dim    = dim
        self.lb      = x_range[0]
        self.ub      =  x_range[1]
        self.counter = 0

    def __call__(self, x):
        self.counter += 1
        result = -f(x)
        return result

blackbox = Rover()

def test_turbo(num, init):
    turbo1 = Turbo1(
        f=blackbox,  # Handle to objective function
        lb=x_range[0],  # Numpy array specifying lower bounds
        ub=x_range[1],  # Numpy array specifying upper bounds
        n_init=init,  # Number of initial bounds from an Latin hypercube design
        max_evals=num,  # Maximum number of evaluations
        batch_size=1,  # How large batch size TuRBO uses
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float32"  # float64 or float32
    )
    turbo1.optimize()

if __name__ == "__main__":
    print(sorted(pbounds.items(), key=lambda z:z[0]))
    test_turbo(10000, 200) #2500, 150
