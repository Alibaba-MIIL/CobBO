import numpy as np
from datetime import datetime
from CobBO.optimizer import CobBO

from synthetic_functions import ackley, Levy, Rastrigin

def black_box_function(**kvargs):
    x = np.array([item[1] for item in sorted(kvargs.items(), key=lambda z: z[0])])
    return Rastrigin(x)


def maximize(obj_func, optimizer):
    """Maximize a given objective function

    Parameters
    ----------
    optimizer : The CobBO optimizer object
    obj_func : method
        The objective function to be optimized
    verbose : bool
        Print the optimization progress details and total runtime

    Returns
    -------
    best_solution : list array-like
        List of `n_suggestions` suggestions to evaluate the objective
        function. Each suggestion is a dictionary where each key
        corresponds to a parameter being optimized.
    """
    assert isinstance(optimizer, CobBO), ' A CobBO optimizer is expected'

    while optimizer.has_budget:
        x_probe_list = optimizer.suggest(n_suggestions=optimizer.batch)
        target_list = obj_func(**x_probe_list)
        optimizer.observe(x_probe_list, target_list)

    return optimizer.best_point


def test(black_box_func, pbounds, num_iter, init_points=0):
    optimizer = CobBO(
        api_config=None,
        pbounds=pbounds,
        n_iter=num_iter,
        init_points=init_points,
        batch=1,
    )

    # The optimizer maximizes the function f
    begin_time = datetime.now()
    solution = maximize(black_box_func, optimizer)
    print('Total Runtime [seconds]:', (datetime.now() - begin_time).total_seconds())


if __name__ == "__main__":
    # Let's start by defining our function, bounds, and instanciating an optimizer object.
    # Dimention of your problem
    dim = 100

    # Domain of each dimension
    pbounds = dict(zip([str(i).zfill(3) for i in np.arange(dim)], [(-5, 10)] * dim))

    test(black_box_function, pbounds, 10000, 300)
