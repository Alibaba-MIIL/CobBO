import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from copy import deepcopy

def ts_sampling(ac, gp, y_max, x_max, bounds, random_state, top_sample=1):
    dim = bounds.shape[0]
    sample = np.clip(200*dim, 500, 3000)
    bounds = shrink_bounds_around_pmax(bounds, x_max)
    with warnings.catch_warnings(record=True) as w:
        x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(sample, dim))
        if x_max is not None and len(x_max) > 0:
            # perturb
            prob_perturb = np.clip(7.0 / dim, 0.5, 0.9)
            for row in x_tries:
                mask = np.random.rand(dim) > prob_perturb
                row[mask] = x_max[mask]

        values = gp.sample_y(x_tries).ravel()
        if top_sample == 1:
            index = values.argmax()
        else:
            index = np.argpartition(values, -top_sample)[-top_sample:]
        point, value = x_tries[index], values[index]
        if len(w):
            return l_bfgs_b(ac, gp, y_max, bounds, random_state, n_warmup=500)

        return np.clip(point, bounds[:, 0], bounds[:, 1]), value

def l_bfgs_b(ac, gp, y_max, bounds, random_state, n_warmup):
    dim = bounds.shape[0]
    point_o = None
    value_o = -np.inf
    num = 1
    for i in range(num):
        n_warmup = n_warmup // (1+100*i)
        x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1], size=(n_warmup, dim))
        ys = ac(x_tries, gp=gp, y_max=y_max)
        index = ys.argmax()
        x_seed = x_tries[index]
        value = ys[index]
        point = x_seed

        try:
            res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                           x_seed,
                           bounds=bounds,
                           method="L-BFGS-B",
                           options={'maxiter': 9000, 'disp': False})
            if res.success:
                point = res.x
                value = -res.fun[0]
        except:
            pass

        if value > value_o:
            value_o = value
            point_o = point

    return np.clip(point_o, bounds[:, 0], bounds[:, 1]), value_o

def acq_max(ac, gp, multisample, y_max, x_max, bounds, random_state, top_sample):
    dim = bounds.shape[0]

    select_points = np.empty((1, dim))
    values = np.empty(1)
    n_warmup = 10000
    if multisample == 1:
        select_points[0], values[0] = l_bfgs_b(ac, gp, y_max, bounds, random_state, n_warmup)
    elif multisample == 2:
        try:
            select_points[0], values[0] = ts_sampling(ac, gp, y_max, x_max, bounds, random_state)
        except:
            select_points[0], values[0] = l_bfgs_b(ac, gp, y_max, bounds, random_state, n_warmup)
    elif multisample == 3:
        return mix_bfgx_ts(ac, gp, y_max, x_max, bounds, random_state)
    elif multisample == 4:
        return mix_bfgx_ts(ac, gp, y_max, x_max, bounds, random_state, top_sample)

    return select_points

def mix_bfgx_ts(ac, gp, y_max, x_max, bounds, random_state, top_sample=1):
    dim = bounds.shape[0]
    select_points = np.empty((2, dim))
    values = np.empty(2)

    n_warmup = 10000
    select_points[0], values[0] = l_bfgs_b(ac, gp, y_max, bounds, random_state, n_warmup)

    try:
        select_points[1], values[1] = ts_sampling(ac, gp, y_max, x_max, bounds, random_state, top_sample)
    except:
        select_points[1], values[1] = l_bfgs_b(ac, gp, y_max, bounds, random_state, int(n_warmup/50))

    return select_points

def shrink_bounds_around_pmax(bounds, pmax):
    ratio = 0.7
    for row in range(len(bounds)):
        bound = bounds[row]
        low = bound[0]
        up = bound[1]
        c = (up + low) / 2.0
        interval = (up - low) * ratio
        point = pmax[row]
        if point < c:
            if point - low >= interval / 2.0:
                bounds[row][0] = point - interval / 2.0
                bounds[row][1] = point + interval / 2.0
            else:
                bounds[row][1] = low + interval
        else:
            if up - point >= interval / 2.0:
                bounds[row][0] = point - interval / 2.0
                bounds[row][1] = point + interval / 2.0
            else:
                bounds[row][0] = up - interval
    return bounds

def normalize_probability(probability):
    probability = 0.9999 * probability / sum(probability)
    probability[-1] = 1.0 - sum(probability[0:len(probability) - 1])
    return probability

def assign_probability(dim):
    probability = np.ones(dim) / float(dim)
    return normalize_probability(probability)

def order_stats(X):
    _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
    obs = np.cumsum(cnt)
    o_stats = obs[idx]
    return o_stats

def copula_gaussian(XX, scale=1):
    X = deepcopy(XX)
    X = np.nan_to_num(np.asarray(X))
    assert X.ndim == 1 and np.all(np.isfinite(X))
    o_stats = order_stats(X)
    quantile = np.true_divide(o_stats, len(X) + 1)
    X_ss = norm.ppf(quantile, scale=scale)
    return X_ss

class UtilityFunction(object):

    def __init__(self, kind, dim, kappa, xi):
        self.kappa = kappa
        self.dim = dim
        self.xi = xi

        if kind not in ['ucb', 'ei', 'poi']:
            err = "{} has not been implemented yet, " \
                  "please use ucb, ei or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    def _ucb(self, x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        if min(std) <= 0.00001:
            return mean + 2.5 * std

        z = (mean - y_max - xi) / std
        return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi) / std
        return norm.cdf(z)

def ensure_rng(random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state

def _hashable(x):
    return tuple(map(float, x))

def lhs(dim, samples, criterion):
    X = np.zeros((samples, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, samples)) / float(2 * samples)
    for i in range(dim):
        X[:, i] = centers[np.random.permutation(samples)]

    pert = np.random.uniform(-1.0, 1.0, (samples, dim)) / float(2 * samples)
    X += pert
    return X

class Colours:
    """Print in nice colors."""

    BLUE = '\033[94m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    END = '\033[0m'
    GREEN = '\033[92m'
    PURPLE = '\033[95m'
    RED = '\033[91m'
    UNDERLINE = '\033[4m'
    YELLOW = '\033[93m'

    @classmethod
    def _wrap_colour(cls, s, colour):
        return colour + s + cls.END

    @classmethod
    def black(cls, s):
        """Wrap text in black."""
        return cls._wrap_colour(s, cls.END)

    @classmethod
    def blue(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.BLUE)

    @classmethod
    def bold(cls, s):
        """Wrap text in bold."""
        return cls._wrap_colour(s, cls.BOLD)

    @classmethod
    def cyan(cls, s):
        """Wrap text in cyan."""
        return cls._wrap_colour(s, cls.CYAN)

    @classmethod
    def darkcyan(cls, s):
        """Wrap text in darkcyan."""
        return cls._wrap_colour(s, cls.DARKCYAN)

    @classmethod
    def green(cls, s):
        """Wrap text in green."""
        return cls._wrap_colour(s, cls.GREEN)

    @classmethod
    def purple(cls, s):
        """Wrap text in purple."""
        return cls._wrap_colour(s, cls.PURPLE)

    @classmethod
    def red(cls, s):
        """Wrap text in red."""
        return cls._wrap_colour(s, cls.RED)

    @classmethod
    def underline(cls, s):
        """Wrap text in underline."""
        return cls._wrap_colour(s, cls.UNDERLINE)

    @classmethod
    def yellow(cls, s):
        """Wrap text in yellow."""
        return cls._wrap_colour(s, cls.YELLOW)

class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Empty queue")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def next(self):
        return self.__next__()

    def add(self, obj):
        self._queue.append(obj)
