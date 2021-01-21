import os
#must set these before loading numpy:
os.environ["OMP_NUM_THREADS"] = '4' # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '4' # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '6' # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = '4' # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = '6' # export NUMEXPR_NUM_THREADS=6

import numpy as np

try:
    from CobBO.util import Colours, Queue, ensure_rng, _hashable,assign_probability, normalize_probability
    from CobBO.util import acq_max, copula_gaussian, UtilityFunction, lhs
    from CobBO.rbf import Rbf
    from CobBO.idw import Tree
except:
    from util import Colours, Queue, ensure_rng, _hashable, assign_probability, normalize_probability
    from util import acq_max, copula_gaussian, UtilityFunction, lhs
    from rbf import Rbf
    from idw import Tree

from collections import deque
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats

from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor


import warnings
from datetime import datetime

import copy

class KernelSpace(object):
    def __init__(self, pbounds, n_iter, init_points, batch,
                 random_state=None, noise=False, open_slow_trust_region=True, open_fast_trust_region=True,
                 consistent_query=None, restart=False, allow_partition=True):

        self.random_state = ensure_rng(random_state)
        self.batch = batch

        # The total number of data points
        self._n_iter = n_iter
        self.init_points = init_points
        self.iteration = 0

        self.dim = len(pbounds)
        self.noise = noise
        self.open_slow_trust_region = open_slow_trust_region
        self.open_fast_trust_region = open_fast_trust_region
        self.consistent_query = consistent_query
        self.restart = restart
        self.restart_at_iteration = 0
        self.can_sample = True if self.dim >= 5 else False
        self.allow_partition = allow_partition

        self.multisample = 1
        self.fail_change_multisample = 0
        self.fail_change_multisample_cap = 50 if n_iter > 1000 else 25

        self.queue_new_X = Queue()
        self.goodness = None
        self.util_id = None
        self.is_rd_sample_list = None

        kappa, xi = 2.5, 0.0
        self.kappa = kappa
        self.util_explore = UtilityFunction(kind='ucb', dim=self.dim, kappa=4.0 * kappa, xi=xi)
        self.threshold_explore = 2 if self.large_trial() else 5
        util_ei = UtilityFunction(kind='ei', dim=self.dim, kappa=kappa, xi=xi)
        util_ucb_1 = UtilityFunction(kind='ucb', dim=self.dim, kappa=kappa, xi=xi)
        util_ucb_2 = UtilityFunction(kind='ucb', dim=self.dim, kappa=0.8 * kappa, xi=xi)
        util_ucb_3 = UtilityFunction(kind='ucb', dim=self.dim, kappa=1.2 * kappa, xi=xi)

        self.util_list = [util_ei, util_ucb_1, util_ucb_2, util_ucb_3]

        self.util_ind_list = range(len(self.util_list))
        self.goodness = assign_probability(len(self.util_list))

        self.queue_new_X = Queue()

        self.k_indexes_list = []
        self.util_id_list = []

        # The default optimizer is "fmin_l_bfgs_b"
        # The other GP regressor used for relatively large dimensions is defined below, optimized by adam
        def optimizer(obj_func, initial_theta, bounds):
            # * 'obj_func' is the objective function to be minimized, which
            #   takes the hyperparameters theta as parameter and an
            #   optional flag eval_gradient, which determines if the
            #   gradient is returned additionally to the function value
            # * 'initial_theta': the initial value for theta, which can be
            #   used by local optimizers
            # * 'bounds': the bounds on the values of theta

            # Use the adam optimizer
            lr, beta_1, beta_2, epsilon = 0.1, 0.9, 0.999, 1e-8
            theta = initial_theta  # initialize the vector
            m_t, v_t, t = np.zeros_like(initial_theta), np.zeros_like(initial_theta), 0

            for t in range(1, 100):
                minus_log_likelihood, g_t = obj_func(theta)  # computes the minus gradient of the stochastic function
                m_t = beta_1 * m_t + (1 - beta_1) * g_t  # updates the moving averages of the gradient
                v_t = beta_2 * v_t + (1 - beta_2) * (g_t * g_t)  # updates the moving averages of the squared gradient
                m_cap = m_t / (1 - (beta_1 ** t))  # calculates the bias-corrected estimates
                v_cap = v_t / (1 - (beta_2 ** t))  # calculates the bias-corrected estimates
                theta += lr * m_cap / (v_cap**0.5 + epsilon)  # updates the parameters
                theta = np.clip(theta, bounds[:,0], bounds[:,1])

            theta_opt, func_min = theta, minus_log_likelihood

            # Returned are the best found hyperparameters theta and the corresponding value of the target function.
            return theta_opt, func_min

        alpha = 1e-4 if not noise else 1e-3
        self.optimizer = optimizer
        self.constant_value_bounds = "fixed"
        self.length_scale_bound = (0.005, self.dim ** 0.5)
        self.k1_constant_value = 1.0
        self.k2_length_scale_bounds = np.repeat([self.length_scale_bound], self.dim, axis=0)
        self.k2_length_scale = np.ones(self.dim) * 0.5
        self._gp = GaussianProcessRegressor(
            kernel=ConstantKernel(constant_value=1.0, constant_value_bounds=self.constant_value_bounds) \
                   * Matern(nu=2.5, length_scale=0.5, length_scale_bounds=self.length_scale_bound),
            alpha=alpha,
            optimizer="fmin_l_bfgs_b",
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.random_state,
        )
        self._gp_default = True
        self._gp_restart_at_iter = 0
        self.weights = None

        self.has_improve = False
        self.big_improve = False

        self.increment_sum = 0.0
        self.increment_num = 0.0
        self.target_sum = 0.0
        self.increment_mean = 0.0
        self.target_mean = 0.0
        self.increment_exp = 0.0
        self.target_exp = 0.0

        # Get the name of the parameters
        self._keys = sorted(pbounds)

        # Initially assign uniform distribution for coordinate selection
        self._probability = assign_probability(self.dim)
        self._k_indexes = np.arange(self.dim)
        self._can_change_indexes = False

        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])],
            dtype=np.float
        )

        self._original_bounds = np.copy(self._bounds)
        self._bounds_stack = deque()
        self._bounds_stack_extra = deque()

        # Pre-allocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._targets = np.empty(shape=(0,))
        self._full_params = np.empty(shape=(0, self.dim))
        self._full_targets = np.empty(shape=(0,))
        self.num_partition_space = 0
        self.partition_base_time = 0

        # The anchor set
        self.rd_anchor_index = 0
        self.rd_anchor_set = []
        self.cycle_for_rd_anchor = 0
        self.rd_anchor_dis_P25 = 0
        self.gradient = None
        self.num_item_gp_smoothing = 0
        self.rbfx = None
        self.rbfx_bounds = None
        self.reuse_rbfx = False
        self.rbf_smooth = 0.02
        self.gp_reuse = 0

        self._success_after_adj = 0.0
        self._adj_num = 0

        self.sufficient_improvement = False
        self.very_close = False

        # Pre-allocate memory for max param and target
        self._max_param = np.empty(shape=(0, self.dim))
        self._max_target = float("-inf")
        self._anchor = self._max_param
        self._anchor_last = self._max_param
        self._global_max_param = np.empty(shape=(0, self.dim))
        self._global_max_target = float("-inf")
        self._copula_targets_max = float("-inf")
        self._copula_targets_min = float("-inf")

        # Keep track of unique points observed so far
        self._cache = {}

        # A queue for suggested data points
        self._queue = Queue()

        # A counter for the number of consecutive non-improvement trials
        self._fails = 0
        self._success = 0
        self._sub_success = 0
        self._consecutive_fails = 0
        self._slowness = 0
        self._count_success = 0
        self._modify_max_count = 0
        self._modify_max_count_3_times = 0
        self.need_init_sample_iteration = 0
        self._guard_consecutive_fails = True
        self._consecutive_local_modes = 0
        self._local_mode_now = False

        # A counter for the number of tested sub-domain
        self._tested_other_region_num = 0
        self._ref_max_num = 0
        self._time_enter_last_region = 0

        # Round-robin
        self.sequence = np.arange(self.dim)
        self.rr_ratio = 0.5
        self.is_round_robin = False
        self.ix_for_rr = 0
        self.stay = 0
        self.stay_max_base = 6 if self.dim >= 200 else\
                        5 if self.dim >= 150 else\
                        4 if self.dim >= 100 else\
                        3 if self.dim >= 70 else \
                        2 if self.dim >= 20 else\
                        1
        if self.noise:
            self.stay_max_base += min(self._n_iter//700, 7)
        else:
            self.stay_max_base += min(self._n_iter//1000, 3)
        if self.consistent_query is not None:
            self.stay_max_base = self.consistent_query

        self.how_often_do_rr = 6 * self.stay_max
        self.rr_idx = 0
        self.rr_set = [4, 5, 10, 20]
        self.rr_condition = False

        # Used for adjust_bounds
        self.bounds_index = np.random.permutation(self.dim)
        self.adjust_end = False
        self.has_reached_max_level = False

        self.coord_group = 1

        self.round = min(int(np.ceil(6/self.coord_group)), 5)
        self.max_phase = self.round * self.coord_group + 1

        self.not_close = False
        self.local_bound_ratio = 1.0

        # Threshold of v_clock
        self.reached_level = 0
        self.iter_local_cap = 0
        self.iter_in_local = 0
        self.iter_global_cap = 0
        self.iter_in_global = 0
        self.fail_threshold = 0
        self.restart_num = 0
        self.restart_threshold = 1 if self._n_iter < 10000 else 3

        # For modification by median number
        self.modify_by_median_num = 0
        self.last_time_modify = 0
        self.last_time_random_max_scratch = 0
        self.merge_random_anchor_iteration = 0
        self.done_reset_random_anchor = False
        self.iter_can_shrink_space = 0

        # For discarding data
        self.discard_data_num = 0
        self.data_num_cap = 900

        # For shrinking_ratio
        self.shrink_ratio_1, self.shrink_ratio_2, self.shrink_ratio_3 = 0.5, 0.6, 0.7

        self.threshold_cap_base = 15 if not self.small_trial() else 10
        self.threshold_cap_start = 15 if not self.small_trial() else 10

        self.can_do_rr = (self.dim >= 30) or (self.dim >= 20 and self._n_iter >= 1000)

        self.bench_level_rd_anchor = 0
        self.set_record_rd_level = False

        self._up_base = 30
        self.change_max_param_level_base = 0
        self.change_max_param_level = 0
        self.rr_condition = False

        self.in_warping_down = False
        self.time_at_max_level = 0

        self.noneRR_iter = 0

        self.k_candidate_1 = [4, 5, 6, 8] if self.dim <= 10 else \
                             [7, 8, 9, 11, 12] if self.dim <= 15 else \
                             [12, 13, 15, 18, 23, 25] if self.dim <= 35 else\
                             [20, 22, 25, 26, 27, 29, 31, 33, 36, 40]

        self.k_candidate_3 = [2, 3, 4] if self.dim <= 10 else \
                             [2, 3, 5, 6, 7] if self.dim <= 15 else \
                             [3, 4, 6, 7, 9, 10] if self.dim <= 35 else\
                             [4, 6, 7, 9, 11, 12, 14, 16]

        self.last_params = None
        self.last_targets = None
        self.last_k_indexes = None
        self.last_bounds = None
        self.last_eval_num = -1
        self.eval_num = 0

        self.rbfx = None
        self.idw_tree = None
        self.option = 0
        self.rbfx_used_num = 0
        self.rbfx_get_top_k_times = 0
        self.rbfx_use_cap = 1

        self.num_restart_space = 0.0

        self.last_suggest = None

        self.data_cluster_array = None
        self.index_partition = -1
        self.cluster_num = 2
        self.cluster_num_cap = 3 if self.dim < 20 else\
                               4 if self.dim < 60 else\
                               5
        self.cluster_num_cap = min(self.cluster_num_cap + int(self._n_iter/4000), 8)

        self.begin_time = datetime.now()
        self.debug = False

    @property
    def stay_max(self):
        stay_max = self.stay_max_base
        if self.progress() > 0.6:
            stay_max = max(stay_max-1, 1)
        return stay_max

    @property
    def upper_bound(self):
        if len(self._bounds_stack_extra) >= 1:
            return int(1.2 * self._up_base)
        else:
            return self._up_base

    def update_rr_condition(self):
        self.rr_condition = self.can_do_rr and \
                            np.mod(self.iteration, self.how_often_do_rr * self.stay_max + 1) \
                                   <= self.rr_ratio * self.how_often_do_rr * self.stay_max
        return self.rr_condition

    def has_unused_trial_budget(self):
        return self.iteration < self._n_iter

    @property
    def shrink_ratio(self):
        ratio_set = [0.5, 0.6]
        check = np.mod((self._adj_num - 1)//self.coord_group, len(ratio_set))
        return ratio_set[check]

    @property
    def threshold_cap(self):
        pi = 3.1415926 * 0.7
        return int(self.threshold_cap_start + self.threshold_cap_base * np.sin(pi * self._adj_num / self.max_phase))

    @property
    def num_round(self):
        return int(self.dim / self.dim_per_round)

    @property
    def dim_per_round(self):
        return min(self.rr_set[self.rr_idx], self.dim//2)

    def progress(self):
        return self.iteration/self._n_iter

    def impl_suggest_kernel(self, n_suggestions=1):
        while len(self.queue_new_X) < n_suggestions and self.has_unused_trial_budget():
            self.enqueue_new_X()

        X_new_list = []
        self.is_rd_sample_list = []
        self.k_indexes_list = []
        self.util_id_list = []
        i = 0
        while i < n_suggestions:
            try:
                x_probe, is_rd_sample, k_indexes, util_id = next(self.queue_new_X)
                i += 1
                X_new_list.append(x_probe)
                self.is_rd_sample_list.append(is_rd_sample)
                self.k_indexes_list.append(k_indexes)
                self.util_id_list.append(util_id)
            except StopIteration:
                break

        return X_new_list

    def enqueue_new_X(self):
        if not self.rd_sample_queue_is_empty():
            while not self.rd_sample_queue_is_empty():
                x_probe = next(self._queue)
                self.queue_new_X.add((x_probe, True, None, None))
                self.iteration += 1

        else:
            if self._fails >= 10 and np.mod(self._fails, self.threshold_explore) == 0 and not self.small_trial():
                self.util_explore.kappa = self.get_kappa()
                self.util_id = len(self.util_list)
                x_probe_list, k_indexes = self.do_suggestion(self.util_explore)
            else:
                self.util_id = np.random.choice(self.util_ind_list, p=self.goodness)
                x_probe_list, k_indexes = self.do_suggestion(self.util_list[self.util_id])

            if x_probe_list is None:
                self.enqueue_new_X()
            else:
                for x_probe in x_probe_list:
                    self.queue_new_X.add((x_probe, False, k_indexes, self.util_id))
                    self.iteration += 1

    def get_kappa(self):
        x = self._consecutive_fails/23
        upper = 3.5 if self.large_trial() else 1.0 if self.small_trial() else 2.5
        ratio = x - np.floor(x/upper) * upper
        return self.kappa * (1.0 + ratio)

    def adjust_util_goodness(self, util_id, has_improve):
        if util_id is None or util_id >= len(self.util_list):
            return
        if has_improve:
            self.goodness[util_id] *= 1.05
        else:
            self.goodness[util_id] /= 1.01
        self.goodness = normalize_probability(self.goodness)

    def opt_margin(self):
        ratio = 0.15 if len(self._bounds_stack_extra) >= 1 else 0.1
        return ratio

    def inc_for_l_g_mode(self):
        inc = 0 if self._consecutive_fails <= 20 else\
              1 if self._consecutive_fails <= 35 else\
              2 if self._consecutive_fails <= 50 else 3
        return inc

    def in_local_mode(self):
        return self._local_mode_now

    def in_local_mode_with_small_ratio(self):
        return self._local_mode_now

    def update_l_g_cap(self):
        progress = self.progress()

        if self._n_iter - self.iteration <= 1000:
            step_global = max(2 * self.stay_max, 6)
            step_local = max(2 * self.stay_max, 6)
            self.fail_threshold = max(4 * self.stay_max, 8)
        else:
            step_global = max(2 * self.stay_max, 16)
            step_local = max(2 * self.stay_max, 12)
            self.fail_threshold = max(4 * self.stay_max, 16)

        inc1 = int(0.7 * step_local * progress)
        inc2 = min( int(self._n_iter//4000), 2 ) * self.stay_max

        self.iter_local_cap = step_local + inc1 + inc2
        self.iter_global_cap = step_global + inc2

    def inc_local_iter(self):
        self.iter_in_local += 1
        if self.iter_in_local >= self.iter_local_cap and len(self) >= 50:
            self.reset_l_g_iter()
            self._local_mode_now = False

    def in_global_mode(self):
        return not self._local_mode_now

    def reset_l_g_iter(self):
        self.iter_in_global = 0
        self.iter_in_local = 0

    def inc_global_iter(self, fail_threshold):
        if self._consecutive_fails < fail_threshold:
            self._consecutive_local_modes = 0
        elif self._consecutive_fails == fail_threshold:
            self._local_mode_now = True
            self.reset_l_g_iter()
            self._consecutive_local_modes += 1
        else:
            self.iter_in_global += 1
            if self.iter_in_global >= self.iter_global_cap:
                self.reset_l_g_iter()
                self._local_mode_now = True
                self._consecutive_local_modes += 1

    def shrink_total_space(self, ratio=0.8):
        print(Colours.blue("shrink_total_space"))
        if self._adj_num >= 1 or len(self._bounds_stack_extra) >= 1 or self.index_partition >= 0:
            self.reset_space_to_full_data_and_bounds(merge_partition=True)

        self._max_param = np.copy(self._global_max_param)
        self._max_target = self._global_max_target
        self._anchor_last = self._anchor
        self._anchor = self._max_param

        self._bounds = self.do_shrink_bounds(ratio=ratio)
        self.filter_data_by_bounds()
        self._full_params = self._params
        self._full_targets = self._targets
        self._original_bounds = np.copy(self._bounds)

    def fast_trust_region(self):
        if not self.open_fast_trust_region or self._anchor is None or len(self._anchor) == 0:
            return

        if self.with_random_anchor():
            if len(self._bounds_stack_extra) >= 1:
                self.pop_halve_bounds()
            return

        # Apply fast trust region
        self.update_l_g_cap()

        if self.in_local_mode():
            # For local_mode
            self.inc_local_iter()
            if len(self._bounds_stack_extra) == 0:
                self.push_halve_bounds(ratio=self.extra_bound_ratio())
        else:
            # For global-mode
            self.inc_global_iter(fail_threshold=self.fail_threshold)

            if len(self._bounds_stack_extra) >= 1:
                self.pop_halve_bounds()

        return

    def extra_bound_ratio(self):
        progress = self.progress()
        factor = 1.0 - 0.5*progress

        ratio_set = [0.7, 0.5]
        index = np.mod(self._consecutive_local_modes-1, len(ratio_set))
        ratio = ratio_set[index]

        return ratio * factor

    def half_rr_ratio(self):
        self.rr_ratio /= 2.0

    def inc_probability(self, k_indexes, can_improve, delta):
        if k_indexes is None or len(k_indexes) == self.dim:
            return
        delta /= self.stay_max
        if can_improve:
            ratio = 1.0 if self.dim <= 250 else 2.0
            self._probability[k_indexes] *= (1.0 + ratio * delta)
        else:
            self._probability[k_indexes] *= 1/(1.0 + 4.0*delta)
        self._probability = normalize_probability(self._probability)

    def select_k_indexes_round_robin(self):
        ix = self.ix_for_rr
        if self.in_local_mode():
            if ix < self.num_round - 2:
                self._k_indexes = sorted(self.sequence[ix * self.dim_per_round: (ix + 2) * self.dim_per_round])
            else:
                self._k_indexes = sorted(self.sequence[(ix - 1) * self.dim_per_round: self.dim])
        else:
            if ix < self.num_round - 1:
                self._k_indexes = sorted(self.sequence[ix * self.dim_per_round: (ix + 1) * self.dim_per_round])
            else:
                self._k_indexes = sorted(self.sequence[ix * self.dim_per_round: self.dim])

        self.ix_for_rr = np.mod(self.ix_for_rr + 1, self.num_round)
        if self.ix_for_rr == 0:
            try:
                gradient = self.rbfx.gradient(self.in_unit_cube(self._max_param, range(self.dim)))
                self.sequence = np.argsort(abs(gradient))
            except AttributeError:
                if self.debug:
                    print("warning: select_k_indexes_round_robin: self.rbfx.gradient is None")
                pass
            self.rr_idx = np.mod(self.rr_idx + 1, len(self.rr_set))

        return self._k_indexes

    def select_k_indexes_weight_update(self, k):
        if self.dim <= 50:
            if k >= max(0.2 * float(self.dim), 4) and np.random.random() <= 0.7:
                self._k_indexes = sorted(np.random.choice(self.dim, k, False, self._probability))
            else:
                if np.random.random() < 0.9:
                    # Descending order
                    self._k_indexes = sorted(np.argsort(-self._probability)[0:k])
                else:
                    self._k_indexes = sorted(np.argsort(self._probability)[0:k])
        else:  # High dim
            if (k > 20 and np.random.random() <= 0.7) or (k >= 3 and np.random.random() <= 0.25):
                self._k_indexes = sorted(np.random.choice(self.dim, k, False, self._probability))
            else:
                if np.random.random() < 0.9:
                    # Descending order
                    self._k_indexes = sorted(np.argsort(-self._probability)[0:k])
                else:
                    self._k_indexes = sorted(np.argsort(self._probability)[0:k])

        self.inc_probability(self._k_indexes, False, delta=0.1)

        return self._k_indexes

    def estimate_epsilon(self, data_points, enough_points, margin_ratio):
        # Default epsilon is "the average distance between nodes" based on a bounding hypercube
        xi = np.asarray([np.asarray(a, dtype=np.float_).flatten() for a in data_points])
        if len(xi.shape) == 1:
            xi = xi[None, :]
        ximax = np.amax(xi, axis=1)
        ximin = np.amin(xi, axis=1)
        edges = ximax - ximin
        edges = edges[edges > 0.01]
        ratio = 1.0 if self.iteration <= 0.2 * self._n_iter else\
                0.95 if self.iteration <= 0.5 * self._n_iter else 0.9
        epsilon = ratio*np.power(np.prod(edges)/len(data_points), 1.0/edges.size)
        if not enough_points and self.in_local_mode_with_small_ratio():
            epsilon *= (1.0 - margin_ratio)
        epsilon = max(epsilon, 0.0001)
        progress = 1.0 - 0.1*self.progress()
        epsilon *= progress
        return epsilon

    def select_k_indexes_gp_smoothing(self, k):
        if self.rbfx is None:
            return self.select_k_indexes_gp_smoothing_2(k)
        gradient = self.rbfx.gradient(self.in_unit_cube(self._anchor, range(self.dim)))
        self._k_indexes = np.argpartition(abs(gradient), -k)[-k:]
        return self._k_indexes

    def select_k_indexes_gp_smoothing_2(self, k):
        if not np.array_equal(self._anchor, self._anchor_last) \
                or self.gradient is None\
                or abs(self.num_item_gp_smoothing-len(self._params)) >= 2*self.stay_max:
            vectors = self.in_unit_cube(self._params, range(self.dim)) \
                      - self.in_unit_cube(self._anchor, range(self.dim))
            distances = np.linalg.norm(vectors, ord=0.5, axis=1)
            md = np.percentile(distances, 35)
            sub_set = np.where(distances < md)
            distances = distances[sub_set]
            vectors = vectors[sub_set]
            sigma = max(np.percentile(distances, 40), 0.01)
            targets = self._targets[sub_set]
            weight = np.exp(-(distances/sigma)**2)
            value_diff = (self._max_target - targets)/(1.0+abs(self._max_target))
            weight = np.multiply(value_diff, weight)
            gradient = np.zeros_like(self._params[0])
            for u_x, f_x in zip(vectors, weight):
                gradient += (f_x * u_x)

            self.gradient = gradient
            self.num_item_gp_smoothing = len(self._params)

        else:
            gradient = self.gradient

        if sum(abs(gradient)) == 0.0:
            self._k_indexes = self.select_k_indexes_weight_update(k)

        else:
            self._k_indexes = np.argpartition(abs(gradient), -k)[-k:]
        return self._k_indexes


    def select_k_indexes(self, k):

        if self.stay < self.stay_max - 1:
            if k <= 30 and len(self._k_indexes) <= 30\
                    or (self.in_local_mode() or self._adj_num > 0.25 * self.max_phase)\
                    and len(self) <= 50:
                return self._k_indexes
        else:
            self.stay = 0

        if self._can_change_indexes or len(self._k_indexes) > 30 or k > 30:
            num = 4 if self.large_trial() else 2
            if self._consecutive_fails <= num:
                return self.select_k_indexes_gp_smoothing(k)

            elif (self.reuse_rbfx and self.gp_reuse == 0 or\
                       not self.reuse_rbfx and np.random.random() < 0.25):
                if self.reuse_rbfx:
                    self.gp_reuse += 1
                else:
                    self.gp_reuse = 0
                return self.select_k_indexes_gp_smoothing(k)
            else:
                if not self.reuse_rbfx:
                    self.gp_reuse = 0

                if self.is_round_robin:
                    # Round-robin mode
                    return self.select_k_indexes_round_robin()
                else:
                    # Not in round-robin
                    return self.select_k_indexes_weight_update(k)

        return self._k_indexes


    def register(self, params, target, is_rd_sample, k_indexes):

        x = self._as_array(params)
        if x in self:
            raise KeyError('Data point {} is not unique'.format(x))

        self._cache[_hashable(x.ravel())] = target


        if not is_rd_sample:

            sub_success_threshod = 4
            self.big_improve = False
            if self._max_target < target - 0.1 * np.absolute(target):  #[0.1, inf] improve
                self.inc_probability(k_indexes, True, delta=0.3)
                self._fails = 0
                self._success += 1
                self._sub_success = 0
                self._can_change_indexes = False
                self.big_improve = True
            elif self._max_target < target - 0.04 * np.absolute(target):  #[0.04, 0.1] improve
                self.inc_probability(k_indexes, True, delta=0.2)
                self._fails = int(self._fails * 0.1)
                self._success += 1
                self._sub_success += 1
                if self._sub_success >= sub_success_threshod:
                    self._can_change_indexes = True
                    self._success = 0
                    self._sub_success = 0
                else:
                    self._can_change_indexes = False
            elif self._max_target < target - 0.01 * np.absolute(target): #[0.01, 0.04] improve
                self.inc_probability(k_indexes, True, delta=0.1)
                self._success += 1
                self._sub_success += 1
                self._fails = int(self._fails * 0.4)
                if self._sub_success >= sub_success_threshod-1:
                    self._can_change_indexes = True
                    self._success = 0
                    self._sub_success = 0
                else:
                    self._can_change_indexes = False
            elif self._max_target < target:                            #[0, 0.01] improve
                self.inc_probability(k_indexes, True, delta=0.05)
                self._sub_success += 1
                self._fails = int(self._fails * 0.7)-1 if self._fails >= 2 else 0
                if self._sub_success >= sub_success_threshod - 1:
                    self._can_change_indexes = True
                    self._success = 0
                    self._sub_success = 0
                else:
                    self._can_change_indexes = False
            else:                                                      # Fail to improve
                self.inc_probability(k_indexes, False, delta=0.2)
                if not self.with_random_anchor():
                    self._fails += 1
                self._success = 0
                self._sub_success = 0
                self._can_change_indexes = True

        if self._max_target < target:
            self.has_improve = True
            self._count_success += 1
            self.fail_change_multisample = 0
            self._consecutive_fails = 0
            self._modify_max_count = 0
            self._modify_max_count_3_times = 0
            if target > self._max_target + 0.01 * abs(target):
                self._slowness = 0
            elif target > self._max_target + 0.005 * abs(target):
                self._slowness = int(self._slowness * 0.5)
            elif target > self._max_target + 0.001 * abs(target):
                self._slowness = int(self._slowness * 0.4)
            else:
                self._slowness = int(self._slowness * 0.3)

            self.rd_anchor_index = len(self.rd_anchor_set)

            if not is_rd_sample:
                this_improvement = target - self._max_target
                self.increment_sum += this_improvement
                self.increment_num += 1.0
                self.target_sum += abs(target)

                self.target_mean = self.target_sum / self.increment_num
                self.target_exp = abs(target) if self.target_exp == 0.0 else\
                                  0.4 * abs(target) + 0.6 * self.target_exp

                self.increment_mean = self.increment_sum/self.increment_num
                self.increment_exp = this_improvement if self.increment_exp == 0.0 else\
                                     0.4 * this_improvement + 0.6 * self.increment_exp
            else:
                this_improvement = 0

            self._max_target = target
            self._max_param = x
            if self._max_target > self._global_max_target:
                self._global_max_target = self._max_target
                self._global_max_param = np.copy(self._max_param)

            self.print_out()
            self.not_close = not self.is_data_close_to_anchor_param(x, ratio=0.3)
            progress = self.progress()
            if self._n_iter >= 1000:
                very_close_ratio = np.random.uniform(0.15, 0.2) if progress <= 0.3 else\
                                   0.15 if progress <= 0.5 else\
                                   0.10 if progress <= 0.7 else 0.05
            else:
                very_close_ratio = 0.12 if progress <= 0.3 else\
                                   0.10 if progress <= 0.5 else\
                                   0.08 if progress <= 0.7 else 0.06
            self.very_close = self.is_data_close_to_anchor_param(x, ratio=very_close_ratio)

            fraction = 1e-4 * (1.0 - 0.6 * self.iteration / self._n_iter)
            self.sufficient_improvement = self.has_sufficient_improvement(this_improvement, fraction)

            if self.sufficient_improvement:
                self._success_after_adj += 1.0
            else:
                self._success_after_adj += 0.5

            if self.iteration <= self.init_points + self.restart_at_iteration\
                  or self.sufficient_improvement or self.very_close:
                self._anchor_last = self._anchor
                self._anchor = self._max_param
                if self.not_close:
                    self.move_center_by_anchor_param()

        else: # no improvement
            if self.iteration > self.init_points + self.need_init_sample_iteration:
                self._consecutive_fails += 1
                self._modify_max_count += 1
                self._modify_max_count_3_times += 1
                self._slowness += 1
                self.fail_change_multisample += 1

            self.not_close = False
            self.has_improve = False
            self.sufficient_improvement = False
            self.very_close = True

            if self._consecutive_fails >= 10 and (not np.array_equal(self._anchor, self._max_param)): #5, 25
                self._anchor_last = self._anchor
                self._anchor = self._max_param
        #end if-else

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._targets = np.concatenate([self._targets, [target]])
        if self._params is not self._full_params:
            self._full_params = np.concatenate([self._full_params, np.copy(x).reshape(1, -1)])
            self._full_targets = np.concatenate([self._full_targets, [target]])

        if self._adj_num > self.reached_level:
            self.reached_level = self._adj_num

        return self.has_improve

    def num_success_for_pop(self):
        num = 3 if self._adj_num <= 2 else \
              2 if self._adj_num <= 3 else 1
        if self.sufficient_improvement and (not self.very_close):
            num = max(num-1, 1)
        return num

    def observe(self, X, y, is_rd_sample, k_indexes, util_id):
        X = self._as_array(X)

        if np.isinf(X).any() or np.isnan(X).any() or np.isinf(y).any() or np.isnan(y).any():
            return

        self.has_improve = False
        if X not in self:
            self.has_improve = self.register(X, y, is_rd_sample, k_indexes)

        if not is_rd_sample:
            if self.has_improve:
                self.adjust_util_goodness(util_id, self.has_improve)
            self.append_new_to_last_subspace(X, y)

        return self.has_improve

    def random_sample(self, num):
        points = np.random.uniform(0, 1, (num, self.dim))
        points = self._bounds[:,0] + (self._bounds[:,1] - self._bounds[:,0]) * points
        return points

    def random_sample_avoid_max_region(self, num, anchor=None):
        points = self.lhs_sample(num)
        if anchor is None:
            anchor = self._max_param

        for col, (lower, upper) in enumerate(self._bounds):
            anchor_col = anchor[col]
            points_col = points[:, col]
            interval = 0.15 * (upper-lower)
            for ix, point in enumerate(points_col):
                if anchor_col - interval > lower and anchor_col + interval < upper:
                    if point < anchor_col:
                        ratio = (anchor_col - interval - lower) / (anchor_col - lower)
                        points_col[ix] = lower + (point - lower) * ratio
                    else:
                        ratio = (upper - anchor_col - interval) / (upper - anchor_col)
                        points_col[ix] = upper - (upper - point) * ratio
                elif anchor_col - interval < lower:
                    points_col[ix] = upper - (upper - point) * 0.7
                else:
                    points_col[ix] = lower + (point - lower) * 0.7
        return points

    def lhs_sample(self, init_points):
        try:
            points = lhs(self.dim, samples=init_points, criterion='maximin') #'centermaximin'
        except:
            points = np.random.random((init_points, self.dim))

        points = self._bounds[:, 0] + (self._bounds[:, 1] - self._bounds[:, 0]) * points

        return points

    def lhs_sample_center(self, init_points):
        try:
            samples = lhs(self.dim, samples=init_points, criterion='maximin')
        except:
            samples = np.random.random((init_points, self.dim))
        points = np.empty((init_points, self.dim))
        anchor_array = self._anchor
        for col, (lower, upper) in enumerate(self._bounds):
            anchor = anchor_array[col]
            interval = 0.3*(upper-lower)
            if anchor - interval >= lower and anchor + interval <= upper:
                newlow, newupper = anchor - interval, anchor + interval
            elif anchor - interval < lower:
                newlow, newupper = lower, lower + 2.0*interval
            else:
                newlow, newupper = upper - 2.0*interval, upper
            points[:, col] = newlow + (newupper-newlow)*samples[:, col]
        return points

    def lhs_sample_within_domain(self, init_points):
        samples = lhs(self.dim, samples=init_points, criterion='maximin')

        points = np.empty((init_points, self.dim))
        for col, (lower, upper) in enumerate(self._bounds):
            newlow = 0.9*lower + 0.1*upper
            newupper = 0.9*upper + 0.1*lower
            points[:, col] = newlow + (newupper-newlow)*samples[:, col]
        return points

    def max(self):
        try:
            res = {
                'target': self.target.max(),
                'params': dict(
                    zip(self.keys, self.params[self.target.argmax()])
                )
            }
        except ValueError:
            res = {}
        return res

    def res(self):
        params = [dict(zip(self.keys, p)) for p in self.params]

        return [
            {"target": target, "params": param}
            for target, param in zip(self.target, params)
        ]

    def set_bounds(self, new_bounds):
        for row, key in enumerate(self.keys):
            if key in new_bounds:
                self._bounds[row] = new_bounds[key]

    @staticmethod
    def is_in_domain(data, bounds):
        for row, bound in enumerate(bounds):
            if data[row] < bound[0] or data[row] > bound[1]:
                return False
        return True

    def reset_param_target(self):
        self._params = self._full_params
        self._targets = self._full_targets

    def reset_bounds(self):
        self._bounds = np.copy(self._original_bounds)
        self.clear_extra_bound_stack()

    def reset_bounds_param_target(self):
        self.reset_bounds()
        self.reset_param_target()

    def filter_data_by_bounds(self, ratio=0.6):
        total = len(self)
        if total <= 100:
            return

        ind = np.array(list(map(lambda x:self.is_in_domain(x, self._bounds), self._params)))
        params = self._params[ind]
        targets = self._targets[ind]

        params_2 = self._params[~ind]
        params_2p = np.clip(params_2, self.bounds[:,0], self.bounds[:,1])
        distance_2 = np.linalg.norm(self.in_unit_cube(params_2, range(self.dim))\
                                    - self.in_unit_cube(params_2p, range(self.dim)), axis=1)
        targets_2 = self._targets[~ind]

        top = int(total * ratio) - len(targets)

        if top >= 1:
            ind_top = np.argpartition(distance_2, top)[:top]
            params_2 = params_2[ind_top]
            targets_2 = targets_2[ind_top]
            params = np.concatenate([params, params_2])
            targets = np.concatenate([targets, targets_2])

        if len(params) >= 100:
            self._params, self._targets = params, targets

    def clear_extra_bound_stack(self):
        self.iter_in_local = 0
        self.iter_in_global = 0
        self._bounds_stack_extra.clear()

    def move_center_by_anchor_param(self):
        if self._adj_num > 0:
            if len(self._bounds_stack_extra) > 0:
                self._bounds_stack_extra.clear()

            num = self._adj_num
            last_coord_group = False
            if num == self.max_phase:
                last_coord_group = True
                num -= 1
            self._bounds_stack.clear()
            self._adj_num = 0
            self.reset_bounds_param_target()
            for _ in np.arange(num):
                self._bounds_stack.append(np.copy(self._bounds))
                self.help_adjust_bounds()
            if last_coord_group:
                self._bounds_stack.append(np.copy(self._bounds))
                self.reduce_last_coord_group_bounds()

            self.filter_data_by_bounds()

        else:
            if len(self._bounds_stack_extra) > 0:
                self.pop_halve_bounds()

    def deque_bounds(self):
        self._bounds_stack_extra.clear()

        self._bounds = self._bounds_stack.pop()
        self._params = self._full_params
        self._targets = self._full_targets
        self.filter_data_by_bounds()

        self._success_after_adj = 0.0
        self._adj_num -= 1
        self.has_reached_max_level = False
        print("pop--: level=", self._adj_num)

    def pop_halve_bounds(self):
        self._params = self._full_params
        self._targets = self._full_targets
        self._bounds = self._bounds_stack_extra.pop()
        self.filter_data_by_bounds()

    def push_halve_bounds(self, ratio=0.5):
        self._bounds_stack_extra.append(np.copy(self._bounds))
        self._bounds = self.do_shrink_bounds(ratio=ratio)
        self.filter_data_by_bounds()

    def clear_bounds_stack(self):
        self._bounds_stack.clear()
        self.clear_extra_bound_stack()
        self._adj_num = 0
        self.adjust_end = True
        self.has_reached_max_level = False

    def cal_length_scale(self, params):
        weights = np.ones(self.dim)
        for row in range(self.dim):
            low, up = self._bounds[row][0], self._bounds[row][1]
            points = params[:, row]
            if len(params) > 4:
                left = np.percentile(points, 20)
                right = np.percentile(points, 80)
            else:
                left, right = low, up
            if low < up and right > left + 1e-3:
                weights[row] = max((right-left)/(up-low), 0.1)
            else:
                weights[row] = 1.0
        weights_max = max(weights)
        weights = np.clip(weights, 0.7*weights_max, None)
        weights = weights / weights_max
        return weights

    def help_adjust_bounds(self):
        ratio = self.shrink_ratio
        self._adj_num += 1
        coord_group = self.coord_group
        weights = self.cal_length_scale(self._params)
        if self.adjust_end:
            if self.rbfx is not None:
                gradient = self.rbfx.gradient(self.in_unit_cube(self._max_param, range(self.dim)))
                self.bounds_index = np.argsort(abs(gradient))
            self.adjust_end = False
        mapping = self.bounds_index
        ix = np.mod(self._adj_num-1, coord_group)
        jump = int(np.ceil(self.dim/coord_group))
        for row in np.arange(ix*jump, min(self.dim, (ix+1)*jump)):
            bound = self._bounds[mapping[row]]
            low = bound[0]
            up = bound[1]
            c = (up + low) / 2.0
            interval = (up - low) * ratio * weights[mapping[row]]
            point = self._anchor[mapping[row]]
            if point < c:
                if point - low >= interval/2.0:
                    self._bounds[mapping[row]][0] = point - interval / 2.0
                    self._bounds[mapping[row]][1] = point + interval / 2.0
                else:
                    self._bounds[mapping[row]][1] = low + interval
            else:
                if up - point >= interval/2.0:
                    self._bounds[mapping[row]][0] = point - interval / 2.0
                    self._bounds[mapping[row]][1] = point + interval / 2.0
                else:
                    self._bounds[mapping[row]][0] = up - interval

    def do_shrink_bounds(self, ratio=0.5, bounds=None, anchor_sub=None):
        weights = self.cal_length_scale(self._params)

        if bounds is None:
            bounds = self._bounds
            anchor_sub = self._anchor

        for row in range(len(bounds)):
            bound = bounds[row]
            low = bound[0]
            up = bound[1]
            c = (up + low) / 2.0
            interval = (up - low) * ratio * weights[row]
            point = anchor_sub[row]
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

    def reduce_last_coord_group_bounds(self):
        self._adj_num += 1
        mapping = self.bounds_index
        weights = self.cal_length_scale(self._params)
        for row in np.arange(self.dim):
            bound = self._bounds[mapping[row]]
            low = bound[0]
            up = bound[1]
            c = (up + low) / 2.0
            interval = (up - low) * 0.5 * weights[mapping[row]]
            point = self._anchor[mapping[row]]
            if point < c:
                if point - low >= interval / 2.0:
                    self._bounds[mapping[row]][0] = point - interval / 2.0
                    self._bounds[mapping[row]][1] = point + interval / 2.0
                else:
                    self._bounds[mapping[row]][1] = low + interval
            else:
                if up - point >= interval / 2.0:
                    self._bounds[mapping[row]][0] = point - interval / 2.0
                    self._bounds[mapping[row]][1] = point + interval / 2.0
                else:
                    self._bounds[mapping[row]][0] = up - interval

    def adjust_bounds(self):
        if len(self._bounds_stack_extra) > 0:
            self._bounds = self._bounds_stack_extra.pop()
        self._bounds_stack.append(np.copy(self._bounds))
        self.clear_extra_bound_stack()

        if self._adj_num < self.max_phase-1:
            self.help_adjust_bounds()
        else:
            self.reduce_last_coord_group_bounds()

        self.filter_data_by_bounds()
        if self.adjust_success():
            self._success_after_adj = 0.0
            print("push++: level=", self._adj_num)

        else:
            self.deque_bounds()

    def adjust_success(self):
        return True if len(self._targets) >= 1 else False

    def is_data_close_to_anchor_param(self, data, ratio=0.2):
        if self._anchor is None or self._anchor.size == 0:
            return False
        for row, bound in enumerate(self._bounds):
            point = data[row]
            pivot = self._anchor[row]
            interval = (bound[1]-bound[0])*ratio*0.5
            if pivot <= bound[0]+interval and point > bound[0]+interval*2.0:
                return False
            elif pivot >= bound[1]-interval and point < bound[1]-interval*2.0:
                return False
            elif point < pivot-interval or point > pivot+interval:
                return False
        return True

    def large_distance(self, x, y, ratio=0.4):
        if len(x) == 0 or len(y) == 0:
            return True

        for row, bound in enumerate(self._bounds):
            xx = x[row]
            yy = y[row]
            threshold = (bound[1]-bound[0])*ratio
            d = abs(xx-yy)
            if d > threshold:
                return True
        return False

    def is_on_center(self, ratio=0.5):
        if self._adj_num == 0:
            return True
        for row, bound in enumerate(self._bounds):
            interval = (bound[1] - bound[0]) * ratio * 0.5
            c = (bound[1] + bound[0]) / 2.0
            left = c - interval
            right = c + interval
            point = self._anchor[row]
            if point < left or point > right:
                return False
        return True

    def add_random_points(self, ini_ratio=0.5, add_num=0, option=0, anchor=None):
        if add_num <= 0:
            num = int( ini_ratio * min(self.init_points, int((self._n_iter - self.iteration)), 250) )
        else:
            num = add_num

        if num == 0 or self.iteration + num >= self._n_iter - 3:
            return

        if option == 0: # latin cube minmax
            points = self.lhs_sample(num)
        elif option == 1: # latin cube around anchor half-domain
            points = self.lhs_sample_center(num)
        elif option == 2: # pure random in domain
            points = self.random_sample(num)
        else: #option == 3: # avoid max region
            points = self.random_sample_avoid_max_region(num, anchor)

        for point in points:
            self._queue.add(point)

    def has_sufficient_improvement(self, this_improvement, fraction):
        return this_improvement > fraction * self.target_exp

    def handle_insufficient_improvement(self):
        if self._max_target > self._global_max_target:
            self._global_max_param = np.copy(self._max_param)
            self._global_max_target = np.copy(self._max_target)

        self.reset_space_to_full_data_and_bounds()

        # Clustering, find the furthest cluster
        centers, values, _ = self.k_means(self._full_params, self._full_targets, cluster_num=5)
        distances = np.linalg.norm(centers - self._anchor, axis=1)
        indexes = np.argsort(values)
        if distances[indexes[1]] >= distances[indexes[2]]:
            index = 1
        else:
            index = 2
        self._max_target = values[index]
        self._anchor_last = self._anchor
        self._anchor = centers[index]
        self._max_param = self._anchor

        self.last_time_random_max_scratch = self.iteration
        self.restart_num += 1

    def set_up_config(self):
        # Setup parameters
        self._adj_num = 0
        self._fails = 0
        self._consecutive_fails = 0
        self._probability = np.ones(self.dim) / float(self.dim)
        self._success_after_adj = 0.0
        self._bounds_stack.clear()
        self.clear_extra_bound_stack()
        self.has_reached_max_level = False

    def reset_space_to_full_data_and_bounds(self, merge_partition=False):
        self.clear_bounds_stack()
        self.set_up_config()

        if merge_partition and self.data_cluster_array is not None and self.index_partition > 0:
            self._full_params, self._full_targets = self.data_cluster_array[0]
            if self.cluster_num >= 1:
                self._full_params = np.concatenate((self._full_params, self.data_cluster_array[1][0]))
                self._full_targets = np.concatenate((self._full_targets, self.data_cluster_array[1][1]))

        self.reset_bounds_param_target()

    def with_random_anchor(self):
        return self.iteration < self.merge_random_anchor_iteration

    def prepare_random_anchor_far(self):
        old_index = self.rd_anchor_index
        self.rd_anchor_index += 1
        while self.rd_anchor_index < len(self.rd_anchor_set) and \
             not self.large_distance(self.rd_anchor_set[self.rd_anchor_index], \
                                    self.rd_anchor_set[old_index], ratio=0.5) and\
                np.linalg.norm(self.rd_anchor_set[self.rd_anchor_index] - self.rd_anchor_set[old_index]) \
                     < self.rd_anchor_dis_P25:
            self.rd_anchor_index += 1

        if self.rd_anchor_index < len(self.rd_anchor_set):
            return self.rd_anchor_set[self.rd_anchor_index]

        fraction = 0.25 if len(self) >= 500 else 0.5
        top_k = min( int(len(self._full_params) * fraction), 250)
        ind_top = np.argpartition(self._full_targets, -top_k)[-top_k:]
        params = self._full_params[ind_top]

        point = self._max_param

        distance = np.linalg.norm(self.in_unit_cube(params, range(self.dim)) \
                       - self.in_unit_cube(point, range(self.dim)), axis=1)
        chosen_index = np.argsort(-distance)[:int(top_k*0.6)]

        params = params[chosen_index]
        targets = self._full_targets[ind_top][chosen_index]

        lower = np.percentile(targets, 20)
        ind = np.where(targets >= lower)
        params = params[ind]
        targets = targets[ind]

        m = min(int(len(targets) * 0.2), 15)
        if m <= len(params) - 5:
            label = KMeans(n_clusters=m, random_state=None).fit(params).labels_
            new_params = np.empty(shape=(0, self.dim))
            for i in np.arange(m):
                tmp_bool = label == i
                ixs = np.argmax(targets[tmp_bool])
                new_params = np.concatenate([new_params, params[tmp_bool][ixs].reshape(1, -1)])
        else:
            new_params = params

        self.rd_anchor_dis_P25 = np.percentile(distance[chosen_index][ind], 40)  # find P40 for negative values
        self.rd_anchor_set = new_params

        self.rd_anchor_index = np.random.randint( max( int(0.5*len(new_params)), 1 ) )
        return self.rd_anchor_set[self.rd_anchor_index]

    def prepare_random_anchor_near(self):
        top_k = min( int(len(self._full_params) * 0.1), 80)
        ind_top = np.argpartition(self._full_targets, -top_k)[-top_k:]

        if len(ind_top) > 20:
            ix = np.random.choice(len(ind_top), 15, replace=False)
            point = self._max_param
            distance = np.linalg.norm(self._full_params[ind_top][ix] - point, axis=1)
            ixx = np.argmax(distance)
            return self._full_params[ind_top][ix][ixx]
        else:
            ixx = np.random.randint(len(ind_top))
            return self._full_params[ind_top][ixx]

    def sweep_partitions(self):
        self.sweep_partitions_k_means()
        while 1 <= len(self._full_params) <= max(0.01 * self.iteration, 5):
            self.sweep_partitions_k_means()

        self.clear_bounds_stack()
        self.reset_bounds_param_target()
        self.set_up_config()

        if len(self) >= 1:
            index = np.argmax(self._targets)
            self._max_param = self._params[index]
            self._anchor = self._max_param
            self._max_target = self._targets[index]
        else:
            self._max_param = None
            self._max_target = float("-inf")
            self._anchor = self._max_param
            self._anchor_last = self._max_param

        self.restart_at_iteration = self.iteration
        self.rbf_smooth = 0.02
        self._consecutive_fails = 0
        self._modify_max_count = 0
        self._modify_max_count_3_times = 0
        self._slowness = 0
        self.num_restart_space += 1.0
        self.partition_base_time = 0

    def sweep_partitions_k_means(self):
        if self.index_partition < 0 or self.data_cluster_array is None:
            self.help_sweep_partition()
            self.index_partition = 1

        elif self.index_partition == self.cluster_num - 1:
            self.data_cluster_array[self.index_partition] = (self._full_params, self._full_targets)
            self._full_params, self._full_targets = self.data_cluster_array[0]
            for i in range(1,  self.cluster_num):
                if len(self.data_cluster_array[i][0]) > 0.02 * self.iteration:
                    self._full_params = np.concatenate((self._full_params, self.data_cluster_array[i][0]))
                    self._full_targets = np.concatenate((self._full_targets, self.data_cluster_array[i][1]))
            self.cluster_num = min(self.cluster_num + 1, self.cluster_num_cap)
            self.help_sweep_partition()
            self.index_partition = 0

        else:
            self.data_cluster_array[self.index_partition] = (self._full_params, self._full_targets)
            self.index_partition += 1

        self._full_params, self._full_targets = self.data_cluster_array[self.index_partition]

    def help_sweep_partition(self):
        self.data_cluster_array = [None] * self.cluster_num

        if self.progress() < 0.5 and self.cluster_num < min(self.cluster_num_cap, 4):
            if self.cluster_num >= 3:
                center_params, center_targets, data_clusters = \
                    self.k_means(self._full_params, self._full_targets, self.cluster_num-1, concate=True)
                sorted_indexes = np.argsort(-center_targets)  # largest target first
                for i in range(self.cluster_num-1):
                    self.data_cluster_array[i] = data_clusters[_hashable(center_params[sorted_indexes[i]])]
            elif self.cluster_num == 2:
                self.data_cluster_array[0] = (self._full_params, self._full_targets)

            self.data_cluster_array[self.cluster_num-1] = (np.empty(shape=(0, self.dim)), np.empty(shape=(0,)))

        else:
            center_params, center_targets, data_clusters = \
                self.k_means(self._full_params, self._full_targets, self.cluster_num, concate=True)
            sorted_indexes = np.argsort(-center_targets)  # largest target first
            for i in range(self.cluster_num):
                self.data_cluster_array[i] = data_clusters[_hashable(center_params[sorted_indexes[i]])]

    def set_random_anchor(self):
        multi = self.stay_max if not self.small_trial() else min(self.stay_max, 2)
        cycle = 5 * multi

        if np.mod(self.iteration, cycle) < cycle-1 and self.with_random_anchor():
            return

        self.cycle_for_rd_anchor += 1
        if np.mod(self.cycle_for_rd_anchor, 6) <= 1:
            self._anchor_last = self._anchor
            self._anchor = self.prepare_random_anchor_far()
        else:
            self._anchor_last = self._anchor
            self._anchor = self.prepare_random_anchor_near()

        if not self.with_random_anchor():
            self.reduce_domain_for_random_anchor(ratio=0.8)
            self.rd_anchor_index += 1

        self.merge_random_anchor_iteration = self.iteration + cycle + 1
        self.done_reset_random_anchor = False

    def reduce_domain_for_random_anchor(self, ratio=0.8):
        self.reset_bounds()
        self._bounds = self.do_shrink_bounds(ratio=ratio)

        indexes = np.array(list(map(lambda x:self.is_in_domain(x, self._bounds), self._full_params)))
        self._params = self._full_params[indexes]
        self._targets = self._full_targets[indexes]

    def reset_random_anchor(self):
        self._anchor_last = self._anchor
        self._anchor = self._max_param
        self.move_center_by_anchor_param()
        self.done_reset_random_anchor = True

    def k_means(self, params, targets, cluster_num, concate=True, normalize=True, use_centers=False):
        """the output new_params are sorted in descending order in new_targets"""
        if normalize:
            n_params = self.in_unit_cube(params, range(self.dim))
        else:
            n_params = np.copy(params)

        # Apply PCA for dimension reduction
        reduc_dim = min(5, self.dim)
        if reduc_dim < self.dim:
            pca = PCA(n_components=reduc_dim)
            pca.fit(n_params)
            reduc_n_params = pca.transform(n_params)
        else:
            reduc_n_params = n_params

        # Add values into consideration
        if concate:
            balance = 0.4
            copula_targets = copula_gaussian(copy.deepcopy(targets))
            reduc_n_params = np.concatenate((reduc_n_params, balance*copula_targets[:, None]), axis=1)

        # Apply k_means, use more cluster_num and then merge them
        if 6*cluster_num < 0.5*len(reduc_n_params):
            new_cluster_num = 6*cluster_num
        else:
            new_cluster_num = cluster_num
        kmeans_result = KMeans(n_clusters=new_cluster_num, random_state=None).fit(reduc_n_params)
        label = kmeans_result.labels_
        centers = kmeans_result.cluster_centers_

        new_params = np.empty(shape=(0, self.dim))
        new_targets = np.empty(shape=(0,))
        data_clusters = {}
        for i in np.arange(new_cluster_num):
            tmp_bool = label == i
            ixs = np.argmax(targets[tmp_bool])
            point = params[tmp_bool][ixs]
            value = targets[tmp_bool][ixs]
            if use_centers:
                point = 0.2*point + 0.8*centers[i]
            new_params = np.concatenate([new_params, point.reshape(1, -1)])
            new_targets = np.concatenate([new_targets, [value]])
            data_clusters[_hashable(params[tmp_bool][ixs])] = (params[tmp_bool], targets[tmp_bool])

        sorted_indexes = np.argsort(-new_targets)  # largest target first
        new_params = new_params[sorted_indexes]
        new_targets = new_targets[sorted_indexes]

        #merge clusters if necessaary
        if cluster_num < new_cluster_num:
            new_data_clusters = {}
            merge_params = new_params[:cluster_num]
            size_dict_merge_params = np.array([len(data_clusters[_hashable(x)][0]) for x in merge_params])
            for i in np.arange(cluster_num, new_cluster_num):
                threshold = 1.8*len(params)/cluster_num
                subset_merge_params = np.arange(cluster_num)[size_dict_merge_params < threshold]
                distances = np.linalg.norm(merge_params[subset_merge_params] - new_params[i], axis=1)
                closest = np.argmin(distances)
                size_dict_merge_params[subset_merge_params][closest] +=\
                                 len(data_clusters[_hashable(new_params[i])][0])
                data_clusters[_hashable(merge_params[subset_merge_params][closest])] = \
                    ( np.concatenate([ data_clusters[_hashable(merge_params[subset_merge_params][closest])][0],\
                                       data_clusters[_hashable(new_params[i])][0] ]),
                      np.concatenate([ data_clusters[_hashable(merge_params[subset_merge_params][closest])][1], \
                                       data_clusters[_hashable(new_params[i])][1]]) )
            for item in merge_params:
                new_data_clusters[_hashable(item)] = data_clusters[_hashable(item)]

            new_params = merge_params
            new_targets = new_targets[:cluster_num]
            data_clusters = new_data_clusters

        return new_params, new_targets, data_clusters

    def help_discard_data_k_means(self, params, targets, keep_ratio, top_ratio=0.1, keep_top=False, modify_top=True):
        top_k = int(len(targets) * top_ratio)  # 0.15 | 0.2
        if top_k == 0:
            ind_low = range(len(targets))
            ind_top = []
        else:
            two_parts = np.argpartition(targets, -top_k)
            ind_top = two_parts[-top_k:]
            ind_low = two_parts[0:(len(targets) - top_k)]

        if keep_ratio < 1.0:
            m = int(len(ind_low) * keep_ratio)
            new_params, new_targets, _ = self.k_means(params[ind_low], targets[ind_low], m)
        else:
            new_params = params[ind_low]
            new_targets = targets[ind_low]

        if modify_top and len(ind_top) >= 1:
            index = np.argmax(targets[ind_top])
            value = np.mean(targets[ind_low])
            new_params = np.concatenate([new_params, [params[ind_top][index]]])
            new_targets = np.concatenate([new_targets, [value]])

        if keep_top and len(ind_top) >= 1:
            # Add the saved params
            new_params = np.concatenate([new_params, params[ind_top]])
            new_targets = np.concatenate([new_targets, targets[ind_top]])

        return new_params, new_targets

    def discard_data(self, keep_ratio=0.4, top_ratio=0.1, keep_top=False, modify_top=True):
        self.discard_data_num += 1
        self.reset_space_to_full_data_and_bounds()
        keep_ratio += 0.2 * self.progress()
        self.discard_data_k_means(keep_ratio=keep_ratio, top_ratio=top_ratio, keep_top=keep_top, modify_top=modify_top)

    def discard_data_k_means(self,keep_ratio,top_ratio=0.1,keep_top=False, modify_top=True):
        print(Colours.cyan('iter={}: discard fraction={} data from len={}'.format(self.iteration, \
                                    1.0-keep_ratio, len(self._full_targets))))

        self._full_params, self._full_targets = \
            self.help_discard_data_k_means(self._full_params, self._full_targets,\
                              keep_ratio, top_ratio, keep_top, modify_top=modify_top)

        index = np.argmax(self._full_targets)
        self._anchor = self._full_params[index]
        self._max_param = self._full_params[index]
        self._max_target = self._full_targets[index]

        self.reset_param_target()
        self.filter_data_by_bounds()

    def help_suggest_reset_random_anchor(self):
        if not self.with_random_anchor() and not self.done_reset_random_anchor:
            self.reset_random_anchor()

    def help_suggest_may_set_random_anchor(self):

        if not self.with_random_anchor() and not self.done_reset_random_anchor:
            self.reset_random_anchor()

        trigger_warping_level = 80 if self.large_trial() else 30

        if self._consecutive_fails < trigger_warping_level:
            self.set_record_rd_level = False

        elif self._consecutive_fails >= trigger_warping_level:
            if not self.set_record_rd_level:
                self.set_record_rd_level = True
                self.bench_level_rd_anchor = self._consecutive_fails

            step = 30*self.stay_max
            ratio = 0.2 + 0.1 * min(self._consecutive_fails/300, 1.0)
            rd = int(step*ratio)
            count = np.mod(self._consecutive_fails - self.bench_level_rd_anchor, step)

            if self.progress() <= 0.9 and count <= rd and not self.with_random_anchor():
                self.set_random_anchor()

    def help_suggest_add_points_if_less_than_level(self):

        add_rd_level = 20 if self.large_trial() else np.clip(self._n_iter//100, 4, 10)
        if len(self.target) <= add_rd_level and\
                self.iteration >= self.init_points:
            if self._anchor is not None and len(self._anchor) > 0:
                self.add_random_points(add_num=add_rd_level, option=1)
            else:
                self.add_random_points(add_num=add_rd_level, option=0)
            return True
        else:
            return False

    def help_est_filter_data_by_margin_dec(self, ratio):
        data_points = self._full_params
        targets = self._full_targets

        if len(self._bounds_stack_extra) == 0 and self._adj_num == 0:
            return self.params, self.target

        modified_data = np.empty(shape=(0, self.dim))
        modified_target = np.empty(shape=(0,))
        bounds = np.copy(self._bounds)

        for row in range(self.dim):
            low, up = bounds[row][0], bounds[row][1]
            margin = (up - low) * ratio
            bounds[row][0] -= margin
            bounds[row][1] += margin

        for row, data in enumerate(data_points):
            if self.is_in_domain(data, bounds):
                modified_data = np.concatenate([modified_data, data.reshape(1, -1)])
                modified_target = np.concatenate([modified_target, [targets[row]]])

        return modified_data, modified_target

    def if_few_points_filter_by_margin(self, margin_ratio):
        cut_for_enough_points = 100
        if len(self.target) >= cut_for_enough_points:
            target = self.target
            data_points = self.params
            enough_points = True
        else:
            data_points, target = self.help_est_filter_data_by_margin_dec(ratio=margin_ratio)
            enough_points = False
        return data_points, target, enough_points

    def is_initial_sampling(self):
        num = self.init_points + min(int(0.5*self.init_points), 100)
        already = len(self)
        if already > num:
            return False
        if self.iteration <= self.restart_at_iteration + num - already:
            self.is_round_robin = False
            return True
        else:
            return False

    def help_suggest_get_k_for_rr_or_normal(self):
        self.rr_condition = self.update_rr_condition()

        # Round-robin
        if self.rr_condition:
            self.is_round_robin = True
            k = self.dim_per_round

        else:  # Pure exploration none-RR mode
            k = self.help_suggest_select_non_RR_mode_k_value()
            self.is_round_robin = False
            self.noneRR_iter += 1
            if self._consecutive_fails > 10 and np.mod(self.noneRR_iter, 5) == 0:
                k = min(self.dim, self.upper_bound)

        return k

    def help_suggest_select_non_RR_mode_k_value(self):

        p = 0.5 + 0.1 * self.progress() + 0.1 * (self._adj_num / self.max_phase)

        if self.in_local_mode() \
                or np.random.random() < p:
            k_candidate_1 = self.k_candidate_1
            ix1 = np.random.randint(len(k_candidate_1))
            k = k_candidate_1[ix1]
        else:
            k_candidate_3 = self.k_candidate_3
            ix3 = np.random.randint(len(k_candidate_3))
            k = k_candidate_3[ix3]

        k = min(k, self.dim)

        return k

    def slow_trust_region(self):
        if self.with_random_anchor():
            return

        cap = self.threshold_cap
        threshold = np.clip(int(self._n_iter / 8), 8, cap)
        if self.dim <= 25 and self._n_iter <= 300:
            threshold = max(threshold // 2, 8)
        begin_level = int(threshold * 0.25)
        trigger_level = int(threshold * 0.3)
        max_phase = self.max_phase
        if self._fails >= begin_level + threshold and self._adj_num < max_phase:
            if self._fails >= begin_level + threshold + trigger_level and not self.in_local_mode():
                self.adjust_bounds()
                self._fails = int(self.threshold_cap * 0.25)
                if self._adj_num == max_phase:
                    self.has_reached_max_level = True
                    self.time_at_max_level = self.iteration
        elif self._adj_num >= 1 and \
                (self._success_after_adj >= self.num_success_for_pop() or \
                 (self._fails < begin_level and np.mod(self._fails, 10) == 9)):
            self.deque_bounds()
            if self._adj_num > 0.5 * max_phase:
                self.deque_bounds()
            if self._adj_num > 0.75 * max_phase:
                self.deque_bounds()
            if self._adj_num == 0:
                self.adjust_end = True
        elif self.has_reached_max_level and self.iteration - self.time_at_max_level >= threshold:

            if self.progress() > 0.7:
                self.reset_space_to_full_data_and_bounds(merge_partition=True)
            else:
                self.handle_insufficient_improvement()

            self.threshold_cap_base = min(self.threshold_cap_base + 5, 40)
            self.threshold_cap_start += 3

    def help_suggest_use_small_k_or_dim_for_fails_num_just_above_threshold(self, k):
        cap = self.threshold_cap
        threshold = np.clip(int(self._n_iter / 8), 8, cap)
        if self.dim <= 25 and self._n_iter <= 300:
            threshold = max(threshold // 2, 8)
        begin_level = int(threshold * 0.25)
        trigger_level = int(threshold * 0.4)
        if self._fails >= begin_level + threshold:
            if self._fails < begin_level + threshold + trigger_level or self.in_local_mode():
                k = 1 if np.mod(self._fails - begin_level - threshold, 6) < 3 else self.dim
        return k

    def get_top_half_points(self, params, targets):
        top_k = int(len(targets)*0.5)

        if top_k < 5:
            return params, targets

        ind = np.argpartition(targets, -top_k)[-top_k:]
        targets = targets[ind]
        params = params[ind]
        return params, targets

    def queue_init_X(self, init_points):
        if self._queue.empty and self.empty:
            init_points = max(init_points, 5)

        if self.restart_at_iteration == 0:
            first_init_points = int(0.7*init_points)
            samples = self.lhs_sample(first_init_points)
            for point in samples:
                if point not in self:
                    self._queue.add(point)

            second_init_points = init_points - first_init_points
            samples2 = self.lhs_sample_within_domain(second_init_points)
            for point in samples2:
                if point not in self:
                    self._queue.add(point)

        else:
            samples = self.random_sample_avoid_max_region(num=init_points, anchor=self._global_max_param)
            for point in samples:
                if point not in self:
                    self._queue.add(point)

    def rd_sample_queue_is_empty(self):
        return self._queue.empty

    def modify_max_if_many_fails(self):
        cap = 50 if self.large_trial() else 25

        if self._modify_max_count_3_times > 3*cap:
            if self.debug:
                print(Colours.cyan("modify multi-points around max at iteration="), self.iteration)
            bounds = np.copy(self._bounds)
            ratio = min(0.01 ** (1 / self.dim), 0.3)
            bounds = self.do_shrink_bounds(ratio=ratio, bounds=bounds, anchor_sub=self._anchor)
            ind = np.array(list(map(lambda x: self.is_in_domain(x, bounds), self._params)))
            dec = max(self._targets) - np.percentile(self._targets, 80)
            self._targets[ind] -= dec
            self._modify_max_count_3_times = 0

        elif self._modify_max_count > cap:
            if self.debug:
                print(Colours.cyan("modify_max_for_many_fails at iteration="), self.iteration)
            ind2 = np.argmax(self._targets)
            v = np.percentile(self._targets, 80)
            self._targets[ind2] = v
        else:
            return

        index = np.argmax(self._targets)
        self._max_param = self._params[index]
        self._anchor_last = self._anchor
        self._anchor = self._max_param
        self._modify_max_count = 0

    def modify_multi_points_around_max(self, ratio):
        self.reset_bounds()
        next_anchor = self.prepare_random_anchor_far()

        self._anchor = self._max_param
        self._bounds = self.do_shrink_bounds(ratio=ratio)

        ind = np.array(list(map(lambda x: self.is_in_domain(x, self._bounds), self._full_params)))
        dec = max(self._full_targets) - np.percentile(self._full_targets, 40)
        self._full_targets[ind] -= dec

        self.reset_bounds_param_target()

        if len(self) > 600:
            self.discard_data(keep_ratio=0.5, top_ratio=0.2, keep_top=True, modify_top=False)

        self._max_target = np.mean(self._targets)
        self._anchor_last = self._max_param
        self._anchor = next_anchor
        self._max_param = self._anchor

    def _gp_before_fit(self, k_indexes):
        params = self._gp.kernel.get_params()
        if self._gp_default:
            params['k1__constant_value'] = self.k1_constant_value
            params['k1__constant_value_bounds'] = "fixed"
            params['k2__length_scale'] = stats.gmean(self.k2_length_scale[k_indexes])
            params['k2__length_scale_bounds'] = self.length_scale_bound
            n = len(k_indexes)
            r_num = 6 if n < 5 else\
                    5 if n < 10 else\
                    4 if n < 15 else\
                    3 if n < 20 else\
                    2 if n < 25 else\
                    1
            self._gp.n_restarts_optimizer = r_num
            self._gp.optimizer = "fmin_l_bfgs_b"
        else:
            params['k1__constant_value'] = self.k1_constant_value
            params['k1__constant_value_bounds'] = self.constant_value_bounds
            params['k2__length_scale'] = self.k2_length_scale[k_indexes]
            params['k2__length_scale_bounds'] = self.k2_length_scale_bounds[k_indexes]
            self._gp.n_restarts_optimizer = 0
            self._gp.optimizer = self.optimizer

        self._gp.kernel.set_params(**params)

    def _gp_after_fit(self, k_indexes):
        scales = self._gp.kernel_.get_params()['k2__length_scale']
        if 0.25 < scales.all()/stats.gmean(self.k2_length_scale[k_indexes]) < 4.0:
            self.k2_length_scale[k_indexes] = \
                np.clip(0.9 * self.k2_length_scale[k_indexes] + 0.1 * scales,\
                    self.length_scale_bound[0], \
                    self.length_scale_bound[1])
            self.k1_constant_value = self._gp.kernel_.get_params()['k1__constant_value']

    def suggest_by_cob(self, utility_function):

        if self.debug:
            self.begin_time = datetime.now()

        params, targets, enough_points = self.params, self.target, True

        targets = copula_gaussian(targets)

        k_indexes = np.arange(self.dim)
        if self.can_sample:
            if self.is_initial_sampling():
                k = min(self.dim, self.upper_bound)
            else:
                k = self.help_suggest_get_k_for_rr_or_normal()
                k = self.help_suggest_use_small_k_or_dim_for_fails_num_just_above_threshold(k)

            if k < self.dim:
                k_indexes = self.select_k_indexes(k)

                if self.debug:
                    self.begin_time = datetime.now()

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    params, targets = self.esti_virt_points_on_subspace(params, targets, \
                                                      k_indexes, self._anchor, enough_points)

                    if self.debug:
                        print("\n step 2: esti_virt_points_on_subspace time=",
                              (datetime.now() - self.begin_time).total_seconds())
                        self.begin_time = datetime.now()

            else:
                self._k_indexes = k_indexes

            self.stay += 1

        if params is None or len(params) <= 1 or np.isnan(params).any() or np.isinf(params).any():
            if self._anchor is not None and len(self._anchor) > 0:
                self.add_random_points(add_num=3, option=1)
            else:
                self.add_random_points(add_num=3, option=0)
            return None, None

        params = self.in_unit_cube(params, k_indexes)

        if len(k_indexes) <= 20 or np.mod(self._consecutive_fails, 20) < 10 and len(k_indexes) <= 30:
            self._gp_default = True
        else:
            self._gp_default = False

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp_before_fit(k_indexes)
            try:
                self._gp.fit(params, targets)
            except ValueError:
                self.add_random_points(add_num=1, option=0)
                return None, None
            self._gp_after_fit(k_indexes)

        if self.debug:
            print("step 3: gp.fit time=", (datetime.now() - self.begin_time).total_seconds())
            self.begin_time = datetime.now()

        self.top_sample = 1
        progress = self.progress()
        if len(k_indexes) < 5:
            if np.mod(self._consecutive_fails, 20) < 10:
                self.multisample = 1
            else:
                self.multisample = 2
        else:
            if self._consecutive_fails > 20:
                if self.large_trial():
                    self.multisample = 4
                    self.top_sample = 5 + int(self._consecutive_fails/15) if self.dim > 20 else\
                                      3 + int(self._consecutive_fails/20)
                else:
                    self.multisample = 3
            else:
                if self._consecutive_fails < 10:
                    self.multisample = 1
                elif len(self) > 400 or not self._gp_default:
                    if self.large_trial() and progress < 0.9:
                        self.multisample = 4
                        self.top_sample = 4 if self.dim > 20 else 2
                    else:
                        self.multisample = 3
                else:
                    if len(k_indexes) > 25 or self._consecutive_fails >= 15:
                        self.multisample = 1
                    else:
                        self.multisample = 2

        bounds = (self.bounds[k_indexes]-self._original_bounds[k_indexes, 0][:, None])\
                 /(self._original_bounds[k_indexes, 1][:, None] - self._original_bounds[k_indexes, 0][:, None])

        suggestion_subindex_list = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            multisample=self.multisample,
            y_max=self._max_target,
            x_max=self._anchor[k_indexes],
            bounds=bounds,
            random_state=self.random_state,
            top_sample=self.top_sample
        )

        suggestion_subindex_list_copy = np.array([self.out_unit_cube(x, k_indexes) for x in suggestion_subindex_list])

        if self.debug:
            print("step 4: max acquisition, multisample=", self.multisample,\
                  "len(k_index)=", len(k_indexes), "time=", (datetime.now() - self.begin_time).total_seconds())
            self.begin_time = datetime.now()

        if self.can_sample:
            suggestion_list = np.empty((len(suggestion_subindex_list_copy), self.dim))
            for i in range(len(suggestion_list)):
                suggestion_list[i][k_indexes] = suggestion_subindex_list_copy[i]
                a = np.arange(self.dim)
                mask = np.zeros(a.shape, dtype=bool)
                mask[k_indexes] = True
                excluded = a[~mask]
                suggestion_list[i][excluded] = self._anchor[excluded]
        else:
            suggestion_list = suggestion_subindex_list_copy

        a = np.arange(len(suggestion_list))
        mask = np.zeros(a.shape, dtype=bool)
        for i in a:
            if suggestion_list[i] not in self:
                mask[i] = True

        return suggestion_list[mask], k_indexes

    def coordi_bo(self, utility_function):
        if self._anchor is not None and not self.is_in_domain(self._anchor, self.bounds):
            self.move_center_by_anchor_param()

        if self.help_suggest_add_points_if_less_than_level():
            return None, None

        return self.suggest_by_cob(utility_function)

    def too_many_consecutive_fails_condition(self):
        if self.progress() > 0.75:
            if self.data_cluster_array is None or \
                    self.data_cluster_array is not None and self.index_partition == 0:
                return False
            else:
                return True

        if self.dim > 5:
            cap = 150 if self.large_trial() else 70
            cap = min(250, int((1.0 + 0.1 * self.num_restart_space) * cap))
            if self.dim <= 20:
                cap = int(1.9*cap)
        else:
            cap = 80 if self.large_trial() else 50 if not self.small_trial() else 25

        return self._slowness >= min(0.15*self._n_iter, cap)

    def enough_trials_for_one_partition(self):
        if self.data_cluster_array is None \
              or self.index_partition == 0 \
              or self._global_max_target == self._max_target:
            return False
        else:
            cap = 160 if self.large_trial() else 80
            if self.big_improve:
                self.partition_base_time += 5
            return self.iteration - self.restart_at_iteration > \
                         cap/(1.5**self.index_partition) + self.partition_base_time\
                   and self._consecutive_fails >= 10

    def do_suggestion(self, utility_function):
        """New points to evaluate"""

        if len(self._full_params) == 0:
            self.need_init_sample_iteration = self.iteration
            self.queue_init_X(self.init_points)
            return None, None

        if self.allow_partition and \
                (self.too_many_consecutive_fails_condition() or self.enough_trials_for_one_partition()):
            self.sweep_partitions()
            print(Colours.yellow("Repulsive escape at iteration="), self.iteration, \
                  'in partition index=', self.index_partition, "with", len(self),\
                  'points from', self.cluster_num, "partitions")
            return None, None

        # Shrink the original_bounds at the last stage of trials
        progress = self.progress()
        if progress > 0.75 and self.iteration >= self.iter_can_shrink_space:
            ratio = 0.8-(progress-0.75)*1.5
            self.shrink_total_space(ratio=ratio)
            self.iter_can_shrink_space = self.iteration + max(int(0.1 * self._n_iter), 25)

        # Filter out data points if too many are in the current space
        if (len(self) > self.data_num_cap or len(self._full_targets) > 2.0*self.data_num_cap)\
                and self._consecutive_fails >= 10 and not self.with_random_anchor():
            self.discard_data(keep_ratio=0.5, top_ratio=0.2, keep_top=True, modify_top=False)

        self.help_suggest_may_set_random_anchor()

        if not self.with_random_anchor():
            self.slow_trust_region()
            self.modify_max_if_many_fails()

        self.fast_trust_region()
        suggest_list, k_indexes = self.coordi_bo(utility_function)

        if suggest_list is not None and len(suggest_list) > 0:
            self.last_suggest = suggest_list[0]

        return suggest_list, k_indexes

    def append_new_to_last_subspace(self, new_point_sub, new_value):
        if self.batch == 1 and self.last_params is not None:
            self.last_params = np.concatenate([self.last_params, new_point_sub.reshape(1, -1)])
            self.last_targets = np.concatenate([self.last_targets, [new_value]])

    def esti_virt_points_on_subspace(self, data_points, targets, k_indexes, anchor, enough_points):
        self.update_reuse_rbfx(data_points, targets, enough_points, 0.0)

        if self.debug:
            print("update_reuse_rbfx time=", (datetime.now() - self.begin_time).total_seconds())
            self.begin_time = datetime.now()

        select_data_points, same_data_index = self.project_to_subspace(data_points, k_indexes, anchor)

        if self.debug:
            print("project_to_subspace time=", (datetime.now() - self.begin_time).total_seconds())
            self.begin_time = datetime.now()

        if len(select_data_points) == 0 and len(same_data_index) >= 1:
            return data_points[same_data_index][:, k_indexes], targets[same_data_index]

        project_data_points = self.in_unit_cube(select_data_points, range(self.dim))

        if self.option == 0:
            new_target = self.rbfx(*project_data_points.T)

            if self.debug:
                print("rbfx(*project_data_points.T) len(data)=", len(project_data_points), "time=", (datetime.now() - self.begin_time).total_seconds())
                self.begin_time = datetime.now()

            targets_max = max(targets)
            if max(new_target) > targets_max:
                self.rbf_smooth += 0.02
                if self.debug:
                    print("local-new_target rbfx exceeds UPPER BOUND", \
                           'max(new_target)=', max(new_target), 'copula_max=', targets_max)
        else: # option == 1:
            new_target = self.idw_tree(project_data_points)

        new_data_points = select_data_points[:, k_indexes]
        if len(same_data_index) > 0:
            np.concatenate([new_data_points, data_points[same_data_index][:, k_indexes]])
            np.concatenate([new_target, targets[same_data_index]])

        return new_data_points, new_target

    def project_to_subspace(self, data_points, k_indexes, anchor):
        a = np.arange(self.dim)
        mask = np.zeros(a.shape, dtype=bool)
        mask[k_indexes] = True
        excluded = a[~mask]

        project_data = []
        project_distance = []
        same_data_index = []
        tmp_map = set()
        for id, x in enumerate(data_points):
            y = np.copy(x)
            if np.all(y[excluded] == anchor[excluded]):
                same_data_index.append(id)
            else:
                y[excluded] = anchor[excluded]
                yy = _hashable(y)
                if yy not in tmp_map:
                    tmp_map.add(yy)
                    project_data.append(y)
                    project_distance.append(np.linalg.norm(x-y))

        project_data, same_data_index = np.array(project_data), np.array(same_data_index)

        if np.mod(self._consecutive_fails, 60) < 30:
            return project_data, same_data_index

        else:
            project_distance = np.array(project_distance)

            if len(same_data_index) > 100 or len(project_data) <= 1:
                return [], same_data_index
            else:
                top = np.clip(min(len(project_data)-1, int(0.5 * len(same_data_index))), 1, 300)
                ind_top = np.argpartition(project_distance, top)[:top]
                project_data = project_data[ind_top]
                return project_data, same_data_index

    def bool_reuse_rbfx(self):
        condition = ( (self.rbfx_used_num < self.rbfx_use_cap and \
                        np.array_equal(self.last_bounds, self._bounds))\
                      or self.last_eval_num == self.eval_num )\
                    and self.rbfx is not None
        return condition

    def update_reuse_rbfx(self, data, target, enough_points, margin_ratio):

        if not self.bool_reuse_rbfx():
            self.rbfx_used_num = 0
            self.reuse_rbfx = False
        else:
            self.reuse_rbfx = True
            self.rbfx_used_num += 1

        if not self.reuse_rbfx:
            data = self.in_unit_cube(data, range(self.dim))
            rbfx_last = self.rbfx
            rbfx_bounds_last = self.rbfx_bounds
            if not self.large_trial() or self.option == 1 and not self.reuse_rbfx:
                try:
                    epsilon = self.estimate_epsilon(data, enough_points, margin_ratio)
                    self.rbfx = Rbf(*data.T, target, epsilon=epsilon, smooth=self.rbf_smooth, norm=lambda X, Y: self.norm_p(X, Y))
                    self.rbfx_bounds = np.copy(self.bounds)
                    self.option = 0
                except:
                    self.option = 1
                    self.idw_tree = Tree(data, target)
                    self.rbfx = rbfx_last
                    self.rbfx_bounds = rbfx_bounds_last
            else:
                self.option = 1
                self.idw_tree = Tree(data, target)
                self.rbfx = rbfx_last
                self.rbfx_bounds = rbfx_bounds_last

    def norm_p(self, X, Y):
        if X.shape != Y.shape:
            raise ValueError("Array lengths must be equal")
        p = 2
        distance = sum(abs(X-Y) ** p) ** (1.0 / p)
        return distance

    def params_to_array(self, params):
        try:
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x):
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return dict(zip(self.keys, x))

    def _as_array(self, x):
        try:
            x = np.asarray(x, dtype=float)
        except TypeError:
            x = self.params_to_array(x)

        x = x.ravel()
        try:
            assert x.size == self.dim
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return x

    def in_unit_cube(self, data_points, k_indexes):
        lb, ub = self._original_bounds[k_indexes, 0], self._original_bounds[k_indexes, 1]
        return (data_points - lb) / (ub - lb)

    def out_unit_cube(self, data_points, k_indexes):
        lb, ub = self._original_bounds[k_indexes, 0], self._original_bounds[k_indexes, 1]
        return data_points * (ub - lb) + lb

    def __contains__(self, x):
        return _hashable(x) in self._cache

    def __len__(self):
        assert len(self._params) == len(self._targets)
        return len(self._targets)

    @property
    def empty(self):
        return len(self) == 0

    @property
    def k_indexes(self):
        return self._k_indexes

    @k_indexes.setter
    def k_indexes(self, k_indexes):
        self._k_indexes = k_indexes

    @property
    def params(self):
        return self._params

    @property
    def max_param(self):
        return self._max_param

    @property
    def max_target(self):
        return self._max_target

    @property
    def target(self):
        return self._targets

    @property
    def keys(self):
        return self._keys

    @property
    def bounds(self):
        return self._bounds

    def enough_trial(self):
        return self._n_iter - self.iteration > 300

    def large_trial(self):
        return self._n_iter > 2000

    def small_trial(self):
        return self._n_iter <= 200

    def budget(self):
        return self._n_iter - self.iteration

    def heart_beat_print(self, num=25):
        if np.mod(self.eval_num, num) == 0:
            output = "| {} | {} | {} |k={}, len(self)={}, "\
                    "rand_anchor={}, in_trust_rg={}, multisample={},"\
                     "default_gp={}, rbf_option={}".format(self.eval_num,\
                    self._global_max_target, self._max_target,\
                    len(self._k_indexes),  len(self), self.with_random_anchor(),\
                    self.in_local_mode(), self.multisample, self._gp_default,\
                    self.option)
            print(output)

    def print_out(self):
        output = "| {} | {} | {} | k={}, len(self)={}, " \
                 "rand_anchor={}, in_trust_rg={}, multisample={}, default_gp={}, rbf_option={}".format(self.eval_num,\
                self._global_max_target,\
                self._max_target, len(self._k_indexes), len(self._targets),\
                self.with_random_anchor(), self.in_local_mode(), self.multisample, self._gp_default,\
                self.option)
        if self._tested_other_region_num == 0:
            print(Colours.purple(output))
        elif self._tested_other_region_num == 1:
            print(Colours.blue(output))
        elif self._tested_other_region_num == 2:
            print(Colours.green(output))
        elif self._tested_other_region_num == 3:
            print(Colours.darkcyan(output))
        elif self._tested_other_region_num == 4:
            print(Colours.yellow(output))
        else:
            print(Colours.red(output))
