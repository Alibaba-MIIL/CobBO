#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
#must set these before loading numpy:
os.environ["OMP_NUM_THREADS"] = '4' # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = '4' # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = '6' # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = '4' # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = '6' # export NUMEXPR_NUM_THREADS=6

import numpy as np
import math

import scipy.special as ss

try:
    from CobBO.kernelspace import KernelSpace
except:
    from kernelspace import KernelSpace

try:
    from bayesmark.abstract_optimizer import AbstractOptimizer
    from bayesmark.experiment import experiment_main
except ImportError as e:
    AbstractOptimizer = object


class CobBO(AbstractOptimizer):
    def __init__(self, api_config, n_iter=100, pbounds=None, init_points=0, batch=1, random_state=None,\
                 noise=False, open_slow_trust_region=True, open_fast_trust_region=True,\
                 consistent_query=None, restart=False, allow_partition=True, minimization=False):
        """Build a wrapper class to use the optimizer.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        n_iter : int
            The query budget for the experiment
        init_points : int
            The number of initial points provided
        """
        self.api_config = api_config
        self.minimization = minimization

        # Set up a proper number of initial points if init_points is wrongly configured
        if init_points <= 0:
            if n_iter > 300:
                init_points = min(int(n_iter * 0.08), 500)
            else:
                init_points = np.clip(int(n_iter * 0.10), 5, 30)

        self.init_points = init_points
        self._n_iter = n_iter

        assert batch == 1, 'Currently CobBO supports a batch of one only'
        self.batch = batch

        self.pbounds = pbounds
        self.api = api_config is not None
        if self.api:
            param_type_dict_name_range, self.round_to_values, self.equiv_point_neighbor, \
            self.logs_params, self.logits_params, self.cats_params, self.ints_params, self.cardinality = \
                CobBO._api_config_to_pbounds_and_rounding(api_config)

            self.pbounds = {}
            for k, d in param_type_dict_name_range.items():
                self.pbounds = {**self.pbounds, **d}

        self.space = KernelSpace(self.pbounds, n_iter, init_points, batch, random_state,
                                 noise, open_slow_trust_region, open_fast_trust_region,
                                 consistent_query, restart, allow_partition)


    @staticmethod
    def _api_config_to_pbounds_and_rounding(api_config):
        """Convert scikit-learn like api_config to CobBO's pbounds
        Example:
        api_config={'max_depth': {'type': 'int', 'space': 'linear', 'range': (1, 15)},
            'min_samples_split': {'type': 'real', 'space': 'logit', 'range': (0.01, 0.99)},
            'min_samples_leaf': {'type': 'real', 'space': 'logit', 'range': (0.01, 0.49)},
            'min_weight_fraction_leaf': {'type': 'real', 'space': 'logit', 'range': (0.01, 0.49)},
            'max_features': {'type': 'real', 'space': 'logit', 'range': (0.01, 0.99)},
            'min_impurity_decrease': {'type': 'real', 'space': 'linear', 'range': (0.0, 0.5)}}
        Take api_config as argument so this can be static.
        """
        # The ordering of iteration prob makes no difference, but just to be
        # safe and consistent with space.py, I will make sorted.
        param_list = sorted(api_config.keys())

        param_type_dict_name_range = {'real': {}, 'int': {}, 'bool': {}, 'cat': {}, 'ordinal': {}}
        round_to_values = {}
        equiv_point_neighbor = {}
        logits = []
        logs = []
        cats = []
        ints = []
        cardinality = 1
        for param_name in param_list:
            param_config = api_config[param_name]

            param_type = param_config["type"]
            param_space = param_config.get("space", None)
            param_range = param_config.get("range", None)
            param_values = param_config.get("values", None)

            # Setup for whitelist of values if provided:
            if (param_values is not None) and (param_type not in ("cat", "ordinal")):
                assert param_range is None
                param_values = sorted(np.unique(param_values))
                param_range = (param_values[0], param_values[-1])

            # handle different types
            if param_type in ("cat", "ordinal"):
                assert param_range is None
                assert param_values is not None
                upper = len(param_values) - 1
                low, high = 0, upper + 0.9999
                param_type_dict_name_range[param_type][param_name] = (low, high)
                cats.append(param_name)
                cardinality *= len(np.unique(param_values))

                def symbol_for_cat(x, param_values=param_values):
                    return param_values[int(math.floor(x))]

                round_to_values[param_name] = symbol_for_cat

                def equiv_point_neighbor_cat(x, upper=upper):
                    a = math.floor(x)
                    gap = x - a
                    if gap <= 0.5:
                        neighbor = a - 1 if a >= 1 else a + 1
                    else:
                        neighbor = a + 1 if a < upper else a - 1

                    equiv = a + 0.5
                    return equiv, neighbor

                equiv_point_neighbor[param_name] = equiv_point_neighbor_cat

            elif param_type == "int":
                low, high = param_range
                param_type_dict_name_range[param_type][param_name] = (low, high + 0.9999)
                ints.append(param_name)
                cardinality *= (high - low + 1)
                round_to_values[param_name] = math.floor

                def equiv_point_neighbor_int(x, low=low, high=high):
                    a = math.floor(x)
                    gap = x - a
                    if gap <= 0.5:
                        neighbor = a - 1 if a > low else a + 1
                    else:
                        neighbor = a + 1 if a < high else a - 1

                    equiv = a + 0.5
                    return equiv, neighbor

                equiv_point_neighbor[param_name] = equiv_point_neighbor_int

            elif param_type == "bool":
                assert param_range is None
                assert param_values is None
                param_type_dict_name_range[param_type][param_name] = (0, 1)
                cardinality *= 2
                round_to_values[param_name] = np.around

            elif param_type == "real":
                cardinality = np.inf
                low, high = param_range
                if param_space == "log":
                    low, high = np.log10(low), np.log10(high)
                    logs.append(param_name)
                if param_space == "logit":
                    low, high = ss.logit(low), ss.logit(high)
                    logits.append(param_name)
                param_type_dict_name_range[param_type][param_name] = (low, high)

            else:
                assert False, "type %s not handled in API" % param_type

        return param_type_dict_name_range, round_to_values, equiv_point_neighbor, logs, logits, cats, ints, cardinality

    def suggest(self, n_suggestions=1):
        """Get suggestions from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output
            Currently the algorithm is optimized for n_suggestions=1

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
        X = self.suggest_as_real_values(n_suggestions)
        X = self.convert_real_to_target_type(X)

        return X

    def suggest_as_real_values(self, n_suggestions=1):
        X = self.space.impl_suggest_kernel(n_suggestions)
        X = [dict(zip(self.space.keys, x)) for x in X]
        return X

    def convert_real_to_target_type(self, X):
        if self.api:
            for x in X:
                for param_name in self.logits_params:
                    x[param_name] = ss.expit(x[param_name])
                for param_name in self.logs_params:
                    x[param_name] = 10 ** x[param_name]
                for param_name in self.cats_params:
                    x[param_name] = self.round_to_values[param_name](x[param_name])
                for param_name in self.ints_params:
                    x[param_name] = int(x[param_name])
        return X

    def maximize(self, obj_func, optimizer, use_real_space=False):
        """Maximize a given objective function

        Parameters
        ----------
        obj_func : method
            The objective function to be optimized
        optimizer : The CobBO optimizer object

        Returns
        -------
        best_point : A dictionary
            The point with the best objective value obsereved. Each key corresponds to a parameter being optimized.
        """
        assert isinstance(optimizer, CobBO), ' A CobBO optimizer is expected'

        while optimizer.has_budget:
            if not use_real_space:
                x_probe_list = self.suggest(n_suggestions=self.batch)
                target_list = [obj_func(**x) for x in x_probe_list]
                self.observe(x_probe_list, target_list)
            else:
                x_probe_real_list = self.suggest_as_real_values(n_suggestions=self.batch)
                x_probe_list = self.convert_real_to_target_type(np.copy(x_probe_real_list))
                target_list = [obj_func(**x) for x in x_probe_list]
                self.observe(x_probe_real_list, target_list)

        return self.best_point

    def observe(self, X, y, verbose=False):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        if self.api:
            if np.isinf(y).any():
                print("y contains -inf: y=", y)
            # Convert linear to log
            for x in X:
                for param_name in self.logs_params:
                    x[param_name] = np.log10(x[param_name])

                for param_name in self.logits_params:
                    x[param_name] = ss.logit(x[param_name])

        # Minimization rather than CobBO's default maximization
        if self.minimization:
            y = [-yy for yy in y]

        try:
            _ = (_ for _ in y)
        except TypeError:
            y = [y]
            X = [X]

        self.space.last_eval_num = self.space.eval_num
        self.space.eval_num += len(X)

        # Update the model with new objective function observations
        for x_probe, target, is_rd_sample, k_indexes, util_id \
                in zip(X, y, self.space.is_rd_sample_list, self.space.k_indexes_list, self.space.util_id_list):
            self.space.observe(x_probe, target, is_rd_sample, k_indexes, util_id)

        if verbose:
            self.space.heart_beat_print(num=25)

    @property
    def has_budget(self):
        return self.space.has_unused_trial_budget()

    @property
    def best_point(self):
        return self.space.max_param


if __name__ == "__main__":
    experiment_main(CobBO)