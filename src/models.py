# Copyright 2022 Todd Cook
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""`models.py` - model classes adapted for cleanlab."""
import logging
import multiprocessing

import numpy as np
from sklearn.svm import SVC, LinearSVC

logging.basicConfig(level=logging.INFO)


class WrappedSVC(SVC):  # Inherits sklearn base classifier
    """
    Cleanlab implementation expects a `predict_proba` function
    Sklearn SVC classes provide the same functionality via a `decision_function` function
    so here we wrap the classes
    """

    min_dist = None
    max_dist = None
    logger = None

    # pylint: disable=too-many-locals
    def __init__(
        self,
        *,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=True,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
        min_dist=None,
        max_dist=None,
    ):
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.logger = multiprocessing.get_logger()
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)

    def predict_proba(self, X):
        r"""Provides confidence probability measurements for the predictions;
        Note: relative values; normed between 0-1.
        We cache min max values across executions."""
        Xdata = self.decision_function(X)
        xmin = np.min(Xdata)
        xmax = np.max(Xdata)
        # This logging is not ideal, but expedient
        if self.min_dist is None or xmin < self.min_dist:
            self.min_dist = xmin
            self.logger.info("setting min_dist: %f", xmin)
        if self.max_dist is None or xmax > self.max_dist:
            self.max_dist = xmax
            self.logger.info("setting max_dist: %f", xmax)
        x_norm = (Xdata - self.min_dist) / (self.max_dist - self.min_dist)
        return x_norm


class WrappedLinearSVC(LinearSVC):
    # pylint: disable=too-many-instance-attributes
    """
    Cleanlab implementation expects a `predict_proba` function
    Sklearn SVC classes provide the same functionality via a `decision_function` function
    so here we wrap the classes
    """
    min_dist = None
    max_dist = None

    # pylint: disable=too-many-locals
    def __init__(
        self,
        probability=True,
        penalty="l2",
        loss="squared_hinge",
        *,
        dual=True,
        tol=1e-4,
        C=1.0,
        multi_class="ovr",
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        verbose=0,
        random_state=None,
        max_iter=1000,
        min_dist=None,
        max_dist=None,
    ):
        self.probability = probability
        self.dual = dual
        self.tol = tol
        self.C = C
        self.multi_class = multi_class
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.max_iter = max_iter
        self.penalty = penalty
        self.loss = loss
        super().__init__(
            dual=self.dual,
            tol=self.tol,
            C=self.C,
            multi_class=self.multi_class,
            fit_intercept=self.fit_intercept,
            intercept_scaling=self.intercept_scaling,
            class_weight=self.class_weight,
            verbose=self.verbose,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.logger = multiprocessing.get_logger()
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)

    def predict_proba(self, X):
        r"""Provides confidence probability measurements for the predictions;
        Note: relative values; normed between 0-1.
        We cache min max values across executions."""
        Xdata = self.decision_function(X)
        xmin = np.min(Xdata)
        xmax = np.max(Xdata)
        # This logging is not ideal, but expedient
        if self.min_dist is None or xmin < self.min_dist:
            self.min_dist = xmin
            self.logger.info("setting min_dist: %f", xmin)
        if self.max_dist is None or xmax > self.max_dist:
            self.max_dist = xmax
            self.logger.info("setting max_dist: %f", xmax)
        x_norm = (Xdata - self.min_dist) / (self.max_dist - self.min_dist)
        return x_norm
