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
import pathlib

import numpy as np
import yaml
from sklearn.svm import LinearSVC

logging.basicConfig(level=logging.INFO)


class WrappedLinearSVC(LinearSVC):
    # pylint: disable=too-many-instance-attributes
    r"""
    Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather
    than libsvm, so it has more flexibility in the choice of penalties and loss functions
    and should scale better to large numbers of samples.

    Cleanlab implementation expects a `predict_proba` function
    Sklearn SVC classes provide the same functionality via a `decision_function` function
    so here we wrap the classes. However, the `decision_function` provides a confidence score
    for a sample that is proportional to the signed distance of that sample to the hyperplane--
    thus, by itself it will not return probabilities in the [0,1] range; thus to normalize
    we must inject high-low values that depend on the characteristics of your data;
    tl;dr: run relabeling more than once and push values seen in logs into the `min_dist` and
    `max_dist` parameters.

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
        self.logger = multiprocessing.get_logger()
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)
        parent_dir = str(pathlib.Path(__file__).parent.parent.resolve())
        try:
            with open(f"{parent_dir}/params.yaml", "rt", encoding="utf8") as fin:
                params = yaml.safe_load(fin)
                self.min_dist = params["relabel"]["min_distance_decision"]
                self.max_dist = params["relabel"]["max_distance_decision"]
        except IOError:
            self.logger.warning("Problem reading ../params.yaml")

    def predict_proba(self, X):
        r"""Provides confidence probability measurements for the predictions;
        Note: relative values; normed between 0-1.
        We cache min max values across executions."""
        Xdata = self.decision_function(X)
        xmin = np.min(Xdata)
        xmax = np.max(Xdata)
        # This logging is not ideal, but expedient
        if self.min_dist is None or xmin < self.min_dist:
            if self.min_dist is not None:
                self.logger.info("xmin was: %f", self.min_dist)
            self.min_dist = xmin
            self.logger.info("setting min_dist: %f", xmin)
        if self.max_dist is None or xmax > self.max_dist:
            if self.max_dist is not None:
                self.logger.info("xmax was: %f", self.max_dist)
            self.max_dist = xmax
            self.logger.info("setting max_dist: %f", xmax)
        x_norm = (Xdata - self.min_dist) / (self.max_dist - self.min_dist)
        return x_norm
