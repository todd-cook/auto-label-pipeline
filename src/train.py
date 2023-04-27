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
"""`train.py` - Train model"""
import json
import pickle
from glob import glob
from typing import List

import numpy as np
import pandas as pd
import yaml
from numpy import ndarray
from sklearn.decomposition import PCA
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm

from utils import fix_path, multiclass_confusion_matrix_metrics, seed_everything


def encode_ys(
    yvals: ndarray, label_enc: LabelEncoder, default_val: str = "UNK"
) -> List[str]:
    """
    Encode y values, if label not in encoder, fall back to 'UNK'
    :param yvals:
    :param label_enc:
    :param default_val:
    :return:
    """
    classes = set(label_enc.classes_.tolist())
    results = []
    for val in yvals:
        if val not in classes:
            results.append(default_val)
        else:
            results.append(label_enc.transform([val])[0])
    return results


# pylint: disable=too-many-locals,too-many-statements
def work() -> None:
    """
    Fetch params, split data, train model, evaluate
    :return: None
    """
    # Fetch processing params
    with open(fix_path("../params.yaml"), "rt", encoding="utf-8") as fd:
        params = yaml.safe_load(fd)
    # Note: we share some parameters with previous stages
    SEED = params["train"]["seed"]
    SPLIT = params["train"]["split"]
    SVM_REGULARIZATION_C = params["train"]["svm_regularization_C"]
    SVM_PENALTY = params["train"]["svm_penalty"]
    SVM_LOSS = params["train"]["svm_loss"]
    SVM_DUAL = params["train"]["svm_dual"]
    USE_PCA = params["train"]["use_pca"]
    NUM_COMPONENTS = params["train"]["num_components"]
    CLASS_TYPES = params["train"]["class_types"]
    SVM_KERNEL = params["train"]["svm_kernel"]
    SVM_DEGREE = params["train"]["svm_degree"]
    GAMMA = params["train"]["svm_gamma"]
    MODEL_TYPE = params["train"]["model_type"]
    UNKNOWN_TAG = params["prepare"]["unknown_tag"]
    KNC_N_NEIGHBORS = params["train"]["knc_n_neighbors"]
    KNC_WEIGHTS = params["train"]["knc_weights"]
    KNC_ALGORITHM = params["train"]["knc_algorithm"]
    KNC_LEAF_SIZE = params["train"]["knc_leaf_size"]
    KNC_P = params["train"]["knc_p"]
    KNC_METRIC = params["train"]["knc_metric"]
    seed_everything(SEED)
    label_enc = LabelEncoder()
    label_enc.fit(CLASS_TYPES)
    print(f"Using class types: {', '.join(label_enc.classes_)} ")
    files = glob(fix_path("../data/prepared/*.csv"))
    print(f"Files to use {', '.join(files)}")
    all_xtrain = []
    all_ytrain = []
    all_xtest = []
    all_ytest = []

    for file in files:
        df = pd.read_csv(file, sep="\t")
        xdata = []
        for _, row in tqdm(df.iterrows()):
            values = row["xdata"].replace("\n", " ").replace("[", " ").replace("]", " ")
            xdata.append(
                np.array([float(tmp) for tmp in values.split() if tmp])
            )  # type:ignore
        xdata = np.vstack(xdata)  # type: ignore

        if USE_PCA:
            pca = PCA(n_components=NUM_COMPONENTS)
            xdata = pca.fit_transform(xdata)

        ydata = encode_ys(df.class_type.tolist(), label_enc, default_val=UNKNOWN_TAG)
        xtrain, xtest, ytrain, ytest = train_test_split(
            xdata,
            ydata,
            random_state=SEED,
            train_size=SPLIT,
            stratify=ydata,
        )
        all_xtrain.append(xtrain)
        all_ytrain.extend(ytrain)
        all_xtest.append(xtest)
        all_ytest.extend(ytest)
    all_xtrain = np.vstack(all_xtrain)  # type:ignore
    all_ytrain = np.array(all_ytrain)  # type:ignore
    all_xtest = np.vstack(all_xtest)  # type:ignore
    all_ytest = np.array(all_ytest)  # type:ignore
    clf = None
    if MODEL_TYPE.upper() == "LINEARSVC":
        clf = LinearSVC(
            random_state=SEED,
            C=SVM_REGULARIZATION_C,
            penalty=SVM_PENALTY,
            loss=SVM_LOSS,
            dual=SVM_DUAL,
        )
    if MODEL_TYPE.upper() == "SVC":
        clf = SVC(
            random_state=SEED,
            C=SVM_REGULARIZATION_C,
            degree=SVM_DEGREE,
            gamma=GAMMA,
            kernel=SVM_KERNEL,
        )
    if MODEL_TYPE.upper() == "KNeighborsClassifier":
        clf = KNeighborsClassifier(
            n_neighbors=KNC_N_NEIGHBORS,
            weights=KNC_WEIGHTS,
            algorithm=KNC_ALGORITHM,
            leaf_size=KNC_LEAF_SIZE,
            p=KNC_P,
            metric=KNC_METRIC,
        )
    clf.fit(all_xtrain, all_ytrain)  # type: ignore
    y_pred = clf.predict(all_xtest)  # type: ignore
    balanced_accuracy = balanced_accuracy_score(y_true=all_ytest, y_pred=y_pred)
    precision = precision_score(y_true=all_ytest, y_pred=y_pred, average="weighted")
    recall = recall_score(y_true=all_ytest, y_pred=y_pred, average="weighted")
    actual = label_enc.inverse_transform(all_ytest).tolist()
    predicted = label_enc.inverse_transform(y_pred).tolist()
    cm = confusion_matrix(actual, predicted, labels=label_enc.classes_)
    metrics_dict = multiclass_confusion_matrix_metrics(cm=cm, labels=label_enc.classes_)
    metrics_dict["balanced_accuracy"] = balanced_accuracy
    metrics_dict["precision"] = precision
    metrics_dict["recall"] = recall
    metrics_dict["test_items"] = len(y_pred)
    metrics_dict["theta_size"] = np.linalg.norm(clf.coef_) if MODEL_TYPE.upper() == "LINEARSVC" else 0  # type: ignore
    print(
        f"Results: {len(y_pred):,} test items, balanced_accuracy: {balanced_accuracy}"
    )
    with open(fix_path("../model/train.metrics.json"), "wt", encoding="utf-8") as fout:
        json.dump(metrics_dict, fout, indent=2)
    with open(fix_path("../model/svm.model.pkl"), "wb") as fout:
        pickle.dump(clf, fout)  # type: ignore
    actual = label_enc.inverse_transform(all_ytest).tolist()
    predicted = label_enc.inverse_transform(y_pred).tolist()
    metrics_df = pd.DataFrame({"actual": actual, "predicted": predicted})
    metrics_df.to_csv(fix_path("../model/class.metrics.csv"), index=False)


if __name__ == "__main__":
    work()
