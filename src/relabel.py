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
"""`relabel.py` - relabel noisy data with Cleanlab."""
import json
from glob import glob
import yaml

import cleanlab
import numpy as np
import pandas as pd
from cleanlab.pruning import get_noise_indices
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm

from train import encode_ys
from utils import fix_path, print_label_corrections, seed_everything


class WrappedSVC(SVC):  # Inherits sklearn base classifier
    """
    Cleanlab implementation expects a `predict_proba` function
    Sklearn SVC classes provide the same functionality via a `decision_function` function
    so here we wrap the classes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict_proba(self, X):
        """Provides confidence probability measurements for the predictions"""
        return self.decision_function(X)


class WrappedLinearSVC(LinearSVC):
    """
    Cleanlab implementation expects a `predict_proba` function
    Sklearn SVC classes provide the same functionality via a `decision_function` function
    so here we wrap the classes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict_proba(self, X):
        """Provides confidence probability measurements for the predictions"""
        return self.decision_function(X)


# pylint: disable=too-many-locals,too-many-statements
def work() -> None:
    """
    Uses the hyperparamters tuned by the `train` stage,
    and the data output by the `prepare` stage,
     to train models using cross validation to predict which instances are mislabeled.
    :return: None
    """
    with open(fix_path("../params.yaml"), "r") as fd:
        params = yaml.safe_load(fd)
    # Note: we share some parameters with previous stages
    SPLIT = params["train"]["split"]
    SEED = params["train"]["seed"]
    REGULARIZATION_C = params["train"]["regularization_C"]
    PENALTY = params["train"]["penalty"]
    LOSS = params["train"]["loss"]
    DUAL = params["train"]["dual"]
    CLASS_TYPES = params["train"]["class_types"]
    KERNEL = params["train"]["kernel"]
    DEGREE = params["train"]["degree"]
    GAMMA = params["train"]["gamma"]
    MODEL_TYPE = params["train"]["model_type"]
    UNKNOWN_TAG = params["prepare"]["unknown_tag"]
    FRAC_NOISE = params["relabel"]["frac_noise"]

    seed_everything(SEED)
    label_enc = LabelEncoder()
    label_enc.fit(CLASS_TYPES)
    print(f"Relabeling using class types: {', '.join(label_enc.classes_ )}")

    files = glob(fix_path("../data/prepared/*.csv"))
    print(f"File to use: {files[0]}")
    X = []
    y = []
    all_words = []
    df = pd.read_csv(files[0], sep="\t")
    df.dropna(subset=["text_data", "class_type"], inplace=True)
    df.fillna("", inplace=True)
    xdata = []
    for idx, row in tqdm(df.iterrows()):
        values = row["xdata"].replace("\n", " ").replace("[", " ").replace("]", " ")
        xdata.append(np.array([float(tmp) for tmp in values.split() if tmp]))
        all_words.append(row["text_data"])
    xdata = np.vstack(xdata)  # type: ignore
    ys = df.class_type.tolist()
    X.append(xdata)
    y.extend(encode_ys(ys, label_enc, default_val=UNKNOWN_TAG))
    X = np.vstack(X)  # type: ignore
    y = np.array(y)  # type: ignore

    # We'll save the mapping of index to words for decoding
    idx_label_map = {}
    for idx, name in enumerate(all_words):
        idx_label_map[idx] = name

    # using the svm config that we are happy with as a classifier;
    # the Cleanlab cross-validation will expose and correct any label weaknesses
    svm = None
    psx = None
    if MODEL_TYPE.upper() == "LINEARSVC":
        psx = cleanlab.latent_estimation.estimate_cv_predicted_probabilities(
            X,
            y,
            clf=WrappedLinearSVC(
                random_state=SEED,
                C=REGULARIZATION_C,
                penalty=PENALTY,
                loss=LOSS,
                dual=DUAL,
            ),
        )

    if MODEL_TYPE.upper() == "SVM":
        psx = cleanlab.latent_estimation.estimate_cv_predicted_probabilities(
            X,
            y,
            clf=WrappedSVC(
                random_state=SEED,
                C=REGULARIZATION_C,
                degree=DEGREE,
                gamma=GAMMA,
                kernel=KERNEL,
            ),
        )
    ordered_label_errors = get_noise_indices(
        s=y,  # numpy_array_of_noisy_labels,
        psx=psx,  # numpy_array_of_predicted_probabilities,
        sorted_index_method="normalized_margin",  # Orders label errors
        frac_noise=FRAC_NOISE,
    )
    print(f"ordered_label_errors size: {len(ordered_label_errors):,}")
    print(f"Retraining without the {len(ordered_label_errors):,} noisy labels")
    X_train = np.delete(X, ordered_label_errors, axis=0)
    y_train = np.delete(y, ordered_label_errors, axis=0)
    to_remove = np.delete(np.arange(0, len(df)), ordered_label_errors, axis=0)
    X_test = np.delete(X, to_remove, axis=0)
    svm = None
    if MODEL_TYPE.upper() == "LINEARSVC":
        svm = LinearSVC(
            random_state=SEED,
            C=REGULARIZATION_C,
            penalty=PENALTY,
            loss=LOSS,
            dual=DUAL,
        )
    if MODEL_TYPE.upper() == "SVM":
        svm = SVC(
            random_state=SEED,
            C=REGULARIZATION_C,
            degree=DEGREE,
            gamma=GAMMA,
            kernel=KERNEL,
        )
    svm.fit(X_train, y_train)  # type: ignore
    print("Predicting...")
    y_pred = svm.predict(X_test)  # type: ignore
    label_enc = LabelEncoder()
    label_enc.fit(CLASS_TYPES)
    y_labeled = encode_ys(y_pred, label_enc)
    print("Saving...")
    # relabel the df
    current_labels = df["class_type"].tolist()
    corrected_labels = ["" for i in range(len(df))]
    cleanlab_idx = [-1 for i in range(len(df))]
    date_updated = ["" for i in range(len(df))]
    for idx, i in enumerate(ordered_label_errors):
        corrected_labels[i] = str(current_labels[i])
        current_labels[i] = y_labeled[idx]
        cleanlab_idx[i] = idx + 1
        date_updated[i] = pd.Timestamp.now()
    df["class_type"] = current_labels
    df["previous_class_type"] = corrected_labels
    df["mislabeled_rank"] = cleanlab_idx
    df["date_updated"] = date_updated
    df.to_csv(fix_path("../data/final/data.csv"), sep="\t", index=False)
    print_label_corrections(df, k=20)
    with open(fix_path("../reports/relabel.metrics.json"), "wt") as fout:
        json.dump(
            {
                "label_errors": len(ordered_label_errors),
                "labels_total": len(df),
                "class_dist": {
                    key: len(df.query(f"class_type == '{key}' ")) for key in CLASS_TYPES
                },
            },
            fout,
        )

    print("Creating final metrics")
    xdata = []
    for _, row in tqdm(df.iterrows()):
        values = row["xdata"].replace("\n", " ").replace("[", " ").replace("]", " ")
        xdata.append(
            np.array([float(tmp) for tmp in values.split() if tmp])
        )  # type:ignore
    xdata = np.vstack(xdata)  # type: ignore
    ydata = encode_ys(df.class_type.tolist(), label_enc, default_val=UNKNOWN_TAG)
    xtrain, xtest, ytrain, ytest = train_test_split(
        xdata,
        ydata,
        random_state=SEED,
        train_size=SPLIT,
        stratify=ydata,
    )
    svm = None
    if MODEL_TYPE.upper() == "LINEARSVC":
        svm = LinearSVC(
            random_state=SEED, C=REGULARIZATION_C, penalty=PENALTY, loss=LOSS, dual=DUAL
        )
    if MODEL_TYPE.upper() == "SVM":
        svm = SVC(
            random_state=SEED,
            C=REGULARIZATION_C,
            degree=DEGREE,
            gamma=GAMMA,
            kernel=KERNEL,
        )
    svm.fit(xtrain, ytrain)  # type: ignore
    y_pred = svm.predict(xtest)  # type: ignore
    actual = label_enc.inverse_transform(ytest).tolist()
    predicted = label_enc.inverse_transform(y_pred).tolist()
    metrics_df = pd.DataFrame({"actual": actual, "predicted": predicted})
    metrics_df.to_csv(fix_path("../data/final/class.metrics.csv"), index=False)
    print("Done!")


if __name__ == "__main__":
    work()
