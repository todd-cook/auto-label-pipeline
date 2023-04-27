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
from copy import deepcopy
from glob import glob

import numpy as np
import pandas as pd
import yaml
from cleanlab.filter import find_label_issues
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from models import WrappedLinearSVC, WrappedSVC
from train import encode_ys
from utils import fix_path, print_label_corrections, seed_everything


# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def work() -> None:
    """
    Uses the hyperparameters tuned by the `train` stage,
    and the data output by the `prepare` stage,
     to train models using cross validation to predict which instances are mislabeled.
    :return: None
    """
    with open(fix_path("../params.yaml"), "rt", encoding="utf-8") as fd:
        params = yaml.safe_load(fd)
    # Note: we share some parameters with previous stages
    SPLIT = params["train"]["split"]
    SEED = params["train"]["seed"]
    REGULARIZATION_C = params["train"]["svm_regularization_C"]
    PENALTY = params["train"]["svm_penalty"]
    LOSS = params["train"]["svm_loss"]
    DUAL = params["train"]["svm_dual"]
    CLASS_TYPES = params["train"]["class_types"]
    KERNEL = params["train"]["svm_kernel"]
    DEGREE = params["train"]["svm_degree"]
    GAMMA = params["train"]["svm_gamma"]
    MODEL_TYPE = params["train"]["model_type"]
    UNKNOWN_TAG = params["prepare"]["unknown_tag"]
    KNC_N_NEIGHBORS = params["train"]["knc_n_neighbors"]
    KNC_WEIGHTS = params["train"]["knc_weights"]
    KNC_ALGORITHM = params["train"]["knc_algorithm"]
    KNC_LEAF_SIZE = params["train"]["knc_leaf_size"]
    KNC_P = params["train"]["knc_p"]
    KNC_METRIC = params["train"]["knc_metric"]
    NUM_CROSSVAL_FOLDS = params["relabel"]["num_crossval_folds"]
    MIN_DISTANCE_DECISION = params["relabel"]["min_distance_decision"]
    MAX_DISTANCE_DECISION = params["relabel"]["max_distance_decision"]
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

    # using the clf sconfig that we are happy with as a classifier;
    # the Cleanlab cross-validation will expose and correct any label weaknesses
    clf = None
    if MODEL_TYPE.upper() == "LINEARSVC":
        clf = WrappedLinearSVC(
            random_state=SEED,
            C=REGULARIZATION_C,
            penalty=PENALTY,
            loss=LOSS,
            dual=DUAL,
            min_dist=MIN_DISTANCE_DECISION,
            max_dist=MAX_DISTANCE_DECISION,
        )
    if MODEL_TYPE.upper() == "SVC":
        clf = WrappedSVC(
            random_state=SEED,
            C=REGULARIZATION_C,
            degree=DEGREE,
            gamma=GAMMA,
            kernel=KERNEL,
            min_dist=MIN_DISTANCE_DECISION,
            max_dist=MAX_DISTANCE_DECISION,
        )
    if MODEL_TYPE.upper() == "KNEIGHBORSCLASSIFIER":
        clf = KNeighborsClassifier(
            n_neighbors=KNC_N_NEIGHBORS,
            weights=KNC_WEIGHTS,
            algorithm=KNC_ALGORITHM,
            leaf_size=KNC_LEAF_SIZE,
            p=KNC_P,
            metric=KNC_METRIC,
        )

    print(
        "Calculating label probabilities using "
        f"{NUM_CROSSVAL_FOLDS} cross validation folds, please wait..."
    )
    pred_probs = cross_val_predict(
        deepcopy(clf),
        X,
        y,
        cv=NUM_CROSSVAL_FOLDS,
        method="predict_proba",
    )
    predicted_labels = pred_probs.argmax(axis=1)
    acc = accuracy_score(y, predicted_labels)
    print(f"Cross-validated estimate of accuracy on held-out data: {acc}")
    ranked_label_issues = find_label_issues(
        y,
        pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    print(f"Cleanlab found {len(ranked_label_issues)} label issues.")
    print("Top 15 most likely label errors:")
    print(", ".join(df.iloc[ranked_label_issues[:15]]["text_data"].tolist()))
    print(f"Retraining without the {len(ranked_label_issues):,} noisy labels")
    X_train = np.delete(X, ranked_label_issues, axis=0)
    y_train = np.delete(y, ranked_label_issues, axis=0)
    to_remove = np.delete(np.arange(0, len(df)), ranked_label_issues, axis=0)
    X_test = np.delete(X, to_remove, axis=0)
    clf2 = deepcopy(clf)
    clf2.fit(X_train, y_train)  # type: ignore
    print("Predicting...")
    y_pred = clf2.predict(X_test)  # type: ignore
    label_enc = LabelEncoder()
    label_enc.fit(CLASS_TYPES)
    y_labeled = encode_ys(y_pred, label_enc)
    print("Saving...")
    # relabel the df
    current_labels = df["class_type"].tolist()
    corrected_labels = ["" for i in range(len(df))]
    cleanlab_idx = [-1 for i in range(len(df))]
    date_updated = ["" for i in range(len(df))]
    for idx, i in enumerate(ranked_label_issues):
        corrected_labels[i] = str(current_labels[i])
        current_labels[i] = y_labeled[idx]
        cleanlab_idx[i] = idx + 1
        date_updated[i] = pd.Timestamp.now()
    df["class_type"] = current_labels
    df["previous_class_type"] = corrected_labels
    df["mislabeled_rank"] = cleanlab_idx
    df["date_updated"] = date_updated
    df_copy = df.copy()  # We copy because we still want to run metrics on xdata
    df_copy.drop(
        columns="xdata", inplace=True
    )  # drop huge embeddings, we're done with them
    df_copy.to_csv(fix_path("../data/final/data.csv"), sep="\t", index=False)
    print_label_corrections(df, k=20)
    with open(
        fix_path("../reports/relabel.metrics.json"), "wt", encoding="utf8"
    ) as fout:
        json.dump(
            {
                "label_errors": len(ranked_label_issues),
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
    clf.fit(xtrain, ytrain)  # type: ignore
    y_pred = clf.predict(xtest)  # type: ignore
    actual = label_enc.inverse_transform(ytest).tolist()
    predicted = label_enc.inverse_transform(y_pred).tolist()
    metrics_df = pd.DataFrame({"actual": actual, "predicted": predicted})
    metrics_df.to_csv(fix_path("../data/final/class.metrics.csv"), index=False)
    print("Done!")


if __name__ == "__main__":
    work()
