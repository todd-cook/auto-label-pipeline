"""`relabel.py` - relabel noisy data with Cleanlab."""
import json
from glob import glob
from typing import Any
import yaml

import cleanlab
import numpy as np
import pandas as pd
from cleanlab.pruning import get_noise_indices
from numpy import ndarray
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm

from train import encode_ys
from utils import fix_path, seed_everything


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
                "num_label_errors": len(ordered_label_errors),
                "num_labels_total": len(df),
                "class_label_distribution": {
                    key: len(df.query(f"class_type == '{key}' ")) for key in CLASS_TYPES
                },
            },
            fout,
        )
    print("Done!")


def compute_confident_joint(psx: ndarray, y: ndarray) -> Any:
    """
    Compute the confident_join for Cleanlab
    :param psx:
    :param y:
    :return:
    """
    # Verify inputs
    psx = np.asarray(psx)

    # Find the number of unique classes if K is not given
    K = len(np.unique(y))

    # Estimate the probability thresholds for confident counting
    # You can specify these thresholds yourself if you want
    # as you may want to optimize them using a validation set.
    # By default (and provably so) they are set to the average class prob.
    thresholds = [np.mean(psx[:, k][y == k]) for k in range(K)]  # P(s^=k|s=k)
    thresholds = np.asarray(thresholds)  # type:ignore

    # Compute confident joint
    confident_joint = np.zeros((K, K), dtype=int)
    for i, row in enumerate(psx):
        y_label = y[i]
        # Find out how many classes each example is confidently labeled as
        confident_bins = row >= thresholds - 1e-6  # type:ignore
        num_confident_bins = sum(confident_bins)
        # If more than one conf class, inc the count of the max prob class
        if num_confident_bins == 1:
            confident_joint[y_label][np.argmax(confident_bins)] += 1
        elif num_confident_bins > 1:
            confident_joint[y_label][np.argmax(row)] += 1

    # Normalize confident joint (use cleanlab, trust me on this)
    confident_joint = cleanlab.latent_estimation.calibrate_confident_joint(
        confident_joint, y
    )
    cleanlab.util.print_joint_matrix(confident_joint)
    return confident_joint


def print_label_corrections(df, k=20) -> None:
    """
    Pretty print noisy labels, before and after
    :param df: the dataframe
    :param k: the number of items to print
    :return: None
    """
    print(f"Showing top {k} mislabeled entries for your log gazing enjoyment:")
    for _, row in (
        df.query("mislabeled_rank > 1 and mislabeled_rank < 20")
        .sort_values("mislabeled_rank")
        .iterrows()
    ):
        print(
            f"{row['text_data']} : previous label: {row['previous_class_type']}, corrected label: {row['class_type']} "
        )


if __name__ == "__main__":
    work()
