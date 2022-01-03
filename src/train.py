"""`train.py` - Train model"""
import json
import pickle
from glob import glob
from typing import List
import yaml

from tqdm import tqdm
import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from utils import fix_path, seed_everything


def compute_test_error(test_y: ndarray, pred_test_y: ndarray) -> float:
    """Compute the test error for the test set."""
    return 1.0 - np.mean(pred_test_y == test_y)  # type:ignore


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
    with open(fix_path("../params.yaml"), "r") as fd:
        params = yaml.safe_load(fd)
    # Note: we share some parameters with previous stages
    SEED = params["train"]["seed"]
    SPLIT = params["train"]["split"]
    REGULARIZATION_C = params["train"]["regularization_C"]
    PENALTY = params["train"]["penalty"]
    LOSS = params["train"]["loss"]
    DUAL = params["train"]["dual"]
    USE_PCA = params["train"]["use_pca"]
    NUM_COMPONENTS = params["train"]["num_components"]
    CLASS_TYPES = params["train"]["class_types"]
    KERNEL = params["train"]["kernel"]
    DEGREE = params["train"]["degree"]
    GAMMA = params["train"]["gamma"]
    MODEL_TYPE = params["train"]["model_type"]
    UNKNOWN_TAG = params["prepare"]["unknown_tag"]

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

        xtrain, xtest, ytrain, ytest = train_test_split(
            xdata,
            df.class_type.tolist(),
            random_state=SEED,
            train_size=SPLIT,
            stratify=df.class_type.tolist(),
        )
        all_xtrain.append(xtrain)
        ytrain = encode_ys(ytrain, label_enc, default_val=UNKNOWN_TAG)
        all_ytrain.extend(ytrain)
        all_xtest.append(xtest)
        ytest = encode_ys(ytest, label_enc, default_val=UNKNOWN_TAG)
        all_ytest.extend(ytest)
    all_xtrain = np.vstack(all_xtrain)  # type:ignore
    all_ytrain = np.array(all_ytrain)  # type:ignore
    all_xtest = np.vstack(all_xtest)  # type:ignore
    all_ytest = np.array(all_ytest)  # type:ignore
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
    svm.fit(all_xtrain, all_ytrain)  # type: ignore
    y_pred = svm.predict(all_xtest)  # type: ignore
    test_error = compute_test_error(test_y=all_ytest, pred_test_y=y_pred)  # type: ignore
    accuracy = accuracy_score(y_true=all_ytest, y_pred=y_pred)
    metrics_dict = {
        "accuracy": accuracy,
        "num_test_items": len(y_pred),
        "test error": test_error,
        "clf._coef": svm.coef_.tolist(),  # type: ignore
        "theta_size": np.linalg.norm(svm.coef_),  # type: ignore
    }
    print(f"Results: {len(y_pred):,} test items, test error: {test_error:.3f}")

    with open(fix_path("../model/train.metrics.json"), "wt") as fout:
        json.dump(metrics_dict, fout, indent=2)
    with open(fix_path("../model/svm.model.pkl"), "wb") as fout:  # type: ignore
        pickle.dump(svm, fout)  # type: ignore
    actual = label_enc.inverse_transform(all_ytest).tolist()
    predicted = label_enc.inverse_transform(y_pred).tolist()
    metrics_df = pd.DataFrame({"actual": actual, "predicted": predicted})
    metrics_df.to_csv(fix_path("../model/classes.metrics.csv"), index=False)


if __name__ == "__main__":
    work()
