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
"""`utils.py` - data utilities"""
import os
import pathlib
import random
from collections import namedtuple
from typing import Dict, List, Tuple, Union, ValuesView

import numpy as np
from numpy import ndarray
from sklearn.decomposition import TruncatedSVD


def fix_path(filepath: str, curr_dir=None) -> str:
    """Correct relative file paths regardless of where execution starts.
    Allows one to use source files in IDEs and DVC pipelines."""
    if not curr_dir:
        curr_dir = pathlib.Path(__file__).parent.resolve()
    new_dir = curr_dir / filepath
    return str(new_dir.resolve())


def seed_everything(seed_value: int):
    """Set random seeds to aim for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    # NOTE: if you copy and use this elsewhere, you'll want to
    # uncomment the sections below as appropriate
    # torch.manual_seed(seed_value)
    # PYTHONHASHSEED should be set before a Python program starts
    # But this applies to forked processes too.
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed_value)
    #     torch.cuda.manual_seed_all(seed_value)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = True
    # tf.random.set_seed(seed_value)


def multiclass_confusion_matrix_metrics(
    cm: np.ndarray, labels: List[str]
) -> Dict[str, Union[int, float]]:
    """
    Create a dictionary of multiple class labels and their TP, TN, FP, FN values
    :param cm: Confusion matrix
    :param labels: string labels corresponding to the Confusion Matrix rows and columns
    :return: a dictionary with "label.[TP | TN | FP | FN]" and count

    >>> from sklearn.metrics import confusion_matrix
    >>> ytest = ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c']
    >>> ypred = ['a', 'a', 'a', 'a', 'b', 'c', 'b', 'a', 'b']
    >>> cm = confusion_matrix(ytest, ypred, labels=list("abc"))
    >>> metrics = multiclass_confusion_matrix_metrics(cm, labels=list("abc"))
    >>> assert(metrics['b.TP'] == 1)
    >>> assert(metrics['b.FP'] == 2)
    >>> assert(metrics['b.TN'] == 4)
    >>> assert(metrics['b.FN'] == 2)
    """
    metrics = {}  # type: Dict[str, Union[int,float]]
    # TP: The actual value and predicted value, which is found in the diagonal row of the m.
    TP = {labels[i]: np.diag(cm)[i] for i in range(len(labels))}
    # FN: The sum of values of corresponding rows except the TP value
    FN = {labels[i]: np.sum(cm[i]) - np.diag(cm)[i] for i in range(len(labels))}
    # FP : The sum of values of corresponding column except the TP value.
    FP = {labels[i]: np.sum(cm.T[i]) - np.diag(cm)[i] for i in range(len(labels))}
    # TN: The sum of values of all columns and row
    # except the values of that class that we are calculating the values for.
    TN = {
        labels[i]: np.sum(cm) - (TP[labels[i]] + FP[labels[i]]) - FN[labels[i]]
        for i in range(len(labels))
    }
    for key, val in TP.items():
        metrics[f"{key}.TP"] = int(val)  # cast to int, for painless json serialization
    for key, val in FP.items():
        metrics[f"{key}.FP"] = int(val)
    for key, val in TN.items():
        metrics[f"{key}.TN"] = int(val)
    for key, val in FN.items():
        metrics[f"{key}.FN"] = int(val)
    return metrics


# The following functions are borrowed from my open source contributions
# for CLTK https://github.com/cltk/cltk/blob/master/src/cltk/embeddings/sentence.py
def rescale_idf(val: float, min_idf: float, max_idf: float) -> float:
    """Rescale idf values."""
    return (val - min_idf) / (max_idf - min_idf)


def compute_pc(x: ndarray, npc: int = 1) -> ndarray:
    """Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!

    :param x: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc

    This has been adapted from the SIF paper code: `https://openreview.net/pdf?id=SyK00v5xx`.
    """
    svd: TruncatedSVD = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(x)
    return svd.components_  # type:ignore


Token = namedtuple("Token", "string embedding")


def get_sent_embeddings(
    sent: List[Token],
    idf_model: Dict[str, float],
    min_idf: float,
    max_idf: float,
    dimensions: int = 300,
) -> ndarray:
    """Provides the weighted average of a sentence's word vectors.

    Expectations:
    Word can only appear once in a sentence, multiple occurrences are collapsed.
    Must have 2 or more embeddings, otherwise Principle Component cannot be found and removed.

    :param sent: ``Sentence``
    :param idf_model: a dictionary of tokens and idf values
    :param min_idf: the min idf score to use for scaling
    :param max_idf: the max idf score to use for scaling
    :param dimensions: the number of dimensions of the embedding

    :return ndarray: values of the sentence embedding, or returns an array of zeroes if no sentence embedding could be computed.
    """
    map_word_embedding: Dict[str, Tuple[float, ndarray]] = {
        token.string: (
            rescale_idf(
                idf_model.get(token.string.lower()), min_idf, max_idf  # type:ignore
            ),
            token.embedding,
        )
        for token in sent
        if not np.all((token.embedding == 0))  # skip processing empty embeddings
    }  # type:ignore
    weight_embedding_tuple: ValuesView = map_word_embedding.values()

    if len(weight_embedding_tuple) == 0:
        return np.zeros(dimensions)
    if len(weight_embedding_tuple) == 1:
        return sent[0].embedding  # type:ignore

    weights, embeddings = zip(*weight_embedding_tuple)
    if sum(weights) == 0:
        return np.zeros(dimensions)
    scale_factor: float = 1 / sum(weights)  # type:ignore
    scaled_weights: List[float] = [weight * scale_factor for weight in weights]
    scaled_values: ndarray = np.array(scaled_weights)
    # Apply our weighted terms to the adjusted embeddings
    weighted_embeds: ndarray = embeddings * scaled_values[:, None]
    return np.sum(weighted_embeds, axis=0)  # type:ignore


def fast_text_prediction_to_language_code(res: List[Tuple[str, str]]) -> List[str]:
    """Convert fastText language predictions to language codes"""
    labels, _ = res
    return [tmp[tmp.rfind("_") + 1 :] for tmp in labels]


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
