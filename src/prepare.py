"""`prepare.py` - Extract and transform the data for model building."""
import json
import pickle
import string
from collections import defaultdict, namedtuple, Counter
from glob import glob
from typing import Dict, List, Tuple, ValuesView
import yaml

from tqdm import tqdm
import numpy as np
from numpy import ndarray
import pandas as pd
import fasttext
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from utils import fix_path

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


def remove_pc(x: ndarray, npc: int = 1) -> ndarray:
    """Remove the projection on the principal components. Calling this on a collection of sentence embeddings, prior to comparison, may improve accuracy.

    :param x: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection

    This has been adapted from the SIF paper code: `https://openreview.net/pdf?id=SyK00v5xx`.
    """
    pc: ndarray = compute_pc(x, npc)
    if npc == 1:
        return x - x.dot(pc.transpose()) * pc  # type:ignore
    return x - x.dot(pc.transpose()).dot(pc)  # type:ignore


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


# pylint: disable=too-many-locals,too-many-statements,too-many-branches,too-many-nested-blocks
def work() -> None:
    """
    Process the CSV files in the directory `data/raw`
    using the downloaded embeddings and language detection models.
    :return: None
    """
    # Fetch processing params
    with open(fix_path("../params.yaml"), "r") as fd:
        params = yaml.safe_load(fd)
    # Note: we share some parameters with previous stages
    EMBEDDINGS_FILE = params["fetch_data"]["file"]
    FILTER_LANG_BOOL = params["prepare"]["filter_lang_bool"]
    FILTER_LANG = params["prepare"]["filter_lang"]
    LANG_DETECT_MODEL = params["prepare"]["lang_detect_model"]
    STOPWORDS_FILTER_LANG = params["prepare"]["stopwords_filter_lang"]
    FILTER_COUNT_GTEQ = params["prepare"]["filter_count_gteq"]
    EMBEDDINGS_DIM = params["prepare"]["embeddings_dim"]
    UNKNOWN_TAG = params["prepare"]["unknown_tag"]
    IDF_MODEL_FILE = params["prepare"]["tfidf_dict_pickle_file"]

    IDF_MODEL = {}
    if IDF_MODEL_FILE:
        with open(IDF_MODEL_FILE, "rb") as fin:
            IDF_MODEL = pickle.load(fin)

    STOPWORDS = stopwords.words(STOPWORDS_FILTER_LANG)
    print("Loading embeddings model...")
    model = fasttext.load_model(fix_path(f"../data/raw/{EMBEDDINGS_FILE}"))
    dfs = []
    # Some placeholders for metrics
    total_candidates = 0
    total_filtered_by_language = 0
    total_filtered_by_count_threshold = 0
    for file_index, file in enumerate(glob(fix_path("../data/raw/*.csv"))):
        df = pd.read_csv(file, sep="\t")
        print(f"Processing file: {file}")
        print(f"Total rows: {len(df):,}")
        total_candidates += len(df)
        print(f"Distinct rows: {len(df.drop_duplicates()):,}")
        df.fillna("", inplace=True)
        df.drop_duplicates(inplace=True)
        # check for count field, if found, take the row with the highest count for a data slot
        if "count" in df.columns:
            label_idx = defaultdict(list)  # type:ignore
            dupes = []
            for idx, row in df.iterrows():
                name = row["text_data"]
                if name in label_idx:
                    dupes.append((idx, row["count"], row.to_dict()))
                    dupes.extend(label_idx[name])
                else:
                    label_idx[name].append((idx, row["count"], row.to_dict()))
            dupes.sort(key=lambda x: x[1], reverse=True)  # type:ignore
            to_remove = set()
            seen = set()
            for idx, item in enumerate(dupes):
                df_idx, _, row_dict = item
                if row_dict["text_data"] in seen:
                    to_remove.add(df_idx)
                else:
                    seen.add(row_dict["text_data"])
            print(
                f"Number of rows with possible duplicates: {len(dupes):,}; items to remove: {len(to_remove):,}; unique items: {len(seen):,}"
            )
            df.drop(to_remove, inplace=True)
            print(
                f"Number of rows with distinct text_data entries: {len(df):,}; keeping highest occurrence counts"
            )
            df.dropna(subset=["text_data"], inplace=True)
            df.fillna("", inplace=True)
            print(f"Size after dropping NaN entries: {len(df):,}")

        if "count" in df.columns:
            print(
                f"Filtering data entries so that their count is greater than or equal to: {FILTER_COUNT_GTEQ}"
            )
            total_filtered_by_count_threshold += len(
                df.query(f"count < {FILTER_COUNT_GTEQ}").index
            )
            df.drop(df.query(f"count < {FILTER_COUNT_GTEQ}").index, inplace=True)
            print(f"Dataframe thinned to {len(df):,} entries")

        if "context" in df.columns and FILTER_LANG_BOOL:
            print("Filtering data entries by language detection.")
            lang_detect_model = fasttext.load_model(
                fix_path(f"../data/raw/{LANG_DETECT_MODEL}")
            )
            to_remove = set()
            for idx, row in df.iterrows():
                if len(row["context"]) > 0:
                    preds = lang_detect_model.predict(
                        f"{row['text_data']} { row['context']}", k=3
                    )
                    preds = fast_text_prediction_to_language_code(preds)
                    if FILTER_LANG == "en":
                        # we want the keyword to contain at least one english letter;
                        # language detection models don't work great with single word predictions
                        if (
                            len(
                                set(list(string.ascii_letters))
                                & set(list(row["text_data"]))
                            )
                            == 0
                        ):
                            to_remove.add(idx)
                            continue
                    if FILTER_LANG not in preds:
                        to_remove.add(idx)
            if to_remove:
                print(
                    f"Number of rows not matching language filter: {len(to_remove):,}"
                )
                total_filtered_by_language += len(to_remove)
                df.drop(to_remove, inplace=True)

        defaults = [np.zeros(EMBEDDINGS_DIM) for i in range(len(df))]
        df.insert(loc=len(df.columns), column="xdata", value=defaults)
        df["xdata"].astype(object)
        embeddings = []
        for idx, row in tqdm(df.iterrows()):
            text_data = row["text_data"]
            tokens = [
                Token(item, model.get_word_vector(item))
                for item in text_data.split()
                if item and item not in STOPWORDS
            ]
            if tokens:
                if len(tokens) > 1:
                    if not IDF_MODEL:
                        embeddings.append(
                            get_sent_embeddings(
                                tokens,
                                idf_model={
                                    tok.string.lower(): 1.0 / len(tokens)
                                    for tok in tokens
                                },
                                min_idf=1e-6,
                                max_idf=1.0,
                            )
                        )
                    else:
                        embeddings.append(
                            get_sent_embeddings(
                                tokens, idf_model=IDF_MODEL, min_idf=1e-6, max_idf=1.0
                            )
                        )
                else:
                    embeddings.append(tokens[0].embedding)
            else:
                embeddings.append(np.zeros(EMBEDDINGS_DIM))
        df["xdata"] = embeddings
        df.to_csv(
            fix_path(f"../data/interim/datafile.{file_index}.csv"),
            index=False,
            sep="\t",
        )
        dfs.append(df)
    # This next section could be a separate job
    # we load all the csv files, and any text_data field that is shared across multiple files
    # is marked as class_type `UNKNOWN_TAG` (usually UNK), although this sounds extreme, any mislabeled entries
    # will be corrected by the relabel process
    alldfs = pd.concat(dfs)
    print(f"Total items before dropping 2nd+ dupes: {len(alldfs):,}")
    text_data_cnt: Dict[int, int] = Counter(alldfs.text_data.tolist())
    dupe_texts = [key for key, val in text_data_cnt.items() if val > 1]
    print(f"Marking duplicates as class_type: {UNKNOWN_TAG}")
    alldfs["class_type"].mask(
        alldfs.text_data.isin(dupe_texts), other=UNKNOWN_TAG, inplace=True
    )
    alldfs.drop_duplicates(subset=["text_data"], keep="first", inplace=True)
    print(f"Total number of labeled items: {len(alldfs):,}")
    alldfs.to_csv(
        fix_path("../data/prepared/data.all.csv"),
        index=False,
        sep="\t",
    )
    with open(fix_path("../reports/prepare.metrics.json"), "wt") as fout:
        json.dump(
            {
                "total_candidates": total_candidates,
                "total_filtered_by_language": total_filtered_by_language,
                "total_filtered_by_count_threshold": total_filtered_by_count_threshold,
            },
            fout,
        )
    print("Done!")


if __name__ == "__main__":
    work()
