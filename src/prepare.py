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
"""`prepare.py` - Extract and transform the data for model building."""
import json
import pickle
import string
from collections import Counter, defaultdict
from glob import glob
from typing import Dict

import fasttext
import numpy as np
import pandas as pd
import yaml
from nltk.corpus import stopwords
from tqdm import tqdm

from utils import (
    Token,
    fast_text_prediction_to_language_code,
    fix_path,
    get_sent_embeddings,
)


# pylint: disable=too-many-locals,too-many-statements,too-many-branches,too-many-nested-blocks
def work() -> None:
    """
    Process the CSV files in the directory `data/raw`
    using the downloaded embeddings and language detection models.
    :return: None
    """
    # Fetch processing params
    with open(fix_path("../params.yaml"), "rt", encoding="utf8") as fd:
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
    dupe_texts = {key for key, val in text_data_cnt.items() if val > 1}
    print(f"Marking {len(dupe_texts):,} duplicates as class_type: {UNKNOWN_TAG}")
    class_labels = alldfs["class_type"].tolist()
    # i =0
    # for idx, row in alldfs.iterrows():
    #     if row['text_data'] in dupe_texts:
    #         class_labels[i] = UNKNOWN_TAG
    #     i+=1
    for i, text_data in enumerate(alldfs["text_data"]):
        if text_data in dupe_texts:
            class_labels[i] = UNKNOWN_TAG
    alldfs["class_type"] = class_labels
    alldfs.drop_duplicates(subset=["text_data"], keep="first", inplace=True)
    print(f"Total number of labeled items: {len(alldfs):,}")
    alldfs.to_csv(
        fix_path("../data/prepared/data.all.csv"),
        index=False,
        sep="\t",
    )
    with open(
        fix_path("../reports/prepare.metrics.json"), "wt", encoding="utf8"
    ) as fout:
        json.dump(
            {
                "candidates": total_candidates,
                "filtered_lang": total_filtered_by_language,
                "filtered_count": total_filtered_by_count_threshold,
            },
            fout,
        )
    print("Done!")


if __name__ == "__main__":
    work()
