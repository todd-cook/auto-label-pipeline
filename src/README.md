# Auto Label Pipeline Source
* Fetch
* Prepare
* Train
* Relabel

## `fetch.py`
This program is called to download models and data.

## `prepare.py`
This program is called to process the CSV files in the directory `data/raw` using the downloaded embeddings and a language detection model.

## `train.py`
This program builds a classifier (SVC or Linear SVC) to cluster label the data.

## `relabel.py`
This program uses the hyperparamters tuned by the `train` stage, and the data output by the `prepare` stage, and functionality of the [Cleanlab](https://github.com/cleanlab/cleanlab) utility library to train models using cross validation to predict which instances are mislabeled. The data is relabeled and exported to the directory `data/final`.
