# Auto Label Pipeline Source
* Fetch
* Prepare
* Train
* Relabel

## `fetch.py`
This program is called to download models and data.

## `prepare.py`
This program is called to process the CSV files in the directory `data/raw` using the downloaded embeddings and language detection models.

## `train.py`
This program builds a classifier (SVM or Linear SVC) to cluster label the data.

## `relabel.py`
This program uses the hyperparamters tuned by the `train` stage, and the data output by the `prepare` stage, to train models using cross validation to predict which instances are mislabeled.
