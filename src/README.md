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
This program builds a classifier (SVC or Linear SVC, or KNC) to cluster label the data.

## `relabel.py`
This program uses the hyperparameters tuned by the `train` stage, and the data output by the `prepare` stage, and functionality of the [Cleanlab](https://github.com/cleanlab/cleanlab) utility library to train models using cross validation to predict which instances are mislabeled. The data is relabeled and exported to the directory `data/final`.

### How To Run experiments:

dvc exp run -n knc_model train -S train.model_type=KNeighborsClassifier
dvc exp run -n svc_model train -S train.model_type=SVC
dvc exp run -n linear_svc_model train -S train.model_type=LINEARSVC

dvc exp run -n svc_margin_1 train -S train.regularization_C=1 -S train.model_type=SVC  
dvc exp run -n svc_margin_2 train -S train.regularization_C=0.1 -S train.model_type=SVC
dvc exp run -n svc_margin_3 train -S train.regularization_C=0.01 -S train.model_type=SVC
dvc exp run -n svc_margin_4 train -S train.regularization_C=10  -S train.model_type=SVC
dvc exp run -n svc_margin_5 train -S train.regularization_C=100  -S train.model_type=SVC

dvc exp run -n knc_kn_1 -S train.knc_n_neighbors=3 -S train.model_type=KNeighborsClassifier
dvc exp run -n knc_kn_2 -S train.knc_n_neighbors=5 -S train.model_type=KNeighborsClassifier
dvc exp run -n knc_kn_3 -S train.knc_n_neighbors=7 -S train.model_type=KNeighborsClassifier

dvc exp run -n margin_tune1 train -S train.regularization_C=1 -S train.model_type=LINEARSVC
dvc exp run -n margin_tune2 train -S train.regularization_C=0.1 -S train.model_type=LINEARSVC
dvc exp run -n margin_tune3 train -S train.regularization_C=0.01 -S train.model_type=LINEARSVC
dvc exp run -n margin_tune4 train -S train.regularization_C=10 -S train.model_type=LINEARSVC
dvc exp run -n margin_tune5 train -S train.regularization_C=100 -S train.model_type=LINEARSVC


### View results

dvc exp show --keep  "theta_size|train\.regularization_C|employer\.*|occupation\.*" --drop "candidates|filtered*|UNK*|balanced*|test_items|label_*|class*|fetch*|prepare*|train\.s*|relabel*|data*|model*|src*"
