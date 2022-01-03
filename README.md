# Auto Label Pipeline 
A practical ML pipeline for data labeling with experiment tracking using [DVC](https://dvc.org)

Goals: 
* Reproducible ML
* Use [DVC](https://dvc.org) to build a pipeline and track experiments
* Automatically clean noisy data labels using [Cleanlab](https://github.com/cleanlab/cleanlab) cross validation
* Determine which [fastText](https://fasttext.cc) subword embedding performs better for semi-supervised cluster classification
* Determine optimal hyperparameters through experiment tracking
* Prepare casually labeled data for human evaluation

## The Data
For our working demo, we will purify some of the slightly noisy/dirty labels found in Wikidata people entries for attributes for Employers and Occupations. Our initial data labels have been harvested from a [json dump of Wikidata](https://dumps.wikimedia.org/wikidatawiki/entities), the [Kensho Wikidata dataset](https://www.kaggle.com/kenshoresearch/kensho-derived-wikimedia-data), and this [notebook script](https://github.com/todd-cook/ML-You-Can-Use/blob/master/getting_data/extracting_occupation_and_employer_data_from_wikidata.ipynb) for extracting the data.

## The Pipeline
* Fetch
* Prepare
* Train
* Relabel

For more info, see the [README](src/README.md) in the [src](./src) folder.
The pipeline is orchestrated via the [dvc.yaml](dvc.yaml) file, and parameterized via [params.yaml](params.yaml).

### Setup env
`conda create --name auto-label-pipeline python=3.9`
 
### Install requirements
`conda install --file requirements.txt`

## Reproduce the pipeline
`dvc repro`

## View Metrics
* `dvc metrics show`
* `dvc metrics diff`
* etc... see also: [DVC metrics](https://dvc.org/doc/command-reference/metrics)

## View Experiments
* `dvc exp show`
* `dvc exp diff`
* etc... see also: [DVC Experiments](https://dvc.org/doc/command-reference/exp)

## Using/Extending the pipeline
1. Drop your own CSV files into the `data/raw` directory
2. Run the pipeline
3. Tune settings, embeddings, etc, until no longer amused
4. Verify your results manually and by submitting `data/final/data.csv` for human evaluation, using random sampling and drawing heavily from the `mislabeled_rank` entries.

## Data Input Format
Tab separated CSV files, with the fields:
* `text_data` - the item that is to be labeled (single word or short group of words)
* `class_type` - the class label
* `context` - any text that surrounds the `text_data` field *in situ*, or defines the `text_data` item in other words.
* `count` - the number of occurrences of this label; how common it appears in the existing data.

## Data Output format
* (same parameters as the data input plus)
* `date_updated` - when the label was updated
* `previous_class_type` - the previous `class_type` label
* `mislabeled_rank` - records how low the confidence was prior to a re-label

### Conclusions
* For relabeling and cleaning, it's important to have more than two labels, and to specifying an `UNK` label for: unknown; labels spanning multiple groups; or low confidence support.
  * The majority of the [Cleanlab](https://github.com/cleanlab/cleanlab) corrections are shifting mislabeled data items to the unknown `UNK` label.
* Standardizing the input data formats allow users to flexibly use many different data sources.
* Language detection is an important part of data cleaning, however problematic because:
  * Modern languages sometimes "borrow" words from other languages (but not just any words!)
  * Language detection models perform inference poorly with limited data, especially just a single word.
  * Normalization utlities, such as `unidecode` aren't helpful; (the wrong word in more reabable letters is still the wrong word).
* Experimentation parameters often have co-dependencies that would make a simple combinatorial grid search inefficient.

### Recommended readings:
* _Confident Learning: Estimating Uncertainty in Dataset Labels_  by Curtis G. Northcutt, Lu Jiang, Isaac L. Chuang, 31 Oct 2019, https://arxiv.org/abs/1911.00068
* _A Simple but tough-to-beat baseline for sentence embeddings_ by Sanjeev Arora, Yingyu Liang, Tengyu Ma, ICLR 2017, https://openreview.net/pdf?id=SyK00v5xx

Note: this repo layout borrows heavily from the [Cookie Cutter Data Science Layout](https://drivendata.github.io/cookiecutter-data-science ) If you're not familiar with it, please check it out.
