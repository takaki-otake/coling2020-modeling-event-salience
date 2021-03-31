# coling2020-modeling-event-salience
Experiments codes for [[Otake+'20] Modeling Event Salience in Narratives via Barthes’ Cardinal Functions (COLING 2020)]( https://www.aclweb.org/anthology/2020.coling-main.160/).

## Requirements
We recommend preparing a new virtual environment (python>=3.6).
- spacy==2.0.18 (for preprocessing)
- sexpdata==0.0.3 (for preprocessing)
- torch==1.3.1
- transformers==2.2.2

## Preparing data
We used [the ProppLearner corpus](https://academic.oup.com/dsh/article/32/2/284/2957394).
You can download the corpus [here](https://dspace.mit.edu/handle/1721.1/100054?show=full).

### Preprocessing the ProppLearner corpus
(1) Extract the necessary annotations from the corpus.
- `$ python src/preprocess_propplearner_with_pred_and_args.py  --input path-to-data-dir-original  --output path-to-data-dir-processed`
> `path-to-data-dir-original`: path to the directory which contains original .sty files in the corpus.

(2) Anonymize sentences for Verb Anonymization (VA)
- `$ python src/preprocess_propplearner_with_verbs.py --input path-to-data-dir-processed --output path-to-data-dir-VA`
 
(3) Anonymize sentences for Predicate and Argument Anonymization (PAA)
- `$ python src/preprocess_propplearner.py --input path-to-data-dir-original --output path-to-data-dir-PAA`


## Reproducing experiments
- Inference w/ Sentence Deletion (SD) using pretrained GPT-2
    - `$ python src/run_sentence_deletion_model.py --event_rem_method SD --model gpt2 --gpu 0 --normalization normalize --contextlen 1024 --input path-to-data-dir-processed --output path-to-results-dir`
    - > `--gpu`: the number of gpu you use. `-1` for CPU.

- Inference w/ Verb Anonymization (VA) using pretrained GPT-2
    - `$ python src/run_sentence_anonymization_model.py --event_rem_method VA --model gpt2 --gpu 0 --normalization normalize --contextlen 1024 --input_original path-to-data-dir-processed --input_anonimized path-to-data-dir-VA --output path-to-results-dir`

- Inference w/ Predicate and Arguments Anonymization (PAA) using pretrained GPT-2
    - `$ python src/run_sentence_anonymization_model.py --event_rem_method PAA --model gpt2 --gpu 0 --normalization normalize --contextlen 1024 --input_original path-to-data-dir-processed --input_anonimized path-to-data-dir-PAA --output path-to-results-dir`

- Evaluation
  - `$ python src/evaluation.py --input path-to-result-dir` 

### Using fine-tuned GPT-2
For all proposed method (SD, VA, PAA), you can use fine-tuned GPT-2 by specifying the path for `--model` argument. For details about fine-tuning GPT-2, see https://huggingface.co/transformers/v2.2.0/examples.html#gpt-2-gpt-and-causal-language-modeling .

## Citation
If you use our code for academic work, please cite:
```
@inproceedings{otake-etal-2020-modeling,
    title = "Modeling Event Salience in Narratives via Barthes{'} Cardinal Functions",
    author = "Otake, Takaki and Yokoi, Sho and Inoue, Naoya and Takahashi, Ryo and Kuribayashi, Tatsuki and Inui, Kentaro",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.160",
    doi = "10.18653/v1/2020.coling-main.160",
    pages = "1784--1794",
}
```
