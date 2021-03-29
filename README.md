# coling2020-modeling-event-salience
Experiments codes for [Otake+'20] Modeling Event Salience in Narratives via Barthes’ Cardinal Functions (COLING 2020).

# run experiments
## Sentence Deletion
`python src/run_sentence_deletion_model.py --event_rem_method SD --model gpt2 -gpu 0 --normalization normalize --contextlen 1024 -input /path-to-dir/tsv_format_v1_modified/ -output /path-to-results-dir/`

## Verb/Predicate and arguments Anonymization
`python src/run_sentence_anonymization_model.py --event_rem_method VA -model gpt2 -gpu 1 --normalization normalize --contextlen 1024 --input_original /path-to-dir/tsv_format_v1_modified/ --input_anonimized /path-to-dir/verb_anonymization_sub_tense_sensitive/ -output /path-to-dir/`
