# Evaluation datasets

The test set has 1200 questions and the calibration set 500. Both are built by
`scripts/build_eval_set.py` from the public datasets below. Sampling uses seed
20261107; test and calibration share no source row and no normalised question
text. Character caps keep each question under 900 tokens with the Prism
tokenizer.

The raw text of questions from CC-BY-SA datasets is not redistributed with this
repository. For SNLI, BoolQ, SIB-200 and DBpedia14 the released files contain
question ids, labels, predictions and probabilities only. The other datasets
are redistributed under their own licences, so only derived questions are used
and the raw downloads stay outside the repository.

| key | repository | split used | rows used (test / calibration) | licence | question construction |
|---|---|---|---|---|---|
| banking77 | `PolyAI/banking77` | test, train | 120 / 50 | CC-BY-4.0 | customer message, 20 candidate intents including the gold label |
| massive | `mteb/amazon_massive_intent` | test, validation | 160 / 80 | Apache-2.0 card; origin CC-BY-4.0 | assistant request in en, hi, bn, ta, te; 20 candidate intents |
| amazon_polarity | `mteb/amazon_polarity` | test | 80 / 40 | Apache-2.0 card | product review, yes/no sentiment |
| civil_comments | `google/civil_comments` | test, validation | 150 / 60 | CC0-1.0 | comment, four toxicity levels from the toxicity score |
| snli | `stanfordnlp/snli` | test, validation | 100 / 40 | CC-BY-SA-4.0 | premise and hypothesis, yes/no entailment |
| boolq | `google/boolq` | validation, train | 80 / 40 | CC-BY-SA-3.0 | passage and question, yes/no answer |
| mmlu | `cais/mmlu` | test, dev | 130 / 52 | MIT | four-choice knowledge question, all subjects interleaved |
| global_mmlu | `CohereLabs/Global-MMLU` | test, dev | 120 / 45 | Apache-2.0 | four-choice knowledge question in hi, bn, te |
| sib200 | `Davlan/sib200` | test, train | 120 / 48 | CC-BY-SA-4.0 | passage, seven topic labels, en, hi, bn, ta |
| dbpedia14 | `fancyzhx/dbpedia_14` | test | 70 / 25 | CC-BY-SA-3.0 | title and content, 14 entity classes |
| when2call | `nvidia/When2Call` | test | 70 / 20 | CC-BY-4.0 | question and available tools, four next-action choices |

`scripts/download_eval_data.py` records the repository revision, file sizes and
the licence read from each dataset card in `data/datasets.json` and
`results/eval/datasets.json`. Datasets rejected for missing, conflicting or
non-commercial licences are listed there as well.

The narrow calibration set is synthetic and does not use a public dataset. It
contains 96 geometry questions over six fixed 32 by 32 layouts plus the three
model-card example questions. `calibration/geometry_questions.py` regenerates
the 48 prompts from the three seeded layouts; the recorded file
`calibration/geometry_calib.jsonl` is authoritative for all 99 prompts.
