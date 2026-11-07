# prismlet

Prismlet runs Sarvam's Prism decision model on a laptop by reading the weights from disk one decoder layer at a time. It also contains the code and measurements for a study that removed routed experts from the same model. The engine is pure PyTorch. The study keeps N of the routed experts per layer, with N = 32, 16 and 8, two calibration sets and two ranking rules.

The repository contains code, numeric results and documentation. It does not contain weights.

## Technical report

The technical report describes the streaming engine and the expert-pruning study:
[PDF](paper/prismlet_report.pdf) and [project page](https://prnvkk.github.io/Prismlet/).

The LaTeX source and figures are in [`paper/`](paper/). The report covers only the two
experiments: running Prism on a laptop integrated GPU and removing routed experts. Every
number comes from a file under [`results/`](results/).

## Hardware and software tested

The measurements come from one machine:

- Intel Arc 140V integrated GPU, 16 GB shared memory
- Intel Core Ultra CPU, 32 GB system RAM
- Windows 11
- Python 3.11
- torch 2.9.1+xpu
- driver 1.6.34541
- SSD read benchmark 1.2147 GB/s with 8 reader threads and 64 MB blocks

The engine also runs on the CPU. It has not been tested on an NVIDIA GPU.

## Install

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions used for these results. The torch wheel comes from the Intel XPU index.

## Get the official weights

```
huggingface-cli download sarvamai/prism --local-dir weights
```

The download is about 106 GB. The engine expects the layout of that repository: `model-embed.safetensors`, `decision_head.safetensors`, `tokenizer.json`, `model.safetensors.index.json` and `model-layer-00.safetensors` through `model-layer-31.safetensors`.

## Run the engine on one request

```python
from prismlet import PrismEngine

engine = PrismEngine(weights_dir="weights", device="xpu")
answer = engine.decide(
    {"ticket": "My card was charged twice for the same order and I still haven't got a refund."},
    {
        "team": {"type": "choice",
                 "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "payments, charges and refunds",
                              "tech": "app or website problems",
                              "delivery": "shipping and order status"}},
        "urgent": {"type": "yes_no",
                   "instructions": "Does this need a reply within the hour?"},
    },
)
print(answer)
```

`PrismEngine` reads `weights` for the embedding, final norm and decision head. `layers_dir` selects a different directory for the 32 decoder layers. `mode` accepts `disk`, `ram` or `device`. The default is `disk`, which uses one pinned ring buffer and a background reader. `emulate_act=True` quantises activations to the e4m3 grid, matching the FP8 path of the published model. The default is bf16.

## Reproduce the results

Run the commands from the repository root. The full model pass over 1200 questions took 52.5 minutes on the tested laptop. A single three-question pass takes 210.8 seconds with FP8 activation emulation.

Engine checks:

```
python scripts/inventory.py --weights weights
python scripts/verify_numerics.py --weights weights
python scripts/verify_layers.py --weights weights
python scripts/golden.py --weights weights
python scripts/environment.py --weights weights
python scripts/perf.py --weights weights --out results
```

Evaluation set:

```
python scripts/download_eval_data.py --data data
python scripts/build_eval_set.py --weights weights --data data --out results
```

Router statistics and pruned weights:

```
python scripts/router_stats.py --weights weights --data data/eval/calib.jsonl --out results
python scripts/build_pruned.py --weights weights --stats results/pruning/router_stats.npz --ranking total --n 32,16 --suffix general --out results
python scripts/build_pruned.py --weights weights --stats results/pruning/router_stats.npz --ranking reap --n 16 --suffix reap --out results
python scripts/build_pruned.py --weights weights --stats results/pruning/router_stats_geometry.npz --ranking total --n 32,16,8 --suffix total --out results
python scripts/aggregate_build.py --out results
```

Probabilities and fidelity:

```
python scripts/run_eval.py --weights weights --variant full --split test --out results
python scripts/run_eval.py --weights weights --variant N32_general --split test --out results
python scripts/run_eval.py --weights weights --variant N16_general --split test --out results
python scripts/run_eval.py --weights weights --variant N16_reap --split test --out results
python scripts/run_eval.py --weights weights --variant N32_total --split test --out results
python scripts/run_eval.py --weights weights --variant N16_total --split test --out results
python scripts/run_eval.py --weights weights --variant N8_total --split test --out results
python scripts/eval_fidelity.py --probabilities results/probabilities --labels results/eval/labels.json --out results
```

Geometry held-out check and ablations:

```
python scripts/eval_geometry.py --weights weights --variant full --write-reference --out results
python scripts/eval_geometry.py --weights weights --variant N32_total --tag N32_total --out results
python scripts/eval_geometry.py --weights weights --variant N32_total --skip 20,28 --tag skip2 --out results
python scripts/eval_geometry.py --weights weights --variant N32_total --skip 20,25,28,29 --tag skip4 --out results
python scripts/eval_geometry.py --weights weights --variant N32_general --tag N32_general --out results
python scripts/eval_geometry.py --weights weights --variant N16_general --tag N16_general --out results
python scripts/ablation_layer_skip.py --out results
```

Throughput aggregation:

```
python scripts/aggregate_perf.py --out results
```

API parity needs `SARVAM_API_KEY` in the environment. The client never reads a key file and redacts the key from errors. With recorded responses in place, the table rebuilds without a key:

```
python scripts/api_parity.py --probe
python scripts/api_parity.py --fetch --data data
python scripts/api_parity.py
```

`scripts/check_readme_numbers.py` checks that every number in this file also appears under `results/`.

## Results

### Engine verification

| check | result |
|---|---|
| tensor count | 732 keys in 33 files |
| total weight bytes | 106216275456 |
| decoder layers | 32 |
| FP8 finite codes | 254 |
| FP8 maximum | 448.0 |
| FP8 minimum positive | 0.001953125 |
| NaN codes in the sampled q projection | 0 |
| dequantisation matches native cast | true for lut, bits and native |
| bf16 linear relative L2 | 0.002436 |
| fp32 linear relative L2 | 2.7159e-07 |
| FP8-emulated linear relative L2 | 0.002281 |
| encoder oracle cases | 203 |
| attention mask cases | 300 |
| option formatting cases | 200 |
| layer comparison failures | 0 |

The layer check runs layers 0 and 1 against an independent fp32 CPU reference. FP32, bf16 and bf16 with FP8 activation emulation all stay inside the thresholds stored in `results/engine/layer.json`.

### Golden example

The three questions from the model card compare as follows. `results/engine/golden.json` stores the full vectors.

| field | engine | published |
|---|---|---|
| team: billing | 0.9862 | 0.986 |
| team: tech | 0.0073 | 0.007 |
| team: delivery | 0.0065 | 0.006 |
| urgent: yes | 0.4677 | 0.45 |
| anger: 0 | 0.0358 | 0.032 |
| anger: 1 | 0.2939 | 0.267 |
| anger: 2 | 0.5206 | 0.51 |
| anger: 3 | 0.1496 | 0.191 |

The largest absolute difference is 0.041383. The argmax choices are billing, yes and anger level 2, which match the model card. The pass took 210.8 seconds in FP8-emulation mode and 151.0 seconds in bf16.

### Throughput

| configuration | batch | tokens | tokens/s | effective TFLOPS |
|---|---|---|---|---|
| full model, disk | 64 | 8662 | 32.0 | 0.920 |
| full model, disk | 16 | 2162 | 14.8 | 0.642 |
| full model, disk | 3 | 402 | 3.0 | 0.143 |
| N32_general, disk | 64 | 8918 | 90.5 | 2.133 |
| N16_general, disk | 64 | 8918 | 111.9 | 2.553 |
| N16_general, CPU, RAM | 3 | 419 | 6.6 | 0.136 |

The full model reads 104068783616 bytes per pass. The measured layer-stream floor is 77.2 seconds. The three-question pass is longer because it also computes the layers.

### API parity

The hosted Sarvam API is the reference. The table covers the 1200 questions present in every run.

| system | n | argmax agreement | mean KL(api\|system) | mean TV | API accuracy | system accuracy | delta (pp) |
|---|---|---|---|---|---|---|---|
| full | 1200 | 0.9267 | 0.0474 | 0.0654 | 0.8175 | 0.8192 | +0.17 |
| N32_general | 1200 | 0.7775 | 0.2773 | 0.1907 | 0.8175 | 0.7092 | -10.83 |
| N16_general | 1200 | 0.6608 | 0.4937 | 0.2877 | 0.8175 | 0.6308 | -18.67 |
| N32_total | 1200 | 0.6508 | 0.5388 | 0.3027 | 0.8175 | 0.6158 | -20.17 |

The noise floor comes from 50 repeated identical calls: argmax agreement 1.0000, mean absolute difference 0.001531, mean TV 0.003252. The API itself differs from the published model-card probabilities by at most 0.0807 on the three-question example, so the remaining gap is not specific to this engine.

### Pruning fidelity

Reference is the full bf16 engine with identical batching. Agreement is the fraction of questions with the same argmax. KL is KL(full || variant). The test set has 1200 questions.

| variant | n | agreement | 95% CI | mean KL | mean TV | accuracy full | accuracy variant | drop (pp) |
|---|---|---|---|---|---|---|---|---|
| N32_general | 1200 | 0.7750 | 0.7505-0.7977 | 0.2866 | 0.1949 | 0.8192 | 0.7092 | 11.00 |
| N16_general | 1200 | 0.6633 | 0.6361-0.6895 | 0.5148 | 0.2963 | 0.8192 | 0.6308 | 18.83 |
| N32_total | 1200 | 0.6508 | 0.6234-0.6773 | 0.5626 | 0.3112 | 0.8192 | 0.6158 | 20.33 |
| N16_total | 1200 | 0.5742 | 0.5460-0.6019 | 0.7535 | 0.4091 | 0.8192 | 0.5500 | 26.92 |
| N16_reap | 1200 | 0.6342 | 0.6065-0.6610 | 0.5960 | 0.3494 | 0.8192 | 0.5925 | 22.67 |
| N8_total | 1200 | 0.4925 | 0.4643-0.5208 | 0.8967 | 0.4823 | 0.8192 | 0.4733 | 34.58 |

The `_general` suffix means the router statistics came from the 500 public-dataset calibration questions. The `_total` suffix means they came from the 96 synthetic geometry questions. `_reap` uses the REAP ranking instead of total saliency.

### Per domain

Argmax agreement with the full model on the general test set.

| domain | n | N32_general | N16_general |
|---|---|---|---|
| NLI | 100 | 0.9800 | 0.9200 |
| intent routing | 280 | 0.9036 | 0.8321 |
| knowledge MCQ | 250 | 0.4080 | 0.3080 |
| reading-comprehension QA | 80 | 0.8875 | 0.8250 |
| sentiment | 80 | 0.9875 | 0.9750 |
| tool-use decision | 70 | 0.8286 | 0.6714 |
| topic / entity type | 70 | 0.9571 | 0.9143 |
| topic classification | 120 | 0.8167 | 0.5917 |
| toxicity / moderation | 150 | 0.6933 | 0.4533 |

### Per language

| language | n | N32_general | N16_general |
|---|---|---|---|
| English | 862 | 0.8225 | 0.7193 |
| Hindi | 102 | 0.6275 | 0.5490 |
| Bengali | 102 | 0.6373 | 0.5294 |
| Tamil | 62 | 0.8065 | 0.5323 |
| Telugu | 72 | 0.5833 | 0.4583 |

The full table, including dataset slices and accuracy, is in `results/pruning/fidelity_table.md`.

### Ablations

Dropping layers from the N32_total variant costs most of its geometry fidelity. Two skipped layers reduce agreement from 0.9683 to 0.7143, and four skipped layers give 0.7302.

TOTAL and REAP differ at N16 on the general test set. TOTAL reaches 0.6633 agreement and 0.5148 mean KL; REAP reaches 0.6342 and 0.5960. The same ordering appears on the 63 held-out geometry pixels: TOTAL 0.9683 agreement and 0.0261 KL, REAP 0.9370 and 0.1225.

The calibration domain matters more than the ranking. General calibration gives 0.7750 agreement at N32 on general questions; geometry calibration gives 0.6508. On the 63 held-out geometry pixels the order reverses: geometry calibration gives 0.9683, general calibration gives 0.6984 at N32 and 0.4762 at N16.

## Limitations

The engine is slow by design. One full pass streams 104068783616 bytes, a 77.2 second floor at the measured disk rate; the three-question example takes 210.8 seconds with FP8 activation emulation. The pruned variants run faster but no pruned weights are included.

The bf16 path is the default. `emulate_act=True` adds FP8 activation quantisation, which is needed to match the published probabilities closely. The engine has no CUDA kernels and has not been tested on NVIDIA hardware.

The evaluation set is capped at 900 tokens per question. Longer inputs are truncated when the set is built. The 1200 test questions use 197585 tokens and the 500 calibration questions use 82555 tokens.

The pruned variants are not released. At N = 32, agreement with the full model is 0.775 overall, 0.980 on NLI and 0.462 on MMLU; accuracy drops 11.00 points. At N = 16 the drop is 18.83 points. These are the measured numbers, and they are the reason the weights stay out of the repository.

Question text from the CC-BY-SA datasets is not redistributed. The released per-question files contain ids, labels, predictions and probabilities only.

## Licence and attribution

The code is licensed under the Apache License 2.0. See `LICENSE` and `NOTICE`.

The input encoding, attention mask, decision head and rotary tables follow the official `sarvamai/prism` implementation (Apache 2.0). The streaming loader, Intel XPU path, router statistics, pruning writer and evaluation scripts are original work.

This project is unofficial. It is not affiliated with or endorsed by Sarvam AI.
