# Result files

Every path in this directory is relative to the repository root. All numbers in
`README.md` come from these files.

## Engine

| file | produced by |
|---|---|
| `engine/inventory.json` | `python scripts/inventory.py --weights weights` |
| `engine/size_table.json` | `python scripts/inventory.py --weights weights` |
| `engine/fp8.json` | `python scripts/verify_numerics.py --weights weights` |
| `engine/encoder.json` | `python scripts/verify_numerics.py --weights weights` |
| `engine/layer.json` | `python scripts/verify_layers.py --weights weights` |
| `engine/golden.json` | `python scripts/golden.py --weights weights` |
| `engine/environment.json` | `python scripts/environment.py --weights weights` |
| `engine/perf.json` | `python scripts/perf.py --weights weights --out results` |

`engine/golden.json` stores one fresh three-question pass in FP8-emulation mode.
A rerun writes the same probabilities; only the `seconds` and `timings` fields
change. The layer thresholds used by `scripts/verify_layers.py` are in the
script, not in the engine.

## Evaluation set

| file | produced by |
|---|---|
| `eval/datasets.json` | `python scripts/download_eval_data.py --data data` |
| `eval/eval_set.json` | `python scripts/build_eval_set.py --weights weights --data data --out results` |
| `eval/labels.json` | `python scripts/build_eval_set.py --weights weights --data data --out results` |

`eval/labels.json` contains one row per test and calibration question with the
id, split, dataset, domain, language, type, label and option count. It contains
no question text.

## Probabilities

| file | produced by |
|---|---|
| `probabilities/<variant>/test.jsonl` | `python scripts/run_eval.py --weights weights --variant <variant> --split test --out results` |
| `probabilities/<variant>/test_meta.json` | the same command |

The JSONL files hold one object per question: `{"id": ..., "probs": [...]}`.
The probability order is the option order stored in the corresponding record of
`data/eval/test.jsonl` and `results/eval/labels.json`. `scripts/run_eval.py`
also writes the per-shard files it starts from; the shipped tree keeps the
merged JSONL.

## Pruning

| file | produced by |
|---|---|
| `pruning/router_stats.npz` | `python scripts/router_stats.py --weights weights --data data/eval/calib.jsonl --out results` |
| `pruning/router_stats.json` | `python scripts/router_stats.py --weights weights --data data/eval/calib.jsonl --compare results/pruning/router_stats_geometry.npz --out results` |
| `pruning/router_stats_geometry.npz` | `python scripts/router_stats.py --weights weights --prompts calibration/geometry_calib.jsonl --name router_stats_geometry --out results` |
| `pruning/router_stats_geometry.json` | `python scripts/router_stats.py --weights weights --prompts calibration/geometry_calib.jsonl --name router_stats_geometry --compare results/pruning/router_stats.npz --out results` |
| `pruning/build_N32_general.json` | `python scripts/build_pruned.py --weights weights --stats results/pruning/router_stats.npz --ranking total --n 32,16 --suffix general --out results` |
| `pruning/build_N16_general.json` | the same command |
| `pruning/build_N16_reap.json` | `python scripts/build_pruned.py --weights weights --stats results/pruning/router_stats.npz --ranking reap --n 16 --suffix reap --out results` |
| `pruning/build_N32_total.json` | `python scripts/build_pruned.py --weights weights --stats results/pruning/router_stats_geometry.npz --ranking total --n 32,16,8 --suffix total --out results` |
| `pruning/build_N16_total.json` | the same command |
| `pruning/build_N8_total.json` | the same command |
| `pruning/build.json` | `python scripts/aggregate_build.py --out results` |
| `pruning/fidelity.json` | `python scripts/eval_fidelity.py --probabilities results/probabilities --labels results/eval/labels.json --out results` |
| `pruning/fidelity_table.md` | the same command |
| `pruning/api_parity.json` | `python scripts/api_parity.py --out results` after `--fetch`, or against the recorded responses |
| `pruning/api_parity.md` | the same command |
| `pruning/perf_runs/n16_general_disk.json` | `python scripts/perf.py --weights weights --variant N16_general --tag n16_general_disk --out results` |
| `pruning/perf_runs/n32_general_disk.json` | `python scripts/perf.py --weights weights --variant N32_general --tag n32_general_disk --out results` |
| `pruning/perf_runs/cpu_n16.json` | `python scripts/perf.py --weights weights --variant N16_general --device cpu --mode ram --tag cpu_n16 --out results` |
| `pruning/perf.json` | `python scripts/aggregate_perf.py --out results` |
| `pruning/perf_table.md` | the same command |
| `pruning/geometry_reference.json` | `python scripts/eval_geometry.py --weights weights --variant full --write-reference --out results` |
| `pruning/geometry_runs/N32_total.json` | `python scripts/eval_geometry.py --weights weights --variant N32_total --tag N32_total --out results` |
| `pruning/geometry_runs/skip2.json` | `python scripts/eval_geometry.py --weights weights --variant N32_total --skip 20,28 --tag skip2 --out results` |
| `pruning/geometry_runs/skip4.json` | `python scripts/eval_geometry.py --weights weights --variant N32_total --skip 20,25,28,29 --tag skip4 --out results` |
| `pruning/geometry_runs/N32_general.json` | `python scripts/eval_geometry.py --weights weights --variant N32_general --tag N32_general --out results` |
| `pruning/geometry_runs/N16_general.json` | `python scripts/eval_geometry.py --weights weights --variant N16_general --tag N16_general --out results` |
| `pruning/ablations.json` | `python scripts/ablation_layer_skip.py --out results` |
| `pruning/ablations.md` | the same command |

`pruning/geometry_runs/N32_reap_quick.json` is the summary recorded during the
narrow-domain ranking comparison. The N32 REAP layer files were removed after
that comparison, so the summary is kept instead of a full probability file.

`pruning/router_stats.json` contains per-layer token counts, zero-count expert
fractions, the eight most used experts and the kept expert ids for N = 32, 16
and 8 under both rankings. The `.npz` files hold the raw count, weight-sum,
saliency-sum and token-count arrays. `router_stats_geometry.json` also stores
the mean kept-set overlap with the general calibration, 0.3212 at N = 32 and
0.2638 at N = 16.

## API

| file | produced by |
|---|---|
| `api/responses.jsonl` | `python scripts/api_parity.py --fetch --data data` with `SARVAM_API_KEY` set |
| `api/golden.json` | `python scripts/api_parity.py --probe` with `SARVAM_API_KEY` set |
| `api/probe.json` | the same command |
| `api/run_summary.json` | the `--fetch` run |

`api/responses.jsonl` has one row per question id with the probability vector,
HTTP status, latency and input-token count. It has no answer text and no key.
The parity script reads the recorded file when `--fetch` is omitted.

## Per-question files

`per_question/per_question_<variant>.csv` is written by
`scripts/eval_fidelity.py`. The six variants are `N32_general`, `N16_general`,
`N32_total`, `N16_total`, `N16_reap` and `N8_total`. The columns are `id`,
`dataset`, `split`, `label`, `pred_full`, `pred_variant`, `p_full_0` through
`p_full_19`, and `p_variant_0` through `p_variant_19`. Unused probability
columns are empty. No question text is stored.

## Checks

`scripts/check_readme_numbers.py` reads `README.md`, removes code blocks and
inline code, and checks that each remaining numeric token appears in a file
under `results/`.
