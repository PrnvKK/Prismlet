# Parity against the hosted Sarvam Prism API

Reference = API. n = questions present in every run.

| system | n | argmax agree | mean KL(api||system) | mean TV | mean abs | api acc | system acc | delta (pp) |
|---|---|---|---|---|---|---|---|---|
| N16_general | 1200 | 0.6608 | 0.4937 | 0.2877 | 0.2787 | 0.8175 | 0.6308 | -18.67 |
| N16_reap | 1200 | 0.6283 | 0.5744 | 0.3406 | 0.3297 | 0.8175 | 0.5925 | -22.50 |
| N16_total | 1200 | 0.5758 | 0.7293 | 0.3989 | 0.3872 | 0.8175 | 0.5500 | -26.75 |
| N32_general | 1200 | 0.7775 | 0.2773 | 0.1907 | 0.1831 | 0.8175 | 0.7092 | -10.83 |
| N32_total | 1200 | 0.6508 | 0.5388 | 0.3027 | 0.2932 | 0.8175 | 0.6158 | -20.17 |
| N8_total | 1200 | 0.4958 | 0.8704 | 0.4709 | 0.4569 | 0.8175 | 0.4733 | -34.42 |
| full | 1200 | 0.9267 | 0.0474 | 0.0654 | 0.0625 | 0.8175 | 0.8192 | +0.17 |

Noise floor from 50 repeated identical calls: argmax agreement 1.0000, mean abs 0.001531, mean KL 0.000324, mean TV 0.003252.

