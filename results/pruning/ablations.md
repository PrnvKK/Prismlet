# Ablations

## Layer skip (N32_total, 63 held-out geometry pixels)

| config | layers skipped | agree | KL | TV | pixel acc | tok/s |
|---|---|---|---|---|---|---|
| unskipped | none | 0.9683 | 0.0261 | 0.0465 | 0.7302 | 70.1 |
| skip2 | [20, 28] | 0.7143 | 0.4141 | 0.2589 | 0.6508 | 73.8 |
| skip4 | [20, 25, 28, 29] | 0.7302 | 0.4656 | 0.2828 | 0.6349 | 69.9 |

## Ranking at N16 (1200 general test questions)

| ranking | n | agree | KL | TV | accuracy | drop (pp) |
|---|---|---|---|---|---|---|
| TOTAL | 1200 | 0.6633 | 0.5148 | 0.2963 | 0.6308 | +18.83 |
| REAP | 1200 | 0.6342 | 0.5960 | 0.3494 | 0.5925 | +22.67 |

## Calibration domain, same 1200 general questions

| calibration | N | agree | KL | accuracy | drop (pp) |
|---|---|---|---|---|---|
| general | N32 | 0.7750 | 0.2866 | 0.7092 | +11.00 |
| general | N16 | 0.6633 | 0.5148 | 0.6308 | +18.83 |
| geometry | N32 | 0.6508 | 0.5626 | 0.6158 | +20.33 |
| geometry | N16 | 0.5742 | 0.7535 | 0.5500 | +26.92 |

