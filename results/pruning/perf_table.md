# Pruned-variant throughput (Intel Arc 140V, 32 GB RAM)

Selected set: geometry prompts of mixed token length; for batch b the first b prompts.
Effective TFLOPS = tokens * 20.534 GFLOP / compute seconds.

| config | variant | mode | device | batch | tokens | total s | tok/s | eff TFLOPS | peak RSS GB | XPU peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| cpu_n16 | N16_general | ram | cpu | 3 | 419 | 63.2 | 6.6 | 0.136 | 18.54 | 0.0 |
| n16_general_disk | N16_general | disk | xpu | 1 | 136 | 21.3 | 6.4 | 0.217 | 4.49 | 0.81 |
| n16_general_disk | N16_general | disk | xpu | 16 | 2230 | 26.2 | 85.1 | 1.950 | 5.48 | 1.35 |
| n16_general_disk | N16_general | disk | xpu | 64 | 8918 | 79.7 | 111.9 | 2.553 | 7.49 | 3.06 |
| n32_general_disk | N32_general | disk | xpu | 1 | 136 | 33.5 | 4.1 | 0.133 | 5.56 | 1.21 |
| n32_general_disk | N32_general | disk | xpu | 16 | 2230 | 45.2 | 49.3 | 1.400 | 6.55 | 1.74 |
| n32_general_disk | N32_general | disk | xpu | 64 | 8918 | 98.6 | 90.5 | 2.133 | 8.56 | 3.41 |

Full model: 32.0 tok/s at batch 64, layer-stream floor 77.2 s.

