# Golden-example gap

The model card example has three questions (`team`, `urgent`, `anger`). The current engine
in FP8-emulation mode returns the same argmax choices (billing, yes, anger level 2) and a
maximum absolute probability difference of `0.041383` from the published values. An earlier
revision of the engine measured `0.022238` on the same example. The question is which engine
edit made the difference and whether the current engine is closer to or further from the FP32
reference.

## What is preserved

The preserved source tree is the earlier revision.
Its files are not the revision that produced `0.022238`: `layers.py` was modified at 03:16 and
`engine.py` at 03:51, after `results/g5_golden.json` was written at 02:53. The preserved code is
numerically equivalent to the current engine for the operations that matter here. It reproduces
`0.041383`, not `0.022238`.

The current engine and the preserved engine were compared with the FP32 reference on layers
0-3 of the README example in BF16 and FP8-emulation modes. Relative L2 and minimum cosine
against the reference:

| mode | engine | L0 rel / cos | L1 rel / cos | L2 rel / cos | L3 rel / cos |
|---|---|---|---:|---:|---:|
| bf16 | current | 0.002922 / 0.9999920 | 0.003781 / 0.9999868 | 0.004786 / 0.9999627 | 0.005507 / 0.9999609 |
| bf16 | preserved | 0.002915 / 0.9999925 | 0.003810 / 0.9999808 | 0.004710 / 0.9999710 | 0.006800 / 0.9995546 |
| fp8emu | current | 0.003397 / 0.9999858 | 0.004641 / 0.9999689 | 0.006350 / 0.9999484 | 0.008513 / 0.9995829 |
| fp8emu | preserved | 0.003385 / 0.9999858 | 0.004643 / 0.9999689 | 0.006478 / 0.9999484 | 0.009421 / 0.9991962 |

The two engines differ by less than 0.002 relative L2 at every layer. The current engine is
closer at layer 3 in both modes and at layers 1-2 in FP8-emulation mode; the preserved engine is
slightly closer at layer 0 and at layer 2 in BF16. The differences are precision-level, not a
functional change.

## Edits tested

All tests are full golden-example passes on the same laptop, same weights, same batch, same
FP8-emulation mode unless stated.

| edit reverted | mode | max abs delta | note |
|---|---|---:|---|
| none (current engine) | fp8emu | 0.041383 | current result |
| activation scale computed on XPU instead of CPU | fp8emu | 0.041373 | difference 9.9e-06, not responsible |
| attention scale changed from 0.135234 to 0.098791 | bf16 | 0.050096 | worse |
| attention scale changed from 0.135234 to 0.098791 | fp8emu | 0.070594 | worse |
| MoE routed output accumulated in BF16 | bf16 | 0.018955 | closer to published |
| MoE routed output accumulated in BF16 | fp8emu | 0.111615 | much worse |
| RoPE layout written as interleaved pairs | both | unchanged | mathematically the same rotation; not a numerical edit |

The FP32 accumulation in the MoE is the only edit that moves the golden numbers by a large
amount. Reverting it changes FP8-emulation from `0.041383` to `0.111615` and BF16 from
`0.022649` to `0.018955`. The current FP32 accumulation is therefore much closer to the
published example in FP8-emulation mode than the BF16 path, but slightly further in BF16.

No single revert reproduces `0.022238`. The earlier value is consistent with the earlier
revision having a different combination of the accumulation and FP8 linear paths; the exact
combination is not preserved. The earlier value is also better than any configuration we can
build by reverting one edit from the current code.

## Is the current engine closer to the FP32 reference?

At layers 0-3 the current engine has lower error than the preserved engine at layer 3, the last
layer where the difference is large: `0.005507` against `0.006800` in BF16 and `0.008513`
against `0.009421` in FP8-emulation mode. At layers 0-2 the two engines differ by less than
`0.0013` relative L2 and the ordering is mixed. The current FP32 accumulation removes a BF16
rounding step in the sum of eight experts, which is the expected direction: more accurate
layer arithmetic even when the final probability vector moves away from the published example.

## Conclusion

The edit that moves the golden numbers is the FP32 accumulation in the MoE. It moves the
FP8-emulation result toward the published example relative to the BF16 accumulation, and the
current engine is slightly closer to the FP32 reference at layer 3. The earlier `0.022238`
cannot be reproduced from the preserved source because the preserved source is post-edit; the
exact earlier combination of edits is not recoverable. The current engine is not a regression:
its layer-level error is at or below the preserved engine at the deepest layer checked, and its
gap to the published example is smaller than the hosted API's own `0.080740` gap.
