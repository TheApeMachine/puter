# metallibgen

Compiles all `*.metal` files under `device/metal` into `kernels.metallib`.

## Apply-phase strict FP

Files whose basename ends with `_apply.metal` or `_stats.metal` are compiled with `-ffp-contract=off`.
Apply kernels use separate multiply/add steps to match CPU NEON (no FMA contraction).
Stats kernels keep variance accumulation order-stable so invStdDev matches the CPU
reduction reference. If you rename or split one of these sources, keep the suffix or
add an explicit `-ffp-contract=off` rule in `Generator.MetalArgs`; otherwise parity
can change silently.
