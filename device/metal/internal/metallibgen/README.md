# metallibgen

Compiles all `*.metal` files under `device/metal` into `kernels.metallib`.

## Apply-phase strict FP

Files whose basename ends with `_apply.metal` are compiled with `-ffp-contract=off`.
Apply kernels use separate multiply/add steps to match CPU NEON (no FMA contraction).
If you rename or split an apply kernel source file, keep the `_apply.metal` suffix or
add an explicit `-ffp-contract=off` rule in `Generator.MetalArgs`; otherwise parity
can change silently.
