# CPU SIMD / Assembly Gap Analysis

**Repository:** `puter`  
**Scope:** `device/cpu` (host CPU backend only; Metal/CUDA/XLA live under `device/metal`, `device/xla`, etc.)  
**Contract:** [caramba/AGENTS.md](../caramba/AGENTS.md) — every operation requires scalar (Go reference) plus **AVX-512, AVX2, SSE2, and NEON**, each with a dedicated vector kernel (no cross-ISA aliasing, no scalar inside `.s` files, tight ULP parity at N ∈ {1, 7, 64, 1024, 8192}), for **every supported compute dtype** on that op.

**Related:** [METAL_KERNEL_GAPS.md](./METAL_KERNEL_GAPS.md) — same contract for `device/metal`.

**Audit date:** 2026-05-22  
**Machine-checkable registration:** `device/cpu/dispatchaudit/` (`go test ./device/cpu/dispatchaudit/...`)  
**Full test sweep (arm64):** `go test ./device/cpu/...` — **31/32 packages pass**; **`optimizer` fails** (Adam/AdamW NEON parity, see §6).

---

## 1. Executive summary

| ISA path    | Domains with ≥1 `.s` or dispatch hook | vs 32 domains | Gap                                                                             |
|-------------|--------------------------------------:|--------------:|---------------------------------------------------------------------------------|
| Scalar (Go) |                                    32 |            32 | None at domain level                                                            |
| AVX-512     |                                    32 |            32 | **File-level registration only** — many ops/configs still scalar inside domains |
| AVX2        |            2 (`activation`, `pospop`) |            32 | **30 domains: entire ISA missing**                                              |
| SSE2        |            2 (`activation`, `pospop`) |            32 | **30 domains: entire ISA missing**                                              |
| NEON        |                                    20 |            32 | **12 domains: no arm64 SIMD at all**                                            |

**Only `activation` and `pospop` have a full amd64 ladder (AVX-512 → AVX2 → SSE2 → generic).**  
**`matmul` is the only other domain with any AVX2/SSE2**, and only for **BF16/FP16** row kernels — not f32.

**Largest structural gaps:**

1. **AVX2 + SSE2** for 30/32 domains (all except `activation`, `pospop`, plus partial `matmul` reduced types).
2. **NEON** for 12 domains that are AVX-512 + scalar only on arm64.
3. **amd64 f32** for many domains: AVX-512 exists but dispatch falls back to scalar for common configs (convolution “NEON-eligible” shapes, optimizers without AVX-512, sparse matmul, most attention/MHA, etc.).
4. **BF16/FP16 on amd64**: mostly `*TypedScalar` or generic loops; NEON often has real kernels where amd64 does not.
5. **Correctness debt**: scalar “reference” paths use `math.Fast*` approximations; several parity suites allow **ULP > 1**; **optimizer NEON Adam/AdamW fails at 2 ULP** against a 1 ULP test bound.
6. **Dtype coverage is uneven**: most domains vectorize **f32 only** on amd64; **bf16/fp16** often have NEON but not AVX2/SSE2/AVX-512; **f64**, **int8**, **int4**, and **fp8** are missing or partial on many ops (see §2.1–§2.3).

---

## 2. What “done” means (per op × dtype × ISA)

For **each** public CPU op (see §4) and **each** supported dtype:

| Requirement                                               | Status today                                                 |
|-----------------------------------------------------------|--------------------------------------------------------------|
| Go scalar reference (exact op definition)                 | Present everywhere; **not always exact math** (see §5)       |
| `*_avx512_amd64.s` kernel using zmm                       | Partial — 32 domains have files; op coverage varies          |
| `*_avx2_amd64.s` kernel using ymm                         | **2 domains only**                                           |
| `*_sse2_amd64.s` kernel using xmm                         | **2 domains only**                                           |
| `*_neon_arm64.s` kernel using v0–v31                      | **20 domains**; op coverage varies                           |
| `select_*` ladder: AVX512 → AVX2 → SSE2 → scalar (amd64)  | **`activation`, `pospop` only**                              |
| `select_arm64` ladder: NEON → scalar                      | Most NEON domains; 12 domains scalar-only                    |
| Parity tests, max **1 ULP** vs scalar at standard lengths | **Violated** in multiple places (§5–§6)                      |
| Benchmark per kernel                                      | Present for many hot paths; not exhaustively verified per op |

**Registration ≠ completion.** `dispatchaudit` counts assembly files and select hooks; it does **not** prove every op, dtype, and config uses SIMD.

### 2.1 Supported dtype set (platform contract)

Every op that accepts a `dtype.DType` must have real SIMD on **all** of the dtypes in the op’s signature, on **all four** CPU ISAs (unless the op is inherently index-only, e.g. `Int32` indices with float payload).

| Dtype | Role | CPU expectation |
|-------|------|-----------------|
| **Float32** | Primary compute | Full SIMD all four ISAs on every numeric op |
| **Float64** | High-precision compute | Full SIMD where op is defined for f64 |
| **Float16** | Reduced precision | Full SIMD; no LUT-only scalar lanes on hot paths |
| **BFloat16** | Reduced precision | Full SIMD; same as FP16 |
| **Int8** | Quantized storage / dot | SIMD dot, quant, dequant |
| **Int4** | Packed quant | SIMD dequant (and quant if defined) |
| **Float8E4M3 / Float8E5M2** | FP8 elementwise (cpu) | SIMD via widen→compute→narrow where registered |
| **Int32** | Indices (embedding, CE targets, shape) | Correct handling; index gather/scatter need fast paths where bulk |
| **Bool** | Masks | Vectorized apply where used in bulk |

**Not acceptable:** implementing only `dtype.Float32` while the public API takes `format dtype.DType` and silently scalarizes bf16/fp16/f64, or panics on non-f32 without a documented restriction in the op contract.

### 2.2 Dtype coverage summary by domain

Legend: **Y** = dedicated SIMD for that dtype on at least one ISA; **P** = partial (some ops/configs or one arch only); **S** = scalar/generic/LUT only; **—** = not applicable; **X** = missing.

| Domain | f32 | f64 | fp16 | bf16 | int8 | int4 | fp8 | Notes |
|--------|-----|-----|------|------|------|------|-----|-------|
| activation | Y | — | S | S | — | — | — | bf16/fp16: per-lane LUT, not vector |
| elementwise | Y | P | P | P | — | — | P | amd64: f32 SIMD; f64/bf16/fp16 generic; arm64 NEON all; fp8 NEON widen |
| matmul | Y | P | P | P | — | — | — | amd64 f32 AVX512 only; bf16/fp16 AVX512/2/SSE2; f64/sparse scalar |
| dot | Y | — | P | P | P | — | — | amd64: f32 SIMD; reduced types generic |
| reduction | Y | — | P | P | — | — | — | amd64 bf16/fp16 generic |
| convolution | P | — | P | P | — | — | — | Many configs scalar; reduced WIP |
| pool | P | — | P | P | — | — | — | Adaptive/non-fast scalar |
| dropout | Y | X | X | X | — | — | — | **panic** on non-f32 |
| losses | P | — | S | S | — | — | — | Huber/BCE/KL/CE scalar |
| layernorm | Y | — | S | S | — | — | — | Reduced dtypes scalar loops |
| attention | Y | — | — | — | — | — | — | f32-oriented |
| embedding | Y | — | S | S | — | — | — | Bag reduced scalar |
| normalization | Y | — | S | S | — | — | — | Full pass largely scalar |
| quant / dequant | — | — | — | — | Y | P | — | int8/int4 paths |
| optimizer | Y | — | P | P | — | — | — | NEON Adam broken; state f32 |
| rope | Y | — | — | — | — | — | — | |
| shape | Y | — | — | — | — | — | — | Most shape ops scalar |
| math | Y | — | — | — | — | — | — | |
| pospop | Y* | — | — | — | — | — | — | *integer bit widths, not float |
| All others (12 no-NEON + research) | Y | — | — | — | — | — | — | Mostly f32 AVX512 + scalar |

### 2.3 Global dtype backlog (all four ISAs each)

For **each** row, deliver **AVX-512 + AVX2 + SSE2 + NEON** kernels (separate bodies) plus scalar reference:

1. **BF16 + FP16 on every domain** that exposes `format dtype.DType` on tensor math (not just `activation` LUTs).
2. **Float64** on: `elementwise`, `matmul`, `reduction`, `layernorm`, `pool` (where f64 is in API).
3. **Int8 dot + quant/dequant** on amd64 AVX2/SSE2 (NEON exists for some).
4. **Dropout**: implement f16/bf16/f64 or narrow API; today **f32-only panic**.
5. **Losses**: vectorize Huber, BCE, KL, CrossEntropy for f32 and reduced types.
6. **FP8 elementwise**: extend beyond current NEON widen tests to all ISAs if FP8 remains in the platform dtype set.
7. **Optimizer**: bf16/fp16 param SIMD with **f32 state** — fix NEON Adam/AdamW first.

---

## 3. ISA registration by domain (assembly file counts)

Counts: `*_avx512_amd64.s` / `*_avx2_amd64.s` / `*_sse2_amd64.s` / `*_neon_arm64.s` (including monolithic names like `activation/avx512_amd64.s`).

| Domain            | AVX-512 | AVX2 | SSE2 | NEON | amd64 dispatch tier                                                                | arm64 dispatch tier                            |
|-------------------|--------:|-----:|-----:|-----:|------------------------------------------------------------------------------------|------------------------------------------------|
| activation        |       7 |    8 |    7 |    7 | **AVX512→AVX2→SSE2→generic**                                                       | **NEON→generic**                               |
| pospop            |       1 |    1 |    1 |    1 | **AVX512→AVX2→SSE2→generic**                                                       | **NEON→generic**                               |
| matmul            |       3 |    2 |    2 |    5 | f32: **AVX512→scalar**; bf16/fp16: **AVX512→AVX2→SSE2→scalar**; f64/sparse: scalar | NEON f32/f64/sparse + reduced bf16/fp16        |
| elementwise       |       2 |    0 |    0 |    8 | f32: **AVX512→generic**; f64/bf16/fp16: **generic**                                | NEON all reduced types                         |
| convolution       |       5 |    0 |    0 |    8 | f32: eligible configs → **scalar**; else AVX512; bf16/fp16: AVX512 or TypedScalar  | NEON fast + TypedScalar tails                  |
| pool              |       3 |    0 |    0 |    3 | f32 fast: AVX512; else scalar; bf16/fp16: AVX512 or TypedScalar                    | NEON fast + TypedScalar                        |
| dot               |       1 |    0 |    0 |    4 | f32: AVX512→generic; bf16/fp16/int8: **generic**                                   | NEON all                                       |
| reduction         |       1 |    0 |    0 |    7 | f32: AVX512→generic; bf16/fp16: **generic**                                        | NEON                                           |
| dropout           |       1 |    0 |    0 |    1 | AVX512→generic                                                                     | NEON→generic                                   |
| losses            |       1 |    0 |    0 |    1 | MSE/MAE: AVX512→generic; **Huber/BCE/KL/CE: scalar**                               | MSE/MAE: NEON→generic; **others scalar**       |
| layernorm         |       1 |    0 |    0 |    1 | AVX512→generic                                                                     | NEON→generic                                   |
| attention         |       1 |    0 |    0 |    1 | **Partial** AVX512 (flash blocks); rest scalar                                     | **Partial** NEON + scalar orchestration        |
| causal            |       1 |    0 |    0 |    1 | **Partial** AVX512; Cholesky/IV/etc. scalar                                        | **Partial** NEON                               |
| dequant           |       2 |    0 |    0 |    2 | AVX512→generic                                                                     | NEON                                           |
| quant             |       1 |    0 |    0 |    1 | AVX512→generic                                                                     | NEON                                           |
| hawkes            |       1 |    0 |    0 |    1 | **Partial** AVX512 + scalar tails                                                  | **Partial** NEON                               |
| physics           |       1 |    0 |    0 |    1 | **Partial** AVX512 stencils; **FFT/Bohmian scalar**                                | **Partial** NEON                               |
| rope              |       1 |    0 |    0 |    1 | AVX512 blocks + scalar tail                                                        | NEON blocks + scalar tail                      |
| vsa               |       1 |    0 |    0 |    1 | AVX512 bind/bundle/sim; **permute scalar**                                         | Uses elementwise/dot NEON + scalar             |
| optimizer         |       1 |    0 |    0 |    1 | **AVX512→scalar** per step; SGD Nesterov scalar                                    | **NEON→scalar**; **Adam/AdamW parity failing** |
| embedding         |       1 |    0 |    0 |    0 | AVX512→generic                                                                     | **generic only**                               |
| normalization     |       1 |    0 |    0 |    0 | AVX512 row helpers; **full pass scalar**                                           | **generic only**                               |
| masking           |       1 |    0 |    0 |    0 | AVX512→generic                                                                     | **generic only**                               |
| math              |       1 |    0 |    0 |    0 | AVX512 (3 f32 ops) → generic                                                       | **generic only**                               |
| sampling          |       1 |    0 |    0 |    0 | AVX512 partial → generic                                                           | **generic only**                               |
| shape             |       1 |    0 |    0 |    0 | AVX512 (3 f32 ops) → generic                                                       | **generic only**                               |
| checkpoint        |       1 |    0 |    0 |    0 | AVX512→scalar                                                                      | **scalar only**                                |
| interpretability  |       1 |    0 |    0 |    0 | AVX512→scalar                                                                      | **scalar only**                                |
| model_editing     |       1 |    0 |    0 |    0 | AVX512→scalar                                                                      | **scalar only**                                |
| active_inference  |       1 |    0 |    0 |    0 | AVX512→scalar                                                                      | **scalar only**                                |
| predictive_coding |       1 |    0 |    0 |    0 | AVX512→scalar                                                                      | **scalar only**                                |
| tokenizer         |       1 |    0 |    0 |    0 | AVX512→generic                                                                     | **generic only**                               |

**Domains with zero NEON (need full arm64 SIMD stack):**  
`embedding`, `normalization`, `masking`, `math`, `sampling`, `shape`, `checkpoint`, `interpretability`, `model_editing`, `active_inference`, `predictive_coding`, `tokenizer`.

---

## 4. Per-domain op inventory and missing SIMD (by ISA)

Below, **Missing** means no dedicated vector kernel on that ISA for that op (dtype noted). **Partial** means some configs/dtypes use SIMD with scalar tails or orchestration.

### 4.1 `activation` — reference tier (still incomplete dtypes)

**Public ops (f32 unless noted):** Exp, Log, Log1p, Expm1, Sigmoid, LogSigmoid, Tanh, Silu, Swish, GeluTanh, Gelu, LeakyReLU, ELU, CELU, SELU, Softplus, Mish, Softsign, HardSigmoid, HardSwish, HardTanh, HardGelu, QuickGelu, TanhShrink, Softmax, LogSoftmax; parametric: PReLU, PReLUV, LeakyReLUSlope, ELUAlpha, CELUAlpha, Threshold, HardTanhRange, Snake, SnakeParametric, HardShrink, SoftShrink, RReLU; gated: GLU, GeGLU, GeGLUTanh, SwiGLU, ReGLU, SiGLU, LinGLU, SeGLU (+ tensor variants).

| ISA                          | Status                                                                   |
|------------------------------|--------------------------------------------------------------------------|
| AVX-512 / AVX2 / SSE2 / NEON | **f32 unary + softmax + gated: present** (separate `.s` per ISA)         |
| All ISAs                     | **BF16/FP16: LUT lane loops (scalar per element), not vector BF16/FP16** |
| All ISAs                     | Parametric ops: SIMD on f32; reduced dtypes via LUT/scalar               |

**Missing for “full” contract:** BF16/FP16 vector kernels on all four ISAs for every unary/gated op.

---

### 4.2 `pospop`

**Ops:** Count8, Count16, Count32, Count64, CountString.

| ISA                          | Status                                                          |
|------------------------------|-----------------------------------------------------------------|
| AVX-512 / AVX2 / SSE2 / NEON | **Complete for f32-width buckets** (8/16/32/64 bit populations) |

**Missing:** None at f32-equivalent widths. Verify CountString path uses same ladder on amd64.

---

### 4.3 `elementwise`

**Ops:** Add, Sub, Mul, Div, Max, Min, Abs, Neg, Sqrt, ReLU, Axpy (+ f64, bf16, fp16 via dispatch).

| ISA     | Missing                                     |
|---------|---------------------------------------------|
| AVX2    | **All ops, all dtypes**                     |
| SSE2    | **All ops, all dtypes**                     |
| AVX-512 | **f64, bf16, fp16** (f32 only)              |
| NEON    | **Complete for f32/f64/bf16/fp16** on arm64 |

---

### 4.4 `matmul`

**Ops:** Matmul (dense); sparse CSR matmul (f32).

| ISA         | Missing                                                                                           |
|-------------|---------------------------------------------------------------------------------------------------|
| AVX2 / SSE2 | **f32 matmul** (entire ISA)                                                                       |
| AVX-512     | **f64 matmul**, **sparse CSR**                                                                    |
| NEON        | **sparse CSR** (verify; dense f32/f64/bf16/fp16 have kernels)                                     |
| All         | **BF16/FP16 on amd64** need AVX2/SSE2 completion only for reduced — **f32 still needs AVX2/SSE2** |

---

### 4.5 `dot`

**Ops:** Dot (f32, bf16, fp16, int8).

| ISA         | Missing                       |
|-------------|-------------------------------|
| AVX2 / SSE2 | **All dtypes**                |
| AVX-512     | **bf16, fp16, int8**          |
| NEON        | Present for all listed dtypes |

---

### 4.6 `reduction`

**Ops:** Sum, Prod, ReduceMin, ReduceMax, L1Norm.

| ISA         | Missing                                                             |
|-------------|---------------------------------------------------------------------|
| AVX2 / SSE2 | **All ops**                                                         |
| AVX-512     | **bf16, fp16** sum (and other ops if not in avx512 file)            |
| NEON        | **Prod, Min, Max, L1** — verify each has `.s` (sum bf16/fp16 exist) |

---

### 4.7 `convolution`

**Ops:** Conv2D, Conv1D, Conv3D, ConvTranspose2D (f32, bf16, fp16).

| ISA         | Missing                                                                                                          |
|-------------|------------------------------------------------------------------------------------------------------------------|
| AVX2 / SSE2 | **All ops, all dtypes**                                                                                          |
| AVX-512     | **f32 “NEON-eligible” configs** deliberately use **scalar** on amd64; general f32 uses AVX512; bf16/fp16 partial |
| NEON        | **f32** full config coverage; bf16/fp16 general/transpose partial (WIP in tree)                                  |
| All         | **Conv3D f32** vector kernel (patch-dot NEON exists; full 3D volume SIMD incomplete)                             |

---

### 4.8 `pool`

**Ops:** MaxPool2D, AvgPool2D, AdaptiveMaxPool2D, AdaptiveAvgPool2D.

| ISA            | Missing                                                                                              |
|----------------|------------------------------------------------------------------------------------------------------|
| AVX2 / SSE2    | **All ops**                                                                                          |
| AVX-512 / NEON | **Adaptive** pools (scalar); **non-fast** f32 configs (scalar); bf16/fp16 **non-fast** (TypedScalar) |

---

### 4.9 `dropout`

**Ops:** Dropout (f32 only implemented).

| ISA         | Missing                                                |
|-------------|--------------------------------------------------------|
| AVX2 / SSE2 | Dropout                                                |
| NEON        | —                                                      |
| All         | **bf16, fp16, f64** not implemented (panic on non-f32) |

---

### 4.10 `losses`

**Ops:** MSE, MAE, Huber, BinaryCrossEntropy, KLDivergence, CrossEntropy.

| ISA            | Missing                                              |
|----------------|------------------------------------------------------|
| AVX2 / SSE2    | **All**                                              |
| AVX-512 / NEON | **Huber, BCE, KL, CrossEntropy** (scalar typed only) |
| All            | **bf16/fp16** reductions for MSE/MAE                 |

---

### 4.11 `layernorm`

**Ops:** LayerNorm, RMSNorm.

| ISA            | Missing                                                |
|----------------|--------------------------------------------------------|
| AVX2 / SSE2    | **All**                                                |
| AVX-512 / NEON | **bf16/fp16** (scalar loops); vector path f32-oriented |

---

### 4.12 `attention`

**Ops:** ScaledDotProductAttention, FlashAttention, MultiHeadAttention.

| ISA            | Missing                                                                                                        |
|----------------|----------------------------------------------------------------------------------------------------------------|
| AVX2 / SSE2    | **Entire attention stack**                                                                                     |
| AVX-512 / NEON | **Partial** — online softmax / strided dot blocks only; score matmul, masking fusion, full MHA loop **scalar** |

---

### 4.13 `causal`

**Ops:** Cholesky, BackdoorAdjustment, FrontdoorAdjustment, DoIntervene, CATE, Counterfactual, IVEstimate, DAGMarkovFactorization, MarkovFlowActive, MarkovFlowInternal.

| ISA            | Missing                                                                                       |
|----------------|-----------------------------------------------------------------------------------------------|
| AVX2 / SSE2    | **All**                                                                                       |
| AVX-512 / NEON | **Cholesky, Backdoor, Frontdoor, DoIntervene, IVEstimate, DAGMarkov, MarkovFlow** — scalar Go |
| AVX-512 / NEON | **CATE, Counterfactual** — partial SIMD                                                       |

---

### 4.14 `embedding`

**Ops:** Lookup, Bag.

| ISA                | Missing                                     |
|--------------------|---------------------------------------------|
| AVX2 / SSE2 / NEON | **All**                                     |
| AVX-512            | f32 row copy only; **Bag bf16/fp16** scalar |

---

### 4.15 `normalization`

**Ops:** GroupNorm, InstanceNorm, BatchNormEval.

| ISA                | Missing                                                       |
|--------------------|---------------------------------------------------------------|
| AVX2 / SSE2 / NEON | **All**                                                       |
| AVX-512            | Row helpers only; **full norm pass scalar** for all three ops |

---

### 4.16 `masking`

**Ops:** ApplyMask, CausalMask, ALiBiBias.

| ISA                | Missing                          |
|--------------------|----------------------------------|
| AVX2 / SSE2 / NEON | **All**                          |
| AVX-512            | f32 elementwise-style masks only |

---

### 4.17 `math`

**Ops:** InvSqrtDimScale, LogSumExp, Outer (f32 native); Fast* helpers used by activations.

| ISA                | Missing                  |
|--------------------|--------------------------|
| AVX2 / SSE2 / NEON | **All three native ops** |
| AVX-512            | 3 f32 kernels only       |

---

### 4.18 `sampling`

**Ops:** GreedySample, TopKSample, TopPSample.

| ISA                | Missing                                                          |
|--------------------|------------------------------------------------------------------|
| AVX2 / SSE2 / NEON | **All**                                                          |
| AVX-512            | Greedy + softmax row partial; **TopK/TopP orchestration scalar** |

---

### 4.19 `shape`

**Ops with f32 SIMD:** CopyContiguous, Where, MaskedFill.  
**Ops scalar only:** Gather, Scatter, Transpose2D, Concat, Slice, Reshape, Pad, etc. (tensor runners).

| ISA                | Missing                                            |
|--------------------|----------------------------------------------------|
| AVX2 / SSE2 / NEON | CopyContiguous, Where, MaskedFill                  |
| All ISAs           | **Gather, Scatter, Transpose2D, Concat, Slice, …** |

---

### 4.20 `rope`

**Ops:** RoPE, RoPEPairs.

| ISA            | Missing                                                    |
|----------------|------------------------------------------------------------|
| AVX2 / SSE2    | **All**                                                    |
| AVX-512 / NEON | Blocks + **scalar tail** (by design until tail vectorized) |

---

### 4.21 `quant`

### 4.22 `dequant`

**Ops:** Quant (f32→int8); Dequant (int8→f32); Dequant4 (int4→f32).

| ISA            | Missing                                 |
|----------------|-----------------------------------------|
| AVX2 / SSE2    | **All**                                 |
| AVX-512 / NEON | int8 paths present; verify int4 on both |

---

### 4.23 `optimizer`

**Ops:** Adam, AdamW, SGD, Adamax, Adagrad, RMSprop, Lion, LARS, LBFGS, Hebbian step slices (+ bf16/fp16 dispatch where registered).

| ISA            | Missing                                                                                                     |
|----------------|-------------------------------------------------------------------------------------------------------------|
| AVX2 / SSE2    | **All optimizers**                                                                                          |
| AVX-512 / NEON | **Per-op assembly exists** but dispatch is **SIMD block + scalar tail**; **SGD Nesterov → scalar** on amd64 |
| NEON           | **Adam, AdamW failing parity** (§6) — treat as **incorrect until fixed**                                    |

---

### 4.24 `hawkes`

**Ops:** HawkesIntensity, HawkesKernelMatrix, HawkesLogLikelihood, MarkovMutualInformation, MarkovBlanketPartition.

| ISA            | Missing                                                |
|----------------|--------------------------------------------------------|
| AVX2 / SSE2    | **All**                                                |
| AVX-512 / NEON | **MarkovMutualInformation** scalar; exp blocks partial |

---

### 4.25 `physics`

**Ops:** Laplacian, Laplacian4, Grad1D, Divergence1D, FFT1D, IFFT1D, QuantumPotential, BohmianVelocity, MadelungContinuity.

| ISA            | Missing                                                                     |
|----------------|-----------------------------------------------------------------------------|
| AVX2 / SSE2    | **All**                                                                     |
| AVX-512 / NEON | **FFT1D, IFFT1D, BohmianVelocity, Divergence1D** — scalar; stencils partial |

---

### 4.26 `vsa`

**Ops:** Bind, Bundle, Permute, InversePermute, Similarity.

| ISA            | Missing                                             |
|----------------|-----------------------------------------------------|
| AVX2 / SSE2    | **All**                                             |
| AVX-512 / NEON | **Permute / InversePermute** — scalar rotate/copy   |
| NEON           | Delegates bind/bundle/similarity to elementwise/dot |

---

### 4.27 `active_inference`

**Ops:** FreeEnergy, ExpectedFreeEnergy, BeliefUpdate, PrecisionWeight.

| ISA                | Missing                          |
|--------------------|----------------------------------|
| AVX2 / SSE2 / NEON | **All**                          |
| AVX-512            | Small f32 kernels; mostly scalar |

---

### 4.28 `predictive_coding`

**Ops:** Prediction, PredictionError, UpdateRepresentation, UpdateWeights.

| ISA                | Missing                          |
|--------------------|----------------------------------|
| AVX2 / SSE2 / NEON | **All**                          |
| AVX-512            | Present with **scalar fallback** |

---

### 4.29 `checkpoint`

**Ops:** EncodeFloat32Data, DecodeFloat32Data.

| ISA                | Missing |
|--------------------|---------|
| AVX2 / SSE2 / NEON | **All** |

---

### 4.30 `interpretability`

**Ops:** ActivationSteer.

| ISA                | Missing |
|--------------------|---------|
| AVX2 / SSE2 / NEON | **All** |

---

### 4.31 `model_editing`

**Ops:** WeightGraftAdd.

| ISA                | Missing |
|--------------------|---------|
| AVX2 / SSE2 / NEON | **All** |

---

### 4.32 `tokenizer`

**Ops:** PackInt32.

| ISA                | Missing |
|--------------------|---------|
| AVX2 / SSE2 / NEON | **All** |

---

## 5. Correctness and approximation debt

These items violate or weaken the AGENTS.md rule that SIMD must match the **exact mathematical definition** of the op with **tight ULP** parity — not widened test bands.

### 5.1 Scalar reference uses fast approximations (`device/cpu/math`)

| Symbol                        | Location                     | Issue                                                | Required fix                                                                                                                 |
|-------------------------------|------------------------------|------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `FastExp32` / `FastExp64`     | `math/f32.go`, `math/f64.go` | Minimax / bit-hack **polynomial exp**                | Either rename ops as approximate variants, or replace scalar + all SIMD paths with **exact** `exp` (libm-quality vector exp) |
| `FastLog32`                   | `math/f32.go`                | Polynomial ln                                        | Same                                                                                                                         |
| `FastTanh32`                  | `math/f32.go`                | Padé approximant                                     | Same                                                                                                                         |
| `FastSigmoid32`, `FastSilu32` | `math/f32.go`                | Built on FastExp                                     | Same                                                                                                                         |
| `FastSin32`                   | `math/f32.go`                | Minimax sine                                         | Same                                                                                                                         |
| `FastGeluTanh32`              | `math/f32.go`                | Tanh-based GELU — **correct for `GeluTanh` op only** | Keep for `GeluTanh`; do not use as reference for exact `Gelu`                                                                |
| `FastQuickGelu32`             | `math/extra_activation.go`   | **Defined approximate** (`σ(1.702x)·x`)              | OK only if API is `QuickGelu`; SIMD must match this definition exactly                                                       |
| `FastGelu32`                  | `math/activation.go`         | Uses `math.Erf` — **exact GELU reference**           | SIMD `GeluF32*` must match `Erf` within **1 ULP**; generic parity allows **2 ULP** today — tighten                           |

**Downstream:** `activation/f32_generic.go` uses `FastExp32`, `FastLog32`, `FastTanh32`, etc. for Exp, Log, Tanh, ELU, Softplus, Mish, … So the **scalar reference is already approximate** for those ops. SIMD is tested against that reference at **maxULP 2** (`activation_avx512_parity_test.go`), not against libm.

**Action:** For each activation op, pick one definition:

- **Exact** (libm / vector libm-quality): rewrite scalar + AVX512 + AVX2 + SSE2 + NEON.
- **Approximate** (documented): keep Fast* but **rename** public ops or document in manifest; parity stays tight vs that definition.

### 5.2 Ops with intentional approximate names

| Op          | Definition              | SIMD status                                                                       |
|-------------|-------------------------|-----------------------------------------------------------------------------------|
| `GeluTanh`  | tanh form               | SIMD matches `FastGeluTanh32` — OK if op name stays GeluTanh                      |
| `QuickGelu` | sigmoid shortcut        | SIMD + scalar aligned; parity maxULP **1** in AVX512 table                        |
| `Gelu`      | `0.5 x (1 + erf(x/√2))` | Scalar exact via `Erf`; SIMD claims erf in comments — **tighten parity to 1 ULP** |

### 5.3 Parity tolerances wider than 1 ULP (must fix kernel or reference)

| Area                           | maxULP / tolerance                              | File                                                                |
|--------------------------------|-------------------------------------------------|---------------------------------------------------------------------|
| Optimizer AVX-512              | **2**                                           | `optimizer/optimizer_avx512_parity_test.go`                         |
| Optimizer NEON Adam/AdamW      | **1 required, 2 observed — FAILING**            | `optimizer/f32_neon_arm64_test.go`                                  |
| Hawkes AVX-512 / NEON          | **4**                                           | `hawkes/hawkes_avx512_parity_test.go`, `hawkes_neon_parity_test.go` |
| Hawkes scalar                  | **4**                                           | `hawkes/hawkes_f32_scalar_parity_test.go`                           |
| Math AVX-512                   | **2**                                           | `math/math_avx512_parity_test.go`                                   |
| Sampling AVX-512               | **2**                                           | `sampling/sampling_avx512_parity_test.go`                           |
| Pool AVX-512 / NEON            | **2** (some cases **0**)                        | `pool/pool_avx512_parity_test.go`, `f32_neon_arm64_test.go`         |
| Convolution NEON               | **2**                                           | `conv2d_neon_arm64_test.go`                                         |
| Conv3d patch dot               | **4**                                           | `conv3d_neon_arm64_test.go`                                         |
| Elementwise AVX-512            | **2** (many cases)                              | `elementwise/elementwise_avx512_parity_test.go`                     |
| Activation AVX-512             | **2** (most unary), **1** (ReLU, hard variants) | `activation/activation_avx512_parity_test.go`                       |
| Active inference log ops       | **2**                                           | `active_inference/test_helpers_test.go`                             |
| Physics NEON                   | **2**                                           | `physics/physics_neon_parity_test.go`                               |
| neon/remaining mixed-precision | **2–4**                                         | `neon/remaining_neon_arm64_test.go`                                 |

**Action:** Fix kernels until **1 ULP** (or **0** for bitwise ops: dropout mask, copy, etc.). Do not widen tests.

### 5.4 Known failing tests (arm64, 2026-05-22)

```
go test ./device/cpu/optimizer/... 
  FAIL TestAdamStepSlicesNEONParity   (ulp=2 at N=64,1024,8192)
  FAIL TestAdamWStepSlicesNEONParity  (same class)
```

### 5.5 Dispatch paths that bypass SIMD (performance + contract)

| Path                                    | Location                                       | Issue                                                  |
|-----------------------------------------|------------------------------------------------|--------------------------------------------------------|
| Conv2D f32 “NEON-eligible” on **amd64** | `convolution/select_amd64.go`                  | Routes to **`Conv2DFloat32Scalar`** instead of AVX-512 |
| Matmul f32 without AVX-512F             | `matmul/select_amd64.go`                       | **Scalar only** — need AVX2/SSE2 ladder                |
| Optimizer without AVX-512 / NEON block  | `optimizer/select_amd64.go`, `select_arm64.go` | **Scalar**                                             |
| SGD Nesterov                            | `optimizer/select_amd64.go`                    | **Always scalar**                                      |
| Sparse matmul                           | `matmul/select_amd64.go`                       | **Always scalar**                                      |
| Dropout non-f32                         | `dropout/ops.go`                               | **panic** — not implemented                            |

---

## 6. Global checklist: missing ISA × domain

Use as implementation backlog. Each cell is “add full vector kernel + select hook + 1 ULP parity + benchmark”.

### AVX2 — missing in 30 domains

`active_inference`, `attention`, `causal`, `checkpoint`, `convolution`, `dequant`, `dot`, `dropout`, `elementwise`, `embedding`, `hawkes`, `interpretability`, `layernorm`, `losses`, `masking`, `math`, `matmul` (f32), `model_editing`, `normalization`, `optimizer`, `physics`, `pool`, `predictive_coding`, `quant`, `reduction`, `rope`, `sampling`, `shape`, `tokenizer`, `vsa`.

**Exception:** `matmul` needs AVX2 for **f32**; already has AVX2 for **bf16/fp16** only.

### SSE2 — same 30 domains as AVX2

Same list as AVX2.

### NEON — missing in 12 domains

`active_inference`, `checkpoint`, `embedding`, `interpretability`, `masking`, `math`, `model_editing`, `normalization`, `predictive_coding`, `sampling`, `shape`, `tokenizer`.

---

## 7. Recommended implementation order

Priority weights **correctness** first, then **coverage of hot paths**, then **ISA breadth**.

1. **Fix optimizer NEON Adam/AdamW** — failing 2 ULP vs 1 ULP bound; then tighten optimizer AVX-512 parity from 2 → 1 ULP.
2. **Define exact vs approximate** for activation family; align scalar reference with SIMD for Exp/Log/Tanh/Mish/… or document approximate ops.
3. **amd64 AVX2/SSE2 ladders** for: `elementwise`, `matmul` (f32), `dot`, `reduction`, `pool`, `convolution` (f32 all configs), `losses` (MSE/MAE first).
4. **NEON for 12 zero-NEON domains** — start with `embedding`, `masking`, `shape` (3 kernels each), `math`, `sampling`.
5. **Complete partial domains:** `attention` (full flash/MHA), `losses` (Huber, CE), `normalization` (full vector norm), `physics` (FFT).
6. **BF16/FP16 on amd64** — match arm64 NEON coverage domain by domain.
7. **Tighten all parity suites** to ≤1 ULP; remove hawkes/conv3d **4 ULP** bands.

---

## 8. Verification commands

```bash
# Registration matrix
go test ./device/cpu/dispatchaudit/... -v -count=1

# Full CPU package (run on target GOARCH)
go test ./device/cpu/... -count=1

# Optimizer NEON parity only
go test ./device/cpu/optimizer/... -run 'TestAdam.*NEON' -v -count=1
```

**Definition of done for any closed gap:** paste test + benchmark output in the PR; parity at N ∈ {1, 7, 64, 1024, 8192}; max ULP ≤ 1 (0 where bitwise exact).

---

## 9. Related docs

- [METAL_KERNEL_GAPS.md](./METAL_KERNEL_GAPS.md) — Metal GPU kernel and dtype gaps (same op surface).
- `device/cpu/dispatchaudit/matrix_test.go` — expected registration counts (32 AVX-512, 2 AVX2, 2 SSE2, 20 NEON).
- `device/cpu/parity/parity.go` — shared ULP helpers and standard lengths.
- `caramba/AGENTS.md` — backend implementation contract.

*This file is the human-readable gap list; keep it updated when domains reach full four-ISA × full-dtype coverage.*
