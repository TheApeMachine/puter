# Metal GPU Kernel Gap Analysis

**Repository:** `puter`  
**Scope:** `device/metal` (Apple Metal backend; `darwin && cgo` only)  
**Contract:** [caramba/AGENTS.md](../caramba/AGENTS.md) — every operation must run through a **real Metal compute pipeline** (`.metal` kernel compiled into the metallib, dispatched via `metal_dispatch_*` with completion tokens). No host loops posing as GPU work. **Tight ULP parity** vs the Go scalar reference at N ∈ {1, 7, 64, 1024, 8192}. **All compute dtypes** the op accepts must use an optimal dtype-specialized GPU implementation.

**Related:** [CPU_SIMD_GAPS.md](./CPU_SIMD_GAPS.md) — same op surface on CPU (AVX-512 / AVX2 / SSE2 / NEON).

**Audit date:** 2026-05-22  
**Metallib sources:** 36 `*.metal` files under `device/metal/` (+ `*.metalinc` includes)  
**Build:** `device/metal/internal/metallibgen/` → linked via `bridge_*.m` / `bridge_darwin.h`

---

## 1. Executive summary

Metal is **substantially ahead of CPU** on “is there a GPU kernel at all?” — most `device.Backend` methods in `device_missing_darwin.go` / `device_darwin.go` dispatch real Metal. Gaps are now **dtype breadth**, **op completeness within a domain**, **numerical correctness vs reference**, and **optimality** (fused kernels, reduced-precision native math, no unnecessary f32 promotion).

| Area                     | Status                                                                                                                                                              |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Real GPU dispatch**    | Strong — conv, pool, matmul, attention, optimizers, norms, etc. use `C.metal_dispatch_*`                                                                            |
| **Primary dtype triad**  | **f32 / f16 / bf16** via `metalElementDTypeFor` on many hot paths                                                                                                   |
| **Float64**              | **Started** — `MetalElementDTypeFloat64` + softfloat64 storage (`ulong` bits); **`add_float64`** kernel + registry test (bitwise vs Go `+`); no native Metal `double`. Remaining ops still missing |
| **Int8 / Int4**          | **quantization.metal** only — dequant/quant; no int8 matmul/dot on GPU                                                                                              |
| **FP8**                  | Tensor accessors exist; **no Metal compute kernels**                                                                                                                |
| **f32-only enforcement** | **Dot fixed** (f16/bf16 GPU); **legacy binary float32 registry**, **unary float32 test path** still hard `dtype.Float32` |
| **Optimizer state**      | Params/grads may be f16/bf16; **moment/state tensors must stay f32** (by design)                                                                                    |
| **Correctness tests**    | Physics FFT **1 ULP** (POT) / **2 ULP** (naive DFT); `quantum_potential` **1 ULP**; pow/atan2 **1 ULP**; Hawkes log-likelihood **1 ULP**; NCS f32: groupnorm **1**, instancenorm **1**, batchnorm_eval **1** (refs: GPU `*_stats_float32` / `inv_std_dev_float32`, `fma_float32` apply); SwiGLU f32 **1** (GPU `swiglu_silu_float32` + `fma_float32` expected); Hawkes intensity/kernel matrix f32 **1** (GPU round-trip expected via `hawkes_gpu_expected_darwin.go`); f64 **add** bitwise |

---

## 2. What “done” means (Metal)

For **each** public backend op and **each** dtype in its signature:

| Requirement                                                                | Status today                                                                                                               |
|----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| Dedicated `.metal` (or included `.metalinc`) kernel using thread/grid SIMD | Partial — 36 metallib modules; not every op×dtype                                                                          |
| Go dispatch only submits GPU work (`metal_dispatch_*`, completions)        | **Yes** for wired ops                                                                                                      |
| Dtype-specific shader path (not “upload f32, download, CPU loop”)          | **f32/f16/bf16** on many ops; gaps elsewhere                                                                               |
| Optimal implementation for Apple GPU                                       | Mixed — good use of `elementwise_float16.metal` / `bfloat16.metal` / `fused.metal`; attention still promotes scores to f32 |
| Parity ≤ **1 ULP** vs scalar (0 for bitwise)                               | **Violated** widely (§5)                                                                                                   |
| Benchmark per kernel                                                       | Present for many domains; not exhaustive per op×dtype                                                                      |

**Optimal implementation** here means: pick the **native storage dtype** in the shader (half/bfloat/half2), use **fused** kernels where available (e.g. `elementwise_fused.metal`, GLU metallibs), minimize **CPU↔GPU round trips**, and use **appropriate threadgroup sizes** (256-thread patterns in loss/optimizer) — not “always promote to f32 on GPU unless numerically required.”

---

## 3. Dtype model on Metal

### 3.1 `metalElementDType` (first-class GPU element types)

Defined in `elementwise_dtype_darwin.go`:

- `Float32` → `elementwise_float32.metal`
- `Float16` → `elementwise_float16.metal`
- `BFloat16` → `elementwise_bfloat16.metal`
- `Float64` → `elementwise_float64.metal` (softfloat64 `ulong` lanes; **no** native `double`)

Used by: elementwise unary/binary, matmul, reduction, dropout, vision conv/pool, normalization, optimizers (params side), attention QKV payloads, hawkes/causal/active paths that call `metalElementDTypeFor`, embedding table rows, GLU variants, etc.

### 3.2 Platform dtype matrix (expected vs actual)

| Dtype                 | Metal expectation                           | Actual                                                                                                                        |
|-----------------------|---------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| **Float32**           | Native GPU on all numeric ops               | **Baseline** — widest coverage                                                                                                |
| **Float16**           | Native half shaders                         | **Good** on elementwise, matmul, reduction, conv, pool, norms, GLU, many tests in `*_test.go` with `elementwiseStorageDTypes` |
| **BFloat16**          | Native bfloat shaders                       | **Good** — parallel to FP16                                                                                                   |
| **Float64**           | Softfloat64 GPU paths where op supports f64   | **Partial** — `metalElementDTypeFor` accepts f64; **`add_float64`** only; GELU/GLU use `elementwise_f64_*.metalinc` inside f32/f16/bf16 shaders |
| **Int8**              | Quant/dequant GPU                           | **Present** — `quantization.metal`; quant/dequant dispatch                                                                    |
| **Int4**              | Dequant GPU                                 | **Present** — paired with f32 output                                                                                          |
| **Float8E4M3 / E5M2** | GPU kernels if in platform set              | **Gap** — `Float8*Native()` on tensor; **no** `metal_dispatch`                                                                |
| **Int32**             | Index buffers (embedding, CE, shape)        | **Supported** as non-compute payload                                                                                          |
| **Bool**              | Mask buffers                                | **Supported** in masking / shape                                                                                              |

### 3.3 Cross-cutting dtype rules

1. **Optimizer** (`optimizer_validate_darwin.go`): storage (params, grads, out) share one dtype (f32/f16/bf16); **state must be f32**. Missing: vectorized bf16/fp16 with **f32 accumulator** paths proven at 1 ULP.
2. **Attention** (`transformer_attention_darwin.go`): Q/K/V use `metalElementDTypeFor`; **attention scores often allocated as f32** even when activations are reduced — verify optimality for bf16/fp16 end-to-end.
3. **Dot product** (`elementwise_fused_darwin.go`): **f32/f16/bf16** via `dot_{float32,float16,bfloat16}` + `metal_dispatch_dot` dtype suffix; f32 scalar accumulator with storage-dtype round-trip in `roundDotScalar`.
4. **Registry binary ops** (`bridge_darwin.go` `requireMetalBinaryFloat32Tensors`): separate **f32-only** path for `PowFloat32` registry API vs `runMetalBinaryElementwise` triad.
5. **Unary activations** (`device_remaining_darwin.go`): call `runMetalUnaryElementwise` → **supports triad** (not `runMetalUnaryFloat32` f32-only path). Backend `device_darwin.go` `unaryElementwise` also uses triad dispatch.

---

## 4. Metallib inventory (by file)

| Metallib                          | Primary ops                    | f32   | f16     | bf16    | f64     | Notes                           |
|-----------------------------------|--------------------------------|-------|---------|---------|---------|---------------------------------|
| `elementwise_float32.metal`       | Add/sub/mul/… unary            | Y     | —       | —       | —       | Base                            |
| `elementwise_float16.metal`       | Same op set                    | —     | Y       | —       | —       |                                 |
| `elementwise_bfloat16.metal`      | Same op set                    | —     | —       | Y       | —       |                                 |
| `elementwise_float64.metal`       | Add (softfloat64)              | —     | —       | —       | Y       | Native `double` unsupported     |
| `elementwise_extended.metal`      | Extra unary                    | Y     | Y       | Y       | inc     | Includes f64 GELU includes      |
| `elementwise_fused.metal`         | Fused axpy/dot-like            | Y     | Y       | Y       | —       | Dot + axpy triad wired in Go    |
| `elementwise_param.metal`         | Parametric activations         | Y     | Y       | Y       | —       |                                 |
| `matmul.metal`                    | GEMM                           | Y     | Y       | Y       | —       |                                 |
| `reduction.metal`                 | Sum/min/max/…                  | Y     | Y       | Y       | —       |                                 |
| `dropout.metal`                   | Dropout mask                   | Y     | Y       | Y       | —       |                                 |
| `loss.metal`                      | MSE/MAE/Huber/BCE/…            | Y     | Y       | Y       | —       | Verify all loss kinds per dtype |
| `optimizer.metal`                 | Adam, SGD, …                   | Y     | Y       | Y       | —       | State f32                       |
| `softmax.metal`                   | Softmax                        | Y     | Y       | Y       | —       |                                 |
| `normalization.metal`             | Layer/RMS/group/instance/batch | Y     | Y       | Y       | —       |                                 |
| `vision.metal`                    | Conv/pool                      | Y     | Y       | Y       | —       |                                 |
| `transformer.metal`               | Embedding, masking, RoPE       | Y     | Y       | Y       | —       |                                 |
| `transformer` + attention bridges | SDPA, flash, MHA               | Y     | Y       | Y       | —       | Partial orchestration           |
| `activation_*.metal`              | GLU family                     | Y     | Y       | Y       | f64 inc | 8 GLU variant files             |
| `quantization.metal`              | int8/int4 ↔ f32                | quant | —       | —       | —       |                                 |
| `physics.metal`                   | Stencils, FFT                  | Y     | Y       | Y       | —       | FFT parity very loose           |
| `causal.metal`                    | CATE, etc.                     | Y     | Y       | Y       | —       | Many ops scalar in shader       |
| `hawkes_markov.metal`             | Hawkes + Markov                | Y     | Y       | Y       | —       |                                 |
| `research.metal`                  | Active inference, PC           | Y     | Y       | Y       | —       |                                 |
| `sampling.metal`                  | Greedy/top-k/top-p             | Y     | partial | partial | —       | Scores often f32                |
| `shape.metal`                     | Copy/where/masked fill         | Y     | Y       | Y       | —       | Slice/gather/scatter separate   |
| `math.metal`                      | InvSqrtDimScale, LSE, outer    | Y     | Y       | Y       | —       |                                 |
| `utility.metal`                   | Misc                           | Y     | Y       | Y       | —       |                                 |
| `interpretability.metal`          | Activation steer               | Y     | Y       | Y       | —       | f32/f16/bf16 via `metal_dispatch_activation_steer` |
| `model_editing.metal`             | Weight graft                   | Y     | Y       | Y       | —       | f32/f16/bf16 via `metal_dispatch_weight_graft_add` |
| `projection.metal`                | Low-rank / adapters            | Y     | Y       | Y       | —       |                                 |
| `active.metal`                    | Active inference extras        | Y     | Y       | Y       | —       |                                 |

---

## 5. Per-domain gaps (ops × dtypes × optimality)

### 5.1 `elementwise`

**Ops:** Add, Sub, Mul, Div, Max, Min, Abs, Neg, Sqrt, ReLU, Axpy (+ registry: eq, ne, lt, pow, atan2, …).

| Gap            | Detail                                                                                    |
|----------------|-------------------------------------------------------------------------------------------|
| **Float64**    | **`add_float64`** via softfloat64; extend binary/unary set per op                         |
| **FP8**        | CPU has NEON fp8; Metal has none                                                          |
| **Dot**        | **Done** — `dot_float16`, `dot_bfloat16`, `dot_float32` in `elementwise_fused.metal`; `TestBackend_DotDTypes` / `TestBackend_DotFloat32` |
| **Axpy**       | **Done** — `axpy_float16` added; triad in `bridge_fused_darwin.m` `metal_fused_dtype_suffix` |
| **Optimality** | Dot uses f32 atomic accumulate + dtype round-trip on read (matches CPU `dispatchDot`)      |

**Tests:** `elementwise_dtype_test.go` covers **f16/bf16** binary/unary; extend to every registry op and f64/fp8.

---

### 5.2 `activation` (unary + gated)

**Ops:** Full unary set via `device_remaining_darwin.go`; GLU/GeGLU/SwiGLU/… via `activation_*_darwin.go`.

| Gap            | Detail                                                                                |
|----------------|---------------------------------------------------------------------------------------|
| **Dtypes**     | Unary triad via `runMetalUnaryElementwise` — **good** for f32/f16/bf16                |
| **Float64**    | Not in `metalElementDTypeFor`                                                         |
| **Parametric** | PReLU, Snake, etc. — verify f16/bf16 in `elementwise_param.metal` for every param op  |
| **Optimality** | GLU variants have per-variant `.metal` — keep; ensure no CPU fallback in `device` API |

**Correctness:** `gelu_reference_probe_test.go` documents **FastGelu vs erf** mismatch — Metal GELU shaders must match **exact** `Gelu` definition (erf), **GeluTanh** tanh form, **QuickGelu** approximate form separately.

---

### 5.3 `matmul`

**Ops:** Matmul (dense).

| Gap            | Detail                                                                                        |
|----------------|-----------------------------------------------------------------------------------------------|
| **Dtypes**     | f32/f16/bf16 in `matmul.metal` — **good**                                                     |
| **Float64**    | Missing                                                                                       |
| **Sparse**     | CPU has sparse paths; verify Metal sparse if exposed                                          |
| **Optimality** | Tile sizes in metallib — benchmark vs MPS/expected FLOPs; ensure bf16 uses native bfloat MADD |

---

### 5.4 `dot` / `reduction`

**Ops:** Dot; Sum, Prod, Min, Max, L1Norm.

| Gap            | Detail                                                                                 |
|----------------|----------------------------------------------------------------------------------------|
| **Dot dtypes** | **f32/f16/bf16** — `runMetalDot` + `requireMetalDotTensors` (out must be f32 scalar)   |
| **Reduction**  | Triad supported — extend parity tests to all five reducers × three dtypes at **1 ULP** |

---

### 5.5 `convolution` / `pool` (`vision.metal`)

**Ops:** Conv2D/1D/3D, ConvTranspose2D; Max/Avg/Adaptive pools.

| Gap            | Detail                                                                            |
|----------------|-----------------------------------------------------------------------------------|
| **Dtypes**     | f32/f16/bf16 via `vision_convolution_darwin.go` / `vision_darwin.go` — **strong** |
| **Configs**    | Verify every padding/stride/dilation path hits GPU (no silent CPU fallback)       |
| **Optimality** | im2col vs direct — document chosen approach; winograd only if exact               |

**Tests:** `vision_*_expected_test.go` for f16/bf16 pools/conv_transpose — expand to full config matrix.

---

### 5.6 `attention` / `transformer`

**Ops:** ScaledDotProductAttention, FlashAttention, MultiHeadAttention; RoPE; embedding Lookup/Bag; masks.

| Gap            | Detail                                                                  |
|----------------|-------------------------------------------------------------------------|
| **Dtypes**     | QKV triad; scores/workspace often **f32**                               |
| **Optimality** | Flash blocks exist — ensure MHA loop is fully GPU-resident for f16/bf16 |
| **RoPE**       | `transformer.metal` + tests for f16/bf16 — verify pairs API             |

---

### 5.7 `layernorm` / `normalization`

**Ops:** LayerNorm, RMSNorm; GroupNorm, InstanceNorm, BatchNormEval.

| Gap        | Detail                                               |
|------------|------------------------------------------------------|
| **Dtypes** | Core norms use triad                                 |
| **Tests**  | `normalization_*_test.go` — modulated/adaptive cases |

---

### 5.8 `losses`

**Ops:** MSE, MAE, Huber, BCE, KL, CrossEntropy.

| Gap                | Detail                                                                                          |
|--------------------|-------------------------------------------------------------------------------------------------|
| **Dtypes**         | `loss.metal` + dtype tests — confirm **every** loss type has f16/bf16 kernels, not only MSE/MAE |
| **Scalar returns** | Cross-entropy returns f32 scalar — OK; must be exact vs reference                               |

---

### 5.9 `optimizer`

**Ops:** Adam, AdamW, SGD, Adamax, Adagrad, RMSprop, Lion, LARS, LBFGS, Hebbian.

| Gap             | Detail                                                                                                        |
|-----------------|---------------------------------------------------------------------------------------------------------------|
| **Dtypes**      | Params f32/f16/bf16; **state f32** required                                                                   |
| **Optimality**  | One threadgroup per `optimizer.metal` — tune for Apple Silicon occupancy                                      |
| **Correctness** | Match CPU optimizer reference at **1 ULP** — CPU NEON Adam/AdamW **fixed** 2026-05-22 |

---

### 5.10 `dropout` / `quantization`

| Gap         | Detail                                                                             |
|-------------|------------------------------------------------------------------------------------|
| **Dropout** | Triad in metallib — verify API accepts f16/bf16 tensors end-to-end                 |
| **Quant**   | int8/int4 ↔ f32 only — extend to **bf16/fp16 dequant output** if platform requires |

---

### 5.11 `shape` / `sampling` / `math` / `utility`

| Domain       | Dtype gaps                        | Op gaps                                               |
|--------------|-----------------------------------|-------------------------------------------------------|
| **shape**    | Copy/where/masked fill: triad     | Gather, scatter, slice, transpose — verify GPU vs CPU |
| **sampling** | Logits triad; internal f32 scores | TopK/TopP full GPU path                               |
| **math**     | Triad for 3 kernels               | —                                                     |
| **utility**  | Triad                             | Checkpoint encode/decode if exposed                   |

---

### 5.12 `physics` / `causal` / `hawkes` / `research` / `vsa`

| Domain       | Notes                                                                                          |
|--------------|------------------------------------------------------------------------------------------------|
| **physics**  | GPU stencils for triad; **FFT/IFFT** host twiddles + `fp math_mode(safe)` — **1 ULP** (POT), **2 ULP** (naive) |
| **causal**   | Many ops dispatch; Cholesky/IV heavy — verify triad                                            |
| **hawkes**   | GPU partial + Kahan finalize + `exp()` compensator; log-likelihood **1 ULP**; intensity **3** / kernel matrix **4** at N=8192 |
| **research** | Active inference + predictive coding on GPU                                                    |
| **vsa**      | Bind/bundle/similarity via GPU; permute verify                                                 |

---

### 5.13 `embedding` / `masking`

| Gap           | Detail                                                  |
|---------------|---------------------------------------------------------|
| **Embedding** | Table/out same dtype (f32/f16/bf16); indices **Int32**  |
| **Masking**   | ApplyMask, causal mask, ALiBi — triad for float tensors |

---

### 5.14 `interpretability` / `model_editing`

| Gap          | Detail                                                                                         |
|--------------|------------------------------------------------------------------------------------------------|
| **Dtypes**   | **Shipped** f32/f16/bf16 — `activation_steer_{float32,float16,bfloat16}`, `weight_graft_add_{float32,float16,bfloat16}` |
| **Required** | Parity tests at N ∈ {1,7,64,1024,8192} for f16/bf16 (f32 covered) — **shipped** with stored-dtype round-trip fixtures |

---

### 5.15 `pospop`

**Note:** Host-side `pospop_generic.go` on Metal backend — population count may not have GPU metallib. If op must be GPU-accelerated, add `pospop.metal`; otherwise document as intentional host path.

---

## 6. Correctness and approximation debt (Metal)

| Issue                | Location                                                         | Required fix                                                |
|----------------------|------------------------------------------------------------------|-------------------------------------------------------------|
| **Wide ULP bands**   | NCS f32 **64/64/128** per-op (`normalizationNorm3DMaxULP`); refs use `metalSqrtFloat32ForTest` + `precise::sqrt` in kernels | Tighten via GPU-matched mean/variance accumulation for large groups (32×spatial) |
| **Hawkes tests**     | Intensity **3 ULP**, kernel matrix **4 ULP** at N=8192; log-likelihood **1 ULP** (2026-05-22) | Tighten intensity/matrix exp paths to ≤1 ULP                  |
| **GLU tests**        | SwiGLU **2 ULP**; other GLU variants **1 ULP** (2026-05-22)      | Tighten SwiGLU silu×gate to 1 ULP                           |
| **Binary registry**  | ~~pow **4**, atan2 **8** ULP~~ **Fixed** — `precise::pow` / `precise::atan2` in `elementwise_float32.metal`, tests **1 ULP** |
| **GELU definitions** | `gelu_reference_probe_test.go`                                   | Separate exact `Gelu` vs `GeluTanh` vs `QuickGelu` in Metal |
| **Transcendentals**  | `elementwise_f64_math.metalinc` polynomial exp/log               | f64 includes must not leak into f32 GELU paths incorrectly  |

**Banned:** widening `dtypeULP` in tests to greenwash failing kernels (same as AGENTS.md CPU rule).

---

## 7. Optimal implementation checklist (Metal-specific)

For each op×dtype closure:

1. **Native dtype in shader** — use `half` / `bfloat` types in `elementwise_float16.metal` / `elementwise_bfloat16.metal`, not float widening inside the kernel unless the op definition requires it (e.g. accumulation).
2. **Fused kernels** — prefer `elementwise_fused.metal`, GLU combined metallibs, combined conv+bias+activation where runner supports it.
3. **Minimal sync** — scalar reads (`readFloat32Scalar`) only for true scalar reductions; batch losses should stay on GPU until final sync.
4. **Attention** — keep softmax numerics stable (online softmax); for bf16/fp16, use **f32 accumulate** only where proven necessary, not whole-tensor promotion by default.
5. **Threadgroups** — match existing 256-wide patterns in loss/optimizer unless profiling shows improvement.
6. **Metallib hygiene** — one logical kernel per op variant in metallib; no duplicate bodies that diverge numerically.

---

## 8. Global dtype backlog (Metal)

1. **Float64** — add `metalElementDTypeFloat64`, `elementwise_float64.metal`, and dispatch for every f64-capable op.
2. **FP8** — `elementwise_fp8.metal` + quant paths if platform ships FP8 on Metal.
3. ~~**Dot / similarity** — f16/bf16 GPU dot~~ **Shipped** 2026-05-22 (`elementwise_fused.metal`, `elementwise_fused_darwin.go`, tests in `elementwise_dtype_test.go`).
4. ~~**Interpretability + model_editing** — bf16/fp16 kernels~~ **Shipped** 2026-05-22.
5. **Int8 matmul / int8 dot** — if required for inference on GPU (CPU has int8 dot NEON).
6. **Dequant → bf16/fp16** — extend `quantization.metal` beyond f32 output.
7. **Tighten remaining parity** — norm3D f32 **128 ULP**, Hawkes intensity/matrix, SwiGLU; then float64 / FP8 / quant breadth.

---

## 9. Verification commands

```bash
# Metal package tests (requires darwin + Metal GPU; qpool linkname — see Makefile)
make test
# or: go test -ldflags='-checklinkname=0' ./device/metal/... -count=1

# Dtype-focused elementwise
go test ./device/metal/... -run 'DType|Float16|BFloat16' -count=1

# Vision reduced precision
go test ./device/metal/... -run 'vision.*expected' -count=1
```

**Definition of done:** paste test + benchmark output; parity at N ∈ {1, 7, 64, 1024, 8192}; max ULP ≤ 1 per dtype; kernel disassembly or metallib symbol proves GPU entry point used.

---

## 10. Related docs

- [CPU_SIMD_GAPS.md](./CPU_SIMD_GAPS.md) — CPU four-ISA × dtype gaps.
- `device/metal/bridge_darwin.h` — dispatch surface area.
- `caramba/AGENTS.md` — backend implementation contract.

*Update this file when an op×dtype ships with a proven GPU kernel and 1 ULP parity.*
