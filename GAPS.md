## Dispatch audit overstates coverage

- **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/cpu/dispatchaudit/matrix.go`** (lines 136–138): registration means “at least one assembly file or dispatch-table entry exists … **does not assert full operation coverage**.”
- **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/cpu/dispatchaudit/matrix_test.go`** (lines 26–43): asserts **32/32 domains** registered for AVX-512, AVX2, SSE2, and NEON. That passes while many ops/dtypes inside those domains remain scalar-only.
- Conflicts with AGENTS.md §1: equal standing for all CPU ISAs per operation **and** per `dtype.DType`.

---

## Same-ISA entry aliasing (contract risk)

- **`activation/param_sse2_amd64.s`** line 33: `PReLUF32SSE2` → `JMP ·LeakyReLUSlopeF32SSE2(SB)`
- **`activation/param_avx2_amd64.s`** line 46: `PReLUF32AVX2` → `JMP ·LeakyReLUSlopeF32AVX2(SB)`
- **`activation/param_avx512_amd64.s`** line 70: `PReLUF32AVX512` → `JMP ·LeakyReLUSlopeF32AVX512(SB)`
- Separate declared symbols share one assembly body (same ISA, different ops). PReLU with a scalar slope may be mathematically identical to LeakyReLU, but this is still aliasing under the “each `.s` file contains its own kernel” rule.

---

## `dtype.DType` coverage gaps (panic or scalar-only)

| Path                  | Evidence                                                                                                                                                                                                                        |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FP8 elementwise       | **`elementwise/fp8.go`** lines 49–52, 84–87: scalar `for index := range left` widen-to-f32 loops; comment lines 16–17 admits no native FP8 ISA. Only **`elementwise/fp8_test.go`**.                                             |
| Dropout               | **`dropout/ops.go`** line 21: `panic("dropout: only dtype.Float32 is implemented")`.                                                                                                                                            |
| Quant / dequant       | **`quant/ops.go`** 20–21: only `Int8 ← Float32`. **`dequant/ops.go`** 21–22, 49–50: only `Float32 ← Int8/Int4`.                                                                                                                 |
| Sampling              | **`sampling/ops.go`** 9–12: `requireSamplingFloat32` panics otherwise.                                                                                                                                                          |
| Hawkes                | **`hawkes/ops.go`** 9–12: f32 only.                                                                                                                                                                                             |
| Predictive coding     | **`predictive_coding/ops.go`** 9–12: f32 only.                                                                                                                                                                                  |
| Causal                | **`causal/ops.go`** 10–13: f32 only; Cholesky (lines 30–37) is scalar Go triple loop despite domain SIMD registration.                                                                                                          |
| RoPE pairs API        | **`rope/ops.go`** 9–12: `RoPEPairs` f32-only; full **`rope/compute.go`** 55–94 supports bf16/fp16 via scalar `ropePairsTyped`.                                                                                                  |
| Float64 elementwise   | **`elementwise/ops.go`** 9–17: only `Add` handles `Float64`; Sub/Mul/Div/etc. do not. amd64 **`elementwise/select_amd64.go`** 433: `addF64Funcs` = Generic only; arm64 has **`elementwise/f64_neon_arm64.s`** but only for Add. |
| Int/uint/complex/bool | No native CPU kernels found under `device/cpu` for `Int64`, `Uint64`, `Complex64/128`, `Bool`, etc.                                                                                                                             |

---

## Declared SIMD vs actual native path (amd64/arm64 asymmetry)

- **Active inference bf16/fp16 on amd64:** **`active_inference/select_amd64_typed.go`** 6–30 — dispatch tables are **Generic-only**. arm64 **`active_inference/select_arm64_typed.go`** 8–40 registers NEON + Generic.
- **Matmul f64 on amd64:** **`matmul/select_amd64.go`** 29–30 → `matmulFloat64Scalar`. arm64 **`matmul/select_arm64.go`** 36–48 uses `MatmulRowFloat64NEONAsm`.
- **Sparse CSR matmul on amd64:** **`matmul/select_amd64.go`** 38–42 → **`matmul/sparse_scalar.go`**. arm64 **`matmul/select_arm64.go`** 90–95 → **`matmul/sparse_f32_neon_arm64.s`**.
- **Attention flash:** **`attention/typed_compute.go`** 38–47: f32 → `RunFlashAttentionRowNative`; bf16/fp16 (lines 52–79) → scalar Go flash loop with `make([]float32, valueDim)`.
- **LayerNorm bf16/fp16:** f32 row apply uses SIMD via **`layernorm/select_amd64.go`**; bf16/fp16 **`layernorm/compute.go`** 208–236 `applyLayerNormRowBF16/F16` are per-element scalar loops.
- **Embedding bf16/fp16 lookup:** **`embedding/compute.go`** 72–97 `runLookupReduced` — nested scalar index/dim loops (no row SIMD); f32 uses SIMD row copy in **`embedding/select_amd64.go`**.
- **RoPE bf16/fp16:** **`rope/compute.go`** 67–94 `ropePairsTyped` — scalar per-pair; f32 uses `RopePairsNative`.
- **Optimizer mixed precision:** **`optimizer/optimizers_dtype.go`** 80–86 — per-lane `loadParam`/`storeOut` scalar callbacks; bf16/fp16 never hit **`optimizer/f32_avx512_amd64.s`** etc.
- **Physics multi-D:** **`physics/select_arm64.go`** — 1D stencil NEON (`Laplacian1DStencilF32NEONAsm`); 2D/3D (lines 50–117) are Go nest loops over 1D calls.
- **Losses bf16/fp16:** **`losses/compute.go`** 34–38 — only f32 hits `runMSEF32`; others → `mseTyped` scalar loader path.

---

## Dispatch bug / silent scalar miss

- **FP16 matmul skips SSE2 on SSE2-only CPUs:** **`matmul/select_amd64_reduced.go`** 70–72:
  ```go
  if !(cpu.X86.HasAVX2 || cpu.X86.HasAVX512F) {
      runMatmulReduced(...) // scalar
      return
  }
  ```
  **`MatmulRowFP16SSE2Asm`** is declared (line 27) but unreachable when CPU has SSE2 but not AVX2/AVX-512.
- **Silent scalar fallbacks (by design, not error):** **`matmul/select_amd64.go`** 26, **`matmul/select_arm64.go`** 25–33 (column tail), **`optimizer/select_amd64.go`** (each step ends in `*Scalar`), **`vsa/select_simd_amd64.go`** 28–30 / 54 / 78 when no x86 SIMD.

---

## Panic dispatch paths (hard failures vs missing ISA)

Pick helpers panic when no `available` candidate remains, e.g. **`elementwise/f32_pick.go`** 54, **`reduction/f32_pick.go`** 18, **`shape/f32_pick.go`** 90–154, **`activation/gated_packed_pick.go`** 18, **`checkpoint/f32_pick.go`** 24–36, **`active_inference/typed_pick.go`** 60–142.

Public ops also panic on dtype: **`masking/ops.go`** 37/78/99, **`attention/compute.go`** 27/50, **`embedding/compute.go`** 49/68/111, **`losses/compute.go`** 21, **`layernorm/compute.go`** 30/51, **`normalization/compute.go`** 30.

---

## Test / benchmark / tolerance gaps

**Missing or thin ISA parity (vs AGENTS §2 N ∈ {1,7,64,1024,8192}, all ISAs):**

- **No `*_avx2_sse2_parity_test.go`:** activation, dot, dropout, elementwise, losses, matmul, pool, reduction, vsa, embedding, hawkes, physics, normalization, pospop (18 domains have it; these do not).
- **No dedicated `*_neon_parity_test.go`:** activation, attention, causal, convolution, dequant, dot, dropout, elementwise, layernorm, losses, matmul, optimizer, pool, quant, reduction, rope, vsa (16 domains have neon parity files; these rely on ad-hoc `*_neon_arm64_test.go` or nothing).
- **Hot paths with AVX-512-only parity suites:** **`dot/dot_avx512_parity_test.go`**, **`elementwise/elementwise_avx512_parity_test.go`**, **`matmul/matmul_avx512_parity_test.go`** (f32 only), **`losses/losses_avx512_parity_test.go`**, **`pool/pool_avx512_parity_test.go`**, **`reduction/reduction_avx512_parity_test.go`**, **`activation/activation_avx512_parity_test.go`** — no symmetric AVX2/SSE2/NEON parity files for the same ops.

**Loose tolerances (AGENTS bans wide epsilons):**

- **`matmul/matmul_fp16_test.go`** 64: `tolerance := … * 0x1p-10` (K-scaled, not ULP).
- **`losses/losses_avx512_parity_test.go`** 40, **`reduction/reduction_avx512_parity_test.go`** 38: `length * 0x1p-50` absolute tolerance for reductions.
- **`reduction/reduction_avx512_parity_test.go`** 65–66: prod allows **ULP gap > 16**.
- **`interpretability/test_helpers_test.go`** 31: `> 1e-6` absolute epsilon.
- **`activation/activation_avx512_parity_test.go`** 29–45: many activations tested at **maxULP 2** (exp, log, gelu, etc.).

**Benchmark gaps:** Many domains have AVX-512 or scalar benches only; e.g. **`vsa/vsa_scalar_bench_test.go`** without matching AVX2/SSE2/NEON bench mirrors for all ops. **`elementwise/fp8.go`** has tests but no ISA benchmarks (scalar-only implementation).

---

## Performance left on the table

- **`embedding/select_amd64.go`** 68–80: outer loop over `indexCount` tokens; SIMD only on `copyRowF32*` / `addRowF32*` inner hidden dim — gather still scalar-indexed.
- **`matmul/select_amd64.go`** 45–54: reference f32 matmul is naive ijk scalar (no blocking/tile) when SIMD unavailable.
- **`matmul/select_amd64_reduced.go`** FP16 SSE2 path unreachable (above) — SSE2-only hosts stay on **`compute.go`** `runMatmulReduced`.
- **`matmul/select_arm64.go`** / **`select_amd64_reduced.go`**: column tails (`cols &^ align`) finished with **`runMatmulReducedCols`** scalar triple loop.
- **`causal/ops.go`**: Cholesky and related ops remain scalar Go despite **`causal/f32_avx512_amd64.s`** etc. for other kernels.
- **`convolution/select_amd64_reduced.go`** 377–400: Conv3D bf16/fp16 → **`Conv3DTypedScalar`** when `!reducedFloatSIMDAvailable()`.
- **`pool/select_amd64.go`** / **`select_arm64.go`**: `PoolWindowMaxScalar` / `PoolWindowAvgScalar` for window max/avg paths.
- **`layernorm/compute.go`** 173–179: RMSNorm f32 allocates `combined := make([]float32, len(row))` per row before `MulFloat32Native`.
- **`dropout/select_arm64.go`** 24–28: 1–3 lane scalar epilogue after NEON block.

---

## Summary severity

1. **Contract:** `dispatchaudit` + tests certify full domain×ISA registration while AGENTS requires per-op, per-dtype native kernels on all ISAs; FP8, most integer/complex dtypes, and many f32-only domains fail that bar.
2. **Correctness risk:** FP16 matmul SSE2 dispatch hole; PReLU/LeakyReLU aliasing; reduction/loss tests use non-ULP tolerances.
3. **Coverage risk:** Large holes in AVX2/SSE2/NEON parity vs AVX-512; bf16/fp16/f64 paths often scalar on amd64 where arm64 has NEON.
4. **Performance:** Hybrid scalar tails, per-token embedding gather, scalar flash-attention for reduced precision, and unreachable SIMD paths.

---

## Host-side computation behind Metal APIs

1. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/pospop_host_darwin.go`** — Darwin `Backend.Count8/16/32/64` call `pospopCount*Generic` in `pospop_generic.go` (plain Go loops). No Metal kernel or dispatch; pospop runs on CPU while `Location()` is Metal.
2. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/device_dispatch_darwin.go:64-94`** — `readFloat32Scalar` / `readInt32Scalar` call `syncTensor` then `Download` (host memcpy). Used by scalar-returning backend ops (`reductionScalar`, `pairLossScalar`, `dotProduct`, etc.).
3. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/device_missing_darwin.go:179-186,466-468`** — `CrossEntropy` and similar paths: GPU kernel → `emptyScalar` → `readFloat32Scalar` (sync + full download for one float).
4. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/device_dispatch_darwin.go:218-241`** + **`device_missing_darwin.go`** (many call sites) — `uploadFloat32Scalar` / `uploadInt32Scalar` host-encode scalars and `Upload` a 1-element buffer per kernel parameter (spacing, slope, Hawkes μ/α/β, etc.) instead of constant buffers.
5. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/physics_darwin.go:122-157`** — Non-power-of-2 FFT: host builds O(n²) twiddle tables via `naiveFFTTwiddles` in `physics_fft_reference.go`, uploads two full GPU buffers per call, then closes them after dispatch.
6. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/backend.go:72-73` vs `bridge_darwin.go:78-116`** — Docs claim unsupported dtypes are converted at upload; `upload`/`copyBytes` only `memcpy` into MTLBuffer with no conversion. Host conversion must happen before `Upload`; `empty()` accepts any dtype `shape.Bytes` allows (including Float64) regardless of `SupportedDTypes()`.

---

## Stubs / missing Darwin implementations

7. **25 `*_stub.go` files** (e.g. `matmul_stub.go`, `activation_swiglu_stub.go`, `normalization_stub.go`, `bridge_stub.go`) — `//go:build !darwin || !cgo`; all return `ErrNeedsPlatformSetup` or panic. Expected for cross-compile, but entire Metal surface is non-functional off Darwin.
8. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/backend.go:161-171`** — `UploadSparse` always returns `tensor.ErrLayoutUnsupported` (“Phase 9” comment).
9. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/pospop_host_darwin.go`** — Darwin build exists but still has no GPU implementation (see #1).
10. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/device_missing_darwin.go:86-131`** — `runMetalCholesky` dispatches `metal_dispatch_cholesky`; **no `_test.go` references `Cholesky`/`cholesky` anywhere in the package**.

---

## Dtype coverage gaps (vs AGENTS.md “all `dtype.DType` natively”)

11. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/backend.go:75-84`** — `SupportedDTypes()` lists only Float32, BFloat16, Float16, Int32, Int8, Int4, Bool. Missing Float64, Float8E4M3/E5M2, Int64/Int16, all Uint*, Complex64/128.
12. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/kernels.go:181-183`** — `registerBinaryFloat64Kernels()` registers only `"add"` for Float64; no sub/mul/div/unary registry.
13. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/elementwise_float64.metal`** — Only `add_float64` kernel defined; no other f64 elementwise kernels in `.metal`.
14. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/bridge_darwin.go:737-741`** — `Float8E4M3Native` / `Float8E5M2Native` exist on `metalTensor`; no Float8 `.metal` kernels or tests in package.
15. **Op-family dtype lists** — Most kernels limited to `{Float32, Float16, BFloat16}` via vars like `metalNormalizationDTypes` (`normalization.go:9-13`), `metalGLUDTypes` (`activation_glu.go:9-13`), `metalTransformerDTypes` (`transformer.go:9-13`), `matmul.go` (F32/F16/BF16 only). No Float64 matmul/normalization/attention paths.
16. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/device_darwin.go:155-183`** — `binaryElementwise` / `unaryElementwise` ignore the `format dtype.DType` argument (`_ = format`); dtype comes only from tensor handles. Wrong `format` with mismatched tensors fails silently or at tensor level, not at API boundary.

---

## CPU reference in tests only / circular GPU gold

17. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/hawkes_intensity_expected_test.go:68-85,83-85`** — Expected values from `hawkesIntensityMetalExpected` (`hawkes_gpu_reference_darwin_test.go:15-54`), which re-runs the same GPU kernel. Parity for Float32 and F16/BF16 is GPU-vs-GPU, not vs Go scalar reference.
18. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/hawkes_kernel_matrix_expected_test.go`** / **`hawkes_gpu_reference_darwin_test.go:60-95`** — Same pattern for `hawkes_kernel_matrix`.
19. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/activation_swiglu_expected_test.go:70-81`** — `swiGLUExpectedFloat32ForTest` uses `metalSiluFloat32Vector` + `metalFMAFloat32Vector` (GPU kernels from `parity_fma_reference_darwin_test.go`) for **all** dtypes including Float32; not `device/cpu` scalar.
20. **Contrast (good)** — `activation_geglu_expected_test.go:19` uses `cpuactivation.GeGLUTensorsF32Generic`; `hawkes_log_likelihood_expected_test.go:27,52` uses `cpuhawkes.HawkesLogLikelihoodScalar`; vision pooling `*_expected_test.go` use `device/cpu/pool` / `device/cpu/convolution`.
21. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/normalization_*_gpu_parity_test.go`** — Reference is Go serial + GPU FMA/sqrt helpers (`parity_fma_reference_darwin_test.go`), not independent CPU backend scalar for the full norm kernel.

---

## Device submission overhead

22. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/bridge_darwin.m:110-125,128-132`** — When not batching, each `metal_get_encoder` allocates a new `commandBuffer`; `metal_end_encoder` ends encoding and `[commandBuffer commit]` per dispatch.
23. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/bridge_elementwise_darwin.m:242-245`** — Typical pattern: `addCompletedHandler` + `metal_end_encoder` → one commit per elementwise dispatch.
24. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/backend.go:217-222`** — `BeginBatch`/`EndBatch` exist; **only production-adjacent use** is `vision_test.go:107-108`. Normal kernel paths never batch.
25. **Every `*_darwin.go` dispatch** — `metalCompletions.Begin`/`BeginMany` per kernel (e.g. `elementwise_dtype_darwin.go:66`, `matmul_darwin.go:42`); completion registry tracks pending tensors but does not coalesce command buffers without explicit batching.

---

## Threadgroup sizing

26. **Single-threadgroup scalar/reduction dispatches** — `dispatchThreadgroups:MTLSizeMake(1, 1, 1)` in:
   - `bridge_reduction_darwin.m:186-187` (256 threads in one TG)
   - `bridge_active_darwin.m:185`
   - `bridge_loss_darwin.m:289`
   - `bridge_causal_scalar_darwin.m:107`
   - `bridge_hawkes_markov_scalar_darwin.m:86,141`
   - `bridge_sampling_common_darwin.m:142`
   - `bridge_active_extra_darwin.m:128`
   - `bridge_causal_dag_darwin.m:81`  
   Global work is serialized into one threadgroup; poor GPU utilization at scale.
27. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/bridge_matmul_darwin.m:122-123`** — 16×16 threadgroups (reasonable contrast).
28. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/elementwise_bfloat16.metal:54-55,82-83`** — Vector path plus scalar tail loops for remainder elements (not full-vector remainder handling).

---

## Memory transfers / synchronization

29. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/bridge_darwin.go:109-116,192-214`** — Upload/download are synchronous host `memcpy` on shared MTLBuffers; every `Download` waits on `target.Sync`.
30. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/device_dispatch_darwin.go:96-108`** — `reductionScalar`: GPU reduction kernel + scalar download per API call.
31. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/physics_darwin.go:122-157`** — Non-PoT FFT: allocates and uploads 2×n² float32 twiddles per transform (memory + transfer spike).
32. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/bridge_darwin.go:143-149,288-293`** — `UploadAsync` copies bytes on host (`append`), spawns goroutine for `finishAsyncUpload`; still host-side memcpy, not GPU DMA pipeline.

---

## Benchmark / parity coverage gaps

33. **`page_write` / `page_gather`** — Registered in `shape.go:40-70`, implemented in `shape_page_darwin.go` + `shape.metal:716-722`; **zero `_test.go` or `*_bench_test.go` references** (grep across package).
34. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/elementwise_float64_test.go`** — Parity only for `add` + Float64; **no `elementwise_float64_bench_test.go`**.
35. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/normalization_*_gpu_parity_test.go`** — Fixed cases (e.g. `spatial := 1024` in `normalization_batchnorm_gpu_parity_test.go:20`); **do not sweep** `parityElementCounts = {1,7,64,1024,8192}` from `backend_test.go:17`.
36. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/normalization_layernorm_gpu_parity_test.go`** — Only `TestLayerNormGPUVersusSerialReferenceN7`, `Cols14`, `GroupSliceN7`; not full N sweep.
37. **Wide ULP budgets (AGENTS: tight ULP, no widening)**:
   - `normalization_test.go:12-15` — `normalizationFloat32MaxULP = 32`, `normalizationNorm3DFloat32MaxULP = 64`
   - `softmax_test.go:14` — `softmaxFloat32MaxULP = 64` (softmax **does** sweep `parityElementCounts`)
   - `elementwise_unary_extended_test.go:24-37` — several ops allow 8–16 ULP for f32
38. **Cholesky** — Implemented on Darwin (`device_missing_darwin.go:86-131`); no parity or benchmark tests.
39. **`gelu_reference_probe_test.go`** — Probes CPU `FastGelu32` vs `math.Erf`; does **not** assert Metal `gelu` kernel vs Go scalar reference (unary extended tests use `cpumath.FastGelu32` at `elementwise_unary_extended_test.go:237-238`).
40. **Float64 registry vs metal** — `kernels.go:182` wires Float64 `add` through `runBinaryElementwise(metalBinaryFloat32Add)`; bridge resolves f64 kernel names (`bridge_elementwise_darwin.m:98`), but only `add` is registered/tested—remaining f64 ops are uncovered.

---

## Additional contract notes

41. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/device_darwin.go:50-65`** — `Matmul` ignores `rows/inner/cols/format`; on `tensorsAt` error it **returns silently** (no panic/error), unlike `devicePanic` used elsewhere.
42. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/activation_geglu_tanh.metal:30-31`** — Comment: “tanh GELU — matches … FastGeluTanh32”; explicit approximate variant (OK if op is defined that way). Standard `gelu` in `elementwise_f64_math.metalinc:370-376` uses softfloat `erf` (exact form).
43. **`/Users/theapemachine/go/src/github.com/theapemachine/puter/device/metal/normalization.metal:17-21`** — `refined_inv_sqrt_norm`: one Newton step on `precise::sqrt` (refinement, not a separate approximate kernel).
44. **Non-Darwin stub surface** — `device_backend_stub.go` + `device_backend_stub_ops.go`: every `Backend` method calls `deviceNeedsPlatform()` → panic on `!darwin||!cgo`; Metal package compiles everywhere but is not a real backend off Apple.

---

## Resolution Plan

This section is an execution contract. Complete each numbered work packet in order. Do not mark a packet complete until every acceptance gate listed under that packet is satisfied. Do not replace a missing native kernel with a scalar loop, a generic Go path, a code generator, a table-driven emitter, a shared assembly body, a Python script, a shell script, a C macro, a widened tolerance, or a test that compares a backend to itself.

### Global Completion Rules

1. Every operation exposed by `device.Backend` must have native implementations for:
   - Go scalar reference.
   - AVX-512 assembly on amd64.
   - AVX2 assembly on amd64.
   - SSE2 assembly on amd64.
   - NEON assembly on arm64.
   - Metal on darwin+cgo.
2. Every operation must support every `dtype.DType` that can be stored by the backend. Do not shrink `SupportedDTypes()` to close a kernel gap.
3. Each assembly entry point must contain the math for its own operation in its own `.s` file. An entry point must not jump to another operation, call another operation's body, or share one body with another declared operation.
4. Each Metal operation must dispatch a real Metal compute kernel. Host code may validate shapes, encode scalar constants, enqueue work, and read explicitly scalar return values; it must not compute tensor results on the host for a Metal operation.
5. Every parity test must compare the target backend against the Go scalar reference for the same operation and dtype. GPU-vs-GPU gold values are not parity tests.
6. Every backend kernel parity suite must run `N = {1, 7, 64, 1024, 8192}` or the shape-equivalent set for multi-dimensional kernels.
7. Every parity assertion must use bitwise equality or a named tight ULP bound justified by the scalar reference's operation order. Do not use arbitrary absolute epsilon, length-scaled epsilon, or widened budgets to pass a failing kernel.
8. Every native kernel must have a benchmark that runs the scalar reference and each native backend on the same input shape and dtype.
9. Verification output must be pasted into the final completion message for each packet: exact `go test` commands, exact benchmark commands, and the passing output.

### Packet 1: Replace Dispatch Audit With Operation-Level Coverage

Modify:
- `device/cpu/dispatchaudit/matrix.go`
- `device/cpu/dispatchaudit/matrix_test.go`
- Create `device/cpu/dispatchaudit/operations.go`.
- Create `device/cpu/dispatchaudit/operations_test.go`.

Steps:
1. Replace domain-level registration with an operation-level matrix keyed by operation name, dtype, and execution target.
2. Record one entry per actual callable kernel symbol or Metal kernel registration. A package-level dispatch table entry is not enough.
3. Add explicit fields for `ScalarGo`, `AVX512`, `AVX2`, `SSE2`, `NEON`, and `Metal`.
4. Fail the audit when any operation/dtype pair has a missing native backend entry.
5. Fail the audit when an amd64 SIMD entry points to a generic Go function or another ISA's symbol.
6. Fail the audit when a Metal entry is implemented by a host-only Darwin function.

Acceptance gates:
- `go test ./device/cpu/dispatchaudit`
- The test output must fail before missing kernels are implemented.
- After all packets below are complete, the same command must pass without suppressions.

### Packet 2: Remove Same-ISA Entry Aliasing

Modify:
- `device/cpu/activation/param_sse2_amd64.s`
- `device/cpu/activation/param_avx2_amd64.s`
- `device/cpu/activation/param_avx512_amd64.s`
- Matching activation parity and benchmark tests.

Steps:
1. Replace `PReLUF32SSE2`, `PReLUF32AVX2`, and `PReLUF32AVX512` jumps with complete operation bodies.
2. Keep PReLU and LeakyReLU symbols separate even when scalar-slope math is equivalent.
3. Add direct parity tests that call each PReLU symbol, not only the selected dispatch function.
4. Add direct benchmarks for each PReLU symbol and matching LeakyReLU symbol.

Acceptance gates:
- `go test ./device/cpu/activation -run 'Test.*PReLU.*(AVX512|AVX2|SSE2)'`
- `go test ./device/cpu/activation -bench 'Benchmark.*PReLU.*(AVX512|AVX2|SSE2)' -run '^$'`
- Assembly inspection must show no `JMP ·LeakyReLUSlope` from any PReLU symbol.

### Packet 3: Complete CPU Dtype Coverage

Modify the packages listed in this file under `dtype.DType coverage gaps`, including:
- `device/cpu/elementwise`
- `device/cpu/dropout`
- `device/cpu/quant`
- `device/cpu/dequant`
- `device/cpu/sampling`
- `device/cpu/hawkes`
- `device/cpu/predictive_coding`
- `device/cpu/causal`
- `device/cpu/rope`
- `device/cpu/attention`
- `device/cpu/layernorm`
- `device/cpu/embedding`
- `device/cpu/optimizer`
- `device/cpu/losses`

Steps:
1. For each public operation, list every dtype accepted by `dtype.DType` and the current code path used by that dtype.
2. Implement missing scalar Go references first. The scalar reference is the oracle and must be simple, direct, and dtype-correct.
3. Implement AVX-512, AVX2, SSE2, and NEON assembly for each dtype that interprets values.
4. Use byte-level copying only for pure data movement operations.
5. Implement bf16 and fp16 math with distinct code paths. Do not share bf16 and fp16 math bodies.
6. Implement integer, unsigned integer, bool, float8, and complex kernels for every operation currently exposed through `device.Backend`.
7. Remove dtype panics from public operations only after the corresponding scalar and native kernels exist.

Acceptance gates:
- `go test ./device/cpu/...`
- `go test ./device/cpu/... -run 'Parity|DType|Native'`
- `go test ./device/cpu/... -bench 'Benchmark' -run '^$'`
- No public operation may panic only because a dtype lacks a native path when that dtype is advertised by the backend.

### Packet 4: Complete CPU ISA Symmetry

Modify all packages listed under `Declared SIMD vs actual native path`.

Steps:
1. Replace amd64 generic-only bf16/fp16 active-inference dispatch with AVX-512, AVX2, and SSE2 assembly.
2. Replace amd64 f64 matmul scalar selection with AVX-512, AVX2, and SSE2 row kernels.
3. Replace amd64 sparse CSR matmul scalar selection with AVX-512, AVX2, and SSE2 kernels.
4. Replace bf16/fp16 flash-attention scalar loops with AVX-512, AVX2, SSE2, and NEON kernels.
5. Replace bf16/fp16 layernorm scalar rows with AVX-512, AVX2, SSE2, and NEON kernels.
6. Replace bf16/fp16 embedding lookup scalar inner loops with native row copy/add kernels for every ISA.
7. Replace bf16/fp16 RoPE scalar pair loops with native kernels for every ISA.
8. Replace mixed-precision optimizer scalar callbacks with dtype-specific native kernels for every ISA.
9. Replace f32-only losses with dtype-specific native kernels for every ISA.

Acceptance gates:
- Each modified package must have `*_avx512_parity_test.go`, `*_avx2_sse2_parity_test.go`, and `*_neon_parity_test.go` coverage for the same operation/dtype set.
- Each modified package must have scalar and native benchmarks for every operation/dtype set.
- The package-level selector files must not register a generic candidate for an operation/dtype/ISA pair that has a required native implementation.

### Packet 5: Fix FP16 Matmul SSE2 Dispatch

Modify:
- `device/cpu/matmul/select_amd64_reduced.go`
- `device/cpu/matmul/*fp16*_test.go`

Steps:
1. Route SSE2-only hosts to `MatmulRowFP16SSE2Asm`.
2. Add a direct test for `MatmulRowFP16SSE2Asm` at inner sizes `1, 7, 64, 1024, 8192`.
3. Extract an unexported `selectMatmulReducedKernel(capabilities cpuCapabilities)` helper and add a selector test that passes `{HasSSE2: true, HasAVX2: false, HasAVX512F: false}`.
4. Add an SSE2 benchmark that compares `MatmulRowFP16SSE2Asm` against the scalar reference.

Acceptance gates:
- `go test ./device/cpu/matmul -run 'FP16.*SSE2|SSE2.*FP16'`
- `go test ./device/cpu/matmul -bench 'FP16.*SSE2|SSE2.*FP16' -run '^$'`
- The SSE2 branch must be reachable without AVX2 or AVX-512 capability bits.

### Packet 6: Remove Silent Error Suppression In Metal Backend Methods

Modify:
- `device/metal/device_darwin.go`
- `device/metal/device_remaining_darwin.go`
- `device/metal/device_missing_darwin.go`
- Tests that call `Backend` methods through resident pointers.

Steps:
1. Replace every `if err != nil { return }` in backend methods with `devicePanic(err)`.
2. Replace every ignored `_ = runMetal...` result with `devicePanic(runMetal...)`.
3. Validate `format dtype.DType` against the resident tensor dtype in every unsafe-pointer backend method.
4. Add tests that pass an invalid resident pointer and assert the method panics with the expected error.
5. Add tests that pass a mismatched `format` and assert the method panics with `tensor.ErrDTypeMismatch`.

Acceptance gates:
- `go test ./device/metal -run 'Backend.*Panic|DTypeMismatch|Resident'`
- `go test ./device/metal`
- No Darwin backend method may silently return after a validation or dispatch error.

### Packet 7: Replace Metal Host-Computed Tensor Operations

Modify:
- `device/metal/pospop_host_darwin.go`
- `device/metal/pospop_generic.go`
- `device/metal/*.metal`
- `device/metal/bridge_*_darwin.m`
- `device/metal/kernels.go`

Steps:
1. Implement Metal kernels for `Count8`, `Count16`, `Count32`, and `Count64`.
2. Dispatch those kernels from Darwin backend methods.
3. Keep Go scalar pospop functions only as scalar references for tests.
4. Replace GPU-circular Hawkes gold helpers with CPU scalar expected-value helpers.
5. Replace GPU-circular SwiGLU expected helpers with CPU scalar expected-value helpers.
6. Replace normalization GPU-helper expected paths with independent CPU scalar expected-value helpers.

Acceptance gates:
- `go test ./device/metal -run 'Count|Hawkes|SwiGLU|Normalization'`
- Tests must fail if the Metal kernel is replaced by the host helper.
- Expected values in tests must not call another Metal kernel.

### Packet 8: Complete Metal Dtype Coverage

Modify:
- `device/metal/backend.go`
- `device/metal/kernels.go`
- `device/metal/elementwise_float64.metal`
- `device/metal/elementwise_float32.metal`
- `device/metal/elementwise_bfloat16.metal`
- `device/metal/elementwise_float16.metal`
- Operation-specific Metal files and registries.

Steps:
1. Keep `SupportedDTypes()` as the target dtype list for Metal and implement native Metal tensor math for every listed dtype.
2. Add Float64 kernels and registrations for every elementwise binary and unary operation that exists for Float32.
3. Add Float8E4M3 and Float8E5M2 kernels for every operation currently registered for Float16 and BFloat16.
4. Add Int64, Int32, Int16, Int8, Uint64, Uint32, Uint16, Uint8, Bool, Complex64, and Complex128 kernels for every operation currently registered for Float32.
5. Add dtype-specific Metal tests and benchmarks for every new kernel.
6. Keep dtype conversion outside `Upload`; update `backend.go` comments to state that `Upload` copies bytes exactly as provided.

Acceptance gates:
- `go test ./device/metal -run 'DType|Float64|Float8|Int|Uint|Bool|Complex'`
- `go test ./device/metal -bench 'Float64|Float8|Int|Uint|Bool|Complex' -run '^$'`
- `SupportedDTypes()` must not list a dtype that lacks operation coverage.

### Packet 9: Fix Metal Scalar Parameters And Scalar Outputs

Modify:
- `device/metal/device_dispatch_darwin.go`
- `device/metal/bridge_*_darwin.m`
- Metal kernels that currently receive one-element scalar tensors.

Steps:
1. Pass scalar parameters of 16 bytes or less through `setBytes:length:atIndex:` in the command encoder.
2. Pass scalar parameter blocks larger than 16 bytes through one persistent typed Metal constant buffer allocated during backend initialization.
3. Remove per-call `uploadFloat32Scalar` and `uploadInt32Scalar` use from tensor kernels.
4. Keep scalar-output downloads only for APIs whose return type is scalar.
5. For tensor-output APIs, keep outputs Metal-resident and avoid host synchronization.
6. Add tests that confirm tensor-output APIs do not call `Download` internally.

Acceptance gates:
- `go test ./device/metal -run 'Scalar|Reduction|Loss|Dot|Physics|Hawkes|Causal'`
- Benchmarks for scalar-parameter kernels must show no one-element upload allocation in the measured path.

### Packet 10: Coalesce Metal Command Submission

Modify:
- `device/metal/bridge_darwin.m`
- `device/metal/bridge_darwin_private.h`
- `device/metal/backend.go`
- Bridge dispatch files that currently open one command buffer per operation.

Steps:
1. Make command-buffer batching internal to the backend for dependent operation chains.
2. Keep explicit `BeginBatch` and `EndBatch`, but do not require callers to use them for ordinary fused chains.
3. Fuse `LogSoftmax` into one Metal kernel.
4. Fuse packed GLU split plus GLU math into one Metal kernel per GLU variant and dtype.
5. Fuse optimizer update chains that read and write the same tensor set.
6. Add benchmarks that record command-buffer counts before and after fusion using the same tensor sizes.

Acceptance gates:
- `go test ./device/metal -run 'LogSoftmax|GLU|Optimizer'`
- `go test ./device/metal -bench 'LogSoftmax|GLU|Optimizer' -run '^$'`
- `LogSoftmax`, every packed GLU variant, and each fused optimizer update must enqueue one command buffer and one compute pass.

### Packet 11: Parallelize Metal Reductions And Scalar Kernels

Modify:
- `device/metal/reduction.metal`
- `device/metal/active.metal`
- `device/metal/loss.metal`
- `device/metal/causal.metal`
- `device/metal/hawkes*.metal`
- `device/metal/sampling*.metal`
- Matching bridge files.

Steps:
1. Replace single-threadgroup reductions with multi-threadgroup partial reductions.
2. Add a second-stage finalize kernel where the reduction output does not fit in one threadgroup.
3. Use threadgroup memory for per-group partials.
4. Bound atomics to final aggregation only when the operation cannot be expressed as a deterministic tree.
5. Add benchmarks at sizes `1, 7, 64, 1024, 8192, 65536, 1048576`.

Acceptance gates:
- `go test ./device/metal -run 'Reduction|Active|Loss|Causal|Hawkes|Sampling'`
- `go test ./device/metal -bench 'Reduction|Active|Loss|Causal|Hawkes|Sampling' -run '^$'`
- Large-size benchmarks must show scaling beyond one threadgroup.

### Packet 12: Replace Metal Packed GLU Temporary Tensors

Modify:
- `device/metal/device_dispatch_darwin.go`
- `device/metal/activation_*glu*.metal`
- `device/metal/bridge_*glu*_darwin.m`
- GLU tests and benchmarks.

Steps:
1. Delete the split-then-GLU execution path from `gluPackedInvoke`.
2. Implement packed kernels that read gate and up halves directly from the packed buffer.
3. Implement separate packed kernels for GLU, GeGLU, GeGLUTanh, SwiGLU, ReGLU, SiGLU, LinGLU, and SeGLU.
4. Implement separate packed kernels for Float32, Float16, and BFloat16.
5. Add parity tests for batch counts `1, 2, 7` and half counts `1, 7, 64, 1024, 8192`.
6. Add benchmarks proving the packed path performs one dispatch and allocates no temporary tensors.

Acceptance gates:
- `go test ./device/metal -run 'Packed.*GLU|GLU.*Packed'`
- `go test ./device/metal -bench 'Packed.*GLU|GLU.*Packed' -run '^$'`
- `gluPackedInvoke` must not allocate gate or up tensors.

### Packet 13: Complete Parity And Benchmark Coverage

Modify every package under:
- `device/cpu`
- `device/metal`

Steps:
1. For each code file that implements a kernel, create or update a mirrored `_test.go` file.
2. For CPU, add `*_avx512_parity_test.go`, `*_avx2_sse2_parity_test.go`, and `*_neon_parity_test.go` for every operation/dtype family.
3. For Metal, add parity tests for every registered kernel and every dtype.
4. Add benchmarks for scalar Go, AVX-512, AVX2, SSE2, NEON, and Metal for every operation/dtype family.
5. Replace absolute epsilon tests with ULP tests or bitwise equality.
6. Delete GPU-vs-GPU expected-value tests after replacing them with scalar-reference tests.
7. Add missing tests and benchmarks for Metal `page_write`, `page_gather`, Cholesky, and Float64 operations.

Acceptance gates:
- `go test ./device/cpu/...`
- `go test ./device/metal`
- `go test ./device/cpu/... -bench 'Benchmark' -run '^$'`
- `go test ./device/metal -bench 'Benchmark' -run '^$'`
- No kernel package may have implementation files without mirrored parity tests and benchmarks.

### Packet 14: Remove Performance-Losing Scalar Tails Where Vector Masks Exist

Modify:
- CPU matmul tail code.
- CPU embedding gather/add code.
- CPU pooling window code.
- CPU layernorm RMSNorm allocation path.
- CPU dropout NEON epilogue.
- Metal elementwise tail handling.

Steps:
1. Replace scalar column tails in matmul with masked vector stores for AVX-512 and NEON.
2. Replace scalar column tails in matmul with width-specific remainder kernels for AVX2 and SSE2.
3. Keep scalar index loads in embedding, but replace every hidden-dimension copy/add loop with vectorized row gather/copy kernels.
4. Replace pooling scalar window kernels with native kernels for fixed and dynamic windows.
5. Replace RMSNorm per-row allocation with an in-place fused kernel that multiplies normalized rows by scale without allocating `combined`.
6. Replace NEON dropout scalar epilogues with lane-mask vector epilogues.
7. Replace Metal scalar tail loops with one guarded vector tail block per threadgroup; do not use per-element tail loops in Metal kernels.

Acceptance gates:
- Existing parity tests must pass at `N = {1, 7, 64, 1024, 8192}`.
- Benchmarks must include sizes that exercise each remainder path.
- No hot kernel may allocate per row or per operation in the measured loop.

### Packet 15: Final Audit And Proof

Steps:
1. Run `go test ./device/cpu/...`.
2. Run `go test ./device/metal`.
3. Run `go test ./device/cpu/... -bench 'Benchmark' -run '^$'`.
4. Run `go test ./device/metal -bench 'Benchmark' -run '^$'`.
5. Run the operation-level dispatch audit from Packet 1.
6. Search for forbidden patterns in kernel paths:
   - Assembly entry points that jump to another operation.
   - Host-only Metal tensor computation.
   - Generic-only candidates for required native backend entries.
   - Absolute epsilon tolerance in kernel parity tests.
   - Code generator scripts used to create kernels.
7. Paste the full test and benchmark output into the completion report.

Acceptance gates:
- All commands pass.
- The dispatch audit reports complete operation/dtype/backend coverage.
- Every remaining scalar Go implementation is the scalar reference, not the implementation used by a native backend entry.

