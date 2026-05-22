# CUDA Implementation Contract

This document is an execution contract for completing `device/cuda`. The CUDA backend must reach full parity with the CPU scalar reference and the Metal backend. Do not mark CUDA complete until every operation exposed by `device.Backend` has a real CUDA implementation, dtype coverage matches the backend capability surface, and the hardware verification packet passes on amd64 NVIDIA hardware.

Development on machines without NVIDIA hardware must still produce complete CUDA source, Go bindings, registrations, parity tests, and benchmarks. Hardware execution is deferred until the full implementation exists. Do not use the development machine as a reason to omit kernels, tests, benchmarks, dtype paths, registrations, or bridge code.

## Non-Negotiable Rules

1. Do not replace a CUDA kernel with Go, C, C++, CPU BLAS, host-side loops, a Metal call, an XLA call, a generated table, a Python script, a shell script, or a fake device wrapper.
2. Do not leave C functions that return `NULL`, `0`, `ErrNeedsPlatformSetup`, or success without calling CUDA runtime or driver APIs in any `//go:build cuda` file.
3. Do not use generator scripts to write kernels. Write CUDA source directly.
4. Use shared CUDA byte-copy kernels only for pure data movement operations with byte-identical dtype storage. Every value-interpreting operation must have its own operation-specific CUDA kernel.
5. Do not share bf16, fp16, fp8, and float32 math paths. Each dtype that interprets values must have a dtype-specific CUDA kernel or a dtype-specific CUDA template instantiation.
6. Do not compare CUDA output against CUDA output. Every parity test must compare CUDA output against the Go scalar reference for the same operation and dtype.
7. Do not widen tolerances to pass CUDA tests. Use bitwise equality or a named tight ULP bound justified by the scalar reference operation order.
8. Do not claim hardware verification before tests run on amd64 NVIDIA hardware. Writing the full implementation without local hardware execution is allowed; completion is not allowed until the hardware packet passes.
9. Do not delete or narrow public backend capability claims to avoid writing kernels.
10. Do not leave `UploadSparse` unsupported. CUDA must implement dense and sparse layouts with real device storage and execution.

## Current State

- `device/cuda/backend.go` exposes only tensor backend lifecycle, capability, upload, download, and close methods.
- `device/cuda/bridge_stub.go` correctly returns `tensor.ErrNeedsPlatformSetup` when built without the `cuda` tag.
- `device/cuda/bridge_real.go` is not a real CUDA bridge. Its C functions return `NULL`, `0`, or success without calling CUDA.
- `device/cuda/backend_test.go` tests only the non-CUDA stub surface and `Location`.
- There are no CUDA kernels, no CUDA operation registries, no resident CUDA tensor type, no CUDA stream/event completion system, no CUDA parity suites, and no CUDA benchmarks.

## Target Surface

CUDA must implement every interface in `device/interface.go`:

1. `PosPop`
2. `Activation`
3. `Elementwise`
4. `Reduction`
5. `Dot`
6. `Matmul`
7. `Pool`
8. `Convolution`
9. `Dropout`
10. `Losses`
11. `Sampling`
12. `Embedding`
13. `Normalization`
14. `LayerNorm`
15. `RoPE`
16. `Hawkes`
17. `Masking`
18. `Attention`
19. `Optimizer`
20. `Checkpoint`
21. `Quant`
22. `Dequant`
23. `VSA`
24. `ActiveInference`
25. `PredictiveCoding`
26. `Causal`
27. `Physics`
28. `Interpretability`
29. `ModelEditing`
30. `Shape`
31. `Tokenizer`

Every operation/dtype pair implemented for CPU and Metal must have a CUDA equivalent.

## Packet 1: Real CUDA Bridge And Tensor Residency

Modify:
- `device/cuda/bridge_real.go`
- `device/cuda/backend.go`
- Create `device/cuda/tensor.go`
- Create `device/cuda/resident.go`
- Create `device/cuda/status.go`
- Create `device/cuda/stream.go`
- Create `device/cuda/memory.go`
- Create `device/cuda/bridge_cuda.cu`
- Create `device/cuda/bridge_cuda.h`

Steps:
1. Replace every static fake C function in `bridge_real.go` with declarations implemented in `bridge_cuda.cu`.
2. Implement device open with `cudaGetDevice`, `cudaGetDeviceProperties`, and capability detection.
3. Implement stream creation, stream destruction, event creation, event recording, event synchronization, and event destruction.
4. Implement device allocation with `cudaMalloc`.
5. Implement device release with `cudaFree`.
6. Implement pinned host staging allocation with `cudaMallocHost`.
7. Implement pinned host staging release with `cudaFreeHost`.
8. Implement upload with `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, stream)` followed by event synchronization for synchronous `Upload`.
9. Implement `UploadAsync` with event-backed pending tensor state.
10. Implement download with `cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, stream)` followed by synchronization before returning bytes.
11. Implement `cudaTensor` with shape, dtype, device pointer, byte count, stream/event state, close state, and resident pointer token.
12. Implement resident pointer lookup for unsafe-pointer backend methods. The pointer token must identify a CUDA tensor; it must not be a host pointer to tensor contents.

Acceptance gates:
- `go test ./device/cuda` without `-tags cuda` must keep passing the stub tests.
- `go test -tags cuda ./device/cuda -run 'Backend|Upload|Download|Resident|Stream|Memory'` must pass on CUDA hardware.
- `bridge_real.go` must contain no fake C function bodies.
- Every `//go:build cuda` bridge path must call real CUDA runtime or driver APIs.

## Packet 2: CUDA Backend Method Surface

Modify:
- `device/cuda/backend.go`
- Create operation files mirroring Metal package organization:
  - `device/cuda/device_cuda.go`
  - `device/cuda/device_remaining_cuda.go`
  - `device/cuda/device_missing_cuda.go`
  - `device/cuda/kernels.go`

Steps:
1. Implement every method required by `device.Backend`, not only `tensor.Backend`.
2. Each unsafe-pointer method must resolve resident CUDA tensors and panic on invalid resident pointers.
3. Each method must validate `format dtype.DType` against the resident tensor dtype before dispatch.
4. Each method must enqueue a CUDA kernel or CUDA library call on a backend stream.
5. Each method must record completion on output tensors.
6. Scalar-returning methods must synchronize only the scalar output buffer.
7. Tensor-output methods must not download tensor contents to host.

Acceptance gates:
- Add `var _ device.Backend = (*Backend)(nil)` in a CUDA build-tagged file.
- `go test -tags cuda ./device/cuda -run 'BackendSurface|DTypeMismatch|Resident'` must pass on CUDA hardware.
- CUDA backend methods must not silently return after validation or dispatch failure.

## Packet 3: Operation Registry Parity

Modify:
- `device/cuda/kernels.go`
- `kernels` package registrations that include CUDA.
- `fusion/catalog.go`

Steps:
1. Register CUDA kernels for every kernel name and signature registered by Metal.
2. Register CUDA kernels for every CPU-only kernel that has no Metal registration.
3. Use `tensor.CUDA` in every CUDA location list.
4. Add operation inventory tests that compare CPU, Metal, and CUDA registry coverage.
5. Fail registry tests when CUDA lacks any operation/dtype signature present on CPU or Metal.

Acceptance gates:
- `go test -tags cuda ./device/cuda ./kernels ./fusion -run 'Registry|Catalog|Coverage'` must pass on CUDA hardware.
- Registry coverage must report zero missing CUDA signatures.

## Packet 4: Elementwise, Activation, And GLU Kernels

Create:
- `device/cuda/elementwise.cu`
- `device/cuda/activation.cu`
- `device/cuda/activation_glu.cu`
- `device/cuda/bridge_elementwise_cuda.cu`
- `device/cuda/bridge_activation_cuda.cu`
- `device/cuda/elementwise_test.go`
- `device/cuda/activation_test.go`
- `device/cuda/activation_glu_test.go`
- Matching benchmark files.

Steps:
1. Implement binary elementwise: add, sub, mul, div, max, min, eq, ne, lt, le, gt, ge, pow, atan2, mod.
2. Implement unary elementwise: abs, neg, sqrt, rsqrt, exp, log, sin, cos, tanh, gelu, sigmoid, silu, swish, softsign, elu, selu, leaky_relu, hard_sigmoid, hard_swish, log1p, expm1, celu, softplus, mish, log_sigmoid, gelu_tanh, hard_tanh, hard_gelu, quick_gelu, tanh_shrink.
3. Implement parametric activations: PReLU, PReLUV, LeakyReLUSlope, ELUAlpha, CELUAlpha, Threshold, HardTanhRange, Snake, SnakeParametric, HardShrink, SoftShrink, RReLU.
4. Implement tensor GLU and packed GLU: GLU, GeGLU, GeGLUTanh, SwiGLU, ReGLU, SiGLU, LinGLU, SeGLU.
5. Implement dtype-specific kernels for Float64, Float32, Float16, BFloat16, Float8E4M3, Float8E5M2, Int64, Int32, Int16, Int8, Uint64, Uint32, Uint16, Uint8, Bool, Complex64, and Complex128 wherever the operation is exposed.
6. Use vectorized CUDA loads and stores where alignment permits. Use one guarded tail block inside the CUDA kernel; do not run a host tail loop.
7. Do not use cuBLAS or cuDNN for elementwise and activation kernels.

Acceptance gates:
- `go test -tags cuda ./device/cuda -run 'Elementwise|Activation|GLU'`
- `go test -tags cuda ./device/cuda -bench 'Elementwise|Activation|GLU' -run '^$'`
- Tests must cover `N = {1, 7, 64, 1024, 8192}` for every operation/dtype pair.

## Packet 5: Reductions, Dot, Matmul, Attention

Create:
- `device/cuda/reduction.cu`
- `device/cuda/dot.cu`
- `device/cuda/matmul.cu`
- `device/cuda/attention.cu`
- Matching bridge, test, and benchmark files.

Steps:
1. Implement Sum, Prod, ReduceMin, ReduceMax, and L1Norm with multi-block reductions.
2. Implement Dot with multi-block partials and deterministic finalization.
3. Implement Matmul for dense row-major tensors with tiled shared-memory kernels.
4. Implement reduced-precision matmul for Float16, BFloat16, Float8E4M3, and Float8E5M2 with dtype-correct accumulation.
5. Implement sparse CSR matmul with real CUDA kernels or cuSPARSE calls.
6. Implement attention score, weighted value, flash attention, and multi-head attention paths exposed by CPU and Metal.
7. Use cuBLAS only for dense matmul paths whose semantics exactly match the scalar reference. Custom kernels are required for custom accumulation, masking, packed layout, and dtype behavior.
8. Do not use host-side loops for reduction finalization.

Acceptance gates:
- `go test -tags cuda ./device/cuda -run 'Reduction|Dot|Matmul|Attention'`
- `go test -tags cuda ./device/cuda -bench 'Reduction|Dot|Matmul|Attention' -run '^$'`
- Large-size benchmarks must include `N = 65536` and `N = 1048576` for reductions and dot.

## Packet 6: Vision, Pooling, Convolution, Shape, Tokenizer

Create:
- `device/cuda/pool.cu`
- `device/cuda/convolution.cu`
- `device/cuda/shape.cu`
- `device/cuda/tokenizer.cu`
- Matching bridge, test, and benchmark files.

Steps:
1. Implement MaxPool2D, AvgPool2D, AdaptiveMaxPool2D, and AdaptiveAvgPool2D.
2. Implement Conv1D, Conv2D, Conv3D, and ConvTranspose2D.
3. Implement shape kernels: copy, slice, reshape-compatible copy, concat, split, gather, scatter, where, masked fill, transpose, page_write, and page_gather.
4. Implement tokenizer pack/unpack/count kernels that match CPU semantics.
5. Implement every dtype supported by the corresponding CPU and Metal operation.
6. Use shared memory tiling for Conv2D, Conv3D, ConvTranspose2D, and transpose kernels.
7. Do not call CPU pooling, CPU convolution, or CPU shape helpers from CUDA backend methods.

Acceptance gates:
- `go test -tags cuda ./device/cuda -run 'Pool|Conv|Shape|Tokenizer|Page'`
- `go test -tags cuda ./device/cuda -bench 'Pool|Conv|Shape|Tokenizer|Page' -run '^$'`

## Packet 7: Model Operations

Create:
- `device/cuda/normalization.cu`
- `device/cuda/layernorm.cu`
- `device/cuda/rope.cu`
- `device/cuda/embedding.cu`
- `device/cuda/dropout.cu`
- `device/cuda/loss.cu`
- `device/cuda/sampling.cu`
- `device/cuda/optimizer.cu`
- Matching bridge, test, and benchmark files.

Steps:
1. Implement GroupNorm, InstanceNorm, BatchNormEval, LayerNorm, and RMSNorm.
2. Implement RoPE and RoPEPairs.
3. Implement embedding lookup and embedding bag.
4. Implement dropout with deterministic CUDA RNG matching the scalar reference seed behavior.
5. Implement MSE, MAE, Huber, BinaryCrossEntropy, KLDivergence, and CrossEntropy.
6. Implement GreedySample, TopKSample, and TopPSample.
7. Implement every optimizer operation exposed by CPU and Metal.
8. Keep all intermediate tensors device-resident.
9. Do not generate random masks or sampling decisions on the host.

Acceptance gates:
- `go test -tags cuda ./device/cuda -run 'Norm|RoPE|Embedding|Dropout|Loss|Sample|Optimizer'`
- `go test -tags cuda ./device/cuda -bench 'Norm|RoPE|Embedding|Dropout|Loss|Sample|Optimizer' -run '^$'`

## Packet 8: Research And Physics Operations

Create:
- `device/cuda/hawkes.cu`
- `device/cuda/masking.cu`
- `device/cuda/vsa.cu`
- `device/cuda/active_inference.cu`
- `device/cuda/predictive_coding.cu`
- `device/cuda/causal.cu`
- `device/cuda/physics.cu`
- `device/cuda/interpretability.cu`
- `device/cuda/model_editing.cu`
- Matching bridge, test, and benchmark files.

Steps:
1. Implement Hawkes intensity, kernel matrix, and log likelihood.
2. Implement masking, causal mask, and ALiBi bias.
3. Implement VSA bind, bundle, permute, similarity, and cleanup.
4. Implement active-inference free energy, expected free energy, belief update, and precision weighting.
5. Implement predictive-coding kernels.
6. Implement causal kernels: backdoor, frontdoor, intervention, CATE, counterfactual, IV estimate, DAG Markov factorization, and Cholesky.
7. Implement physics kernels: Laplacian, gradient, divergence, quantum potential, Bohmian velocity, Madelung continuity, and FFT.
8. Implement interpretability and model-editing kernels.
9. Use cuFFT only after a parity test proves result ordering and numeric semantics match the scalar reference. Implement custom kernels for every FFT semantic not covered by cuFFT.

Acceptance gates:
- `go test -tags cuda ./device/cuda -run 'Hawkes|Mask|VSA|Active|Predictive|Causal|Physics|Interpretability|ModelEditing'`
- `go test -tags cuda ./device/cuda -bench 'Hawkes|Mask|VSA|Active|Predictive|Causal|Physics|Interpretability|ModelEditing' -run '^$'`

## Packet 9: Sparse Layouts

Modify:
- `device/cuda/backend.go`
- `device/cuda/memory.go`
- `device/cuda/kernels.go`
- Sparse operation files.

Steps:
1. Implement `UploadSparse` for CSR and every sparse layout supported by `manifesto/tensor`.
2. Store sparse values and indices in CUDA device buffers.
3. Implement sparse matmul, sparse embedding, sparse reductions, and sparse shape operations for every sparse operation exposed by the backend.
4. Use cuSPARSE only after a parity test proves it exactly matches scalar reference semantics.
5. Add sparse parity tests against CPU scalar sparse references.

Acceptance gates:
- `go test -tags cuda ./device/cuda -run 'Sparse|CSR'`
- `go test -tags cuda ./device/cuda -bench 'Sparse|CSR' -run '^$'`
- `Capabilities().SupportsSparse` must be true after sparse operations pass.

## Packet 10: Local Non-Hardware Verification

Run on the development machine without NVIDIA hardware:

1. `go test ./device/cuda`
2. `go test ./...`
3. `go test ./device/cpu/...`
4. `go test ./device/metal`
5. Static search over `device/cuda` for forbidden fake implementation patterns:
   - `return NULL`
   - `return 0` in CUDA C bridge functions that are required to call CUDA APIs
   - `ErrNeedsPlatformSetup` in `//go:build cuda` operation paths
   - CPU package imports from CUDA operation files
   - Python or shell generator scripts under `device/cuda`

Acceptance gates:
- Non-CUDA stub tests must pass.
- Static search must find no fake CUDA implementation in `//go:build cuda` files.
- This packet does not certify runtime correctness.

## Packet 11: Hardware Verification On amd64 NVIDIA

Run only after Packets 1 through 10 are complete.

Hardware:
- amd64 host.
- NVIDIA GPU with CUDA runtime and driver installed.
- Hopper or Blackwell GPU for FP8 verification.
- Any sm_70+ GPU for base Float64, Float32, Float16, BFloat16, integer, bool, and complex verification.

Commands:
1. `go test -tags cuda ./device/cuda`
2. `go test -tags cuda ./device/cuda -run 'Parity|DType|Native|Sparse|Registry'`
3. `go test -tags cuda ./device/cuda -bench 'Benchmark' -run '^$'`
4. `go test -tags cuda ./...`
5. On a runner that has both amd64 NVIDIA CUDA and Metal available, run `go test -tags 'cuda metal' ./...`.

Acceptance gates:
- Every command available on the hardware runner must pass.
- Failures discovered on hardware must be fixed in CUDA source, bridge code, tests, or scalar references.
- Do not weaken tests after hardware failures.
- Do not replace failing kernels with host computation after hardware failures.
- Paste full command output into the completion report.

## Final Completion Standard

CUDA completion requires all of these facts:

1. `device/cuda` implements `device.Backend`.
2. Every operation/dtype pair present on CPU and Metal has a CUDA registration.
3. Every CUDA registration dispatches a real CUDA kernel or exact-semantics CUDA library call.
4. Every CUDA tensor result stays on device until the caller downloads it.
5. Every parity test compares CUDA against the Go scalar reference.
6. Every benchmark includes scalar and CUDA measurements.
7. Non-hardware verification passes on the development machine.
8. Hardware verification passes on amd64 NVIDIA hardware.
