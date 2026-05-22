# XLA Implementation Contract

This document is an execution contract for completing `device/xla`. The XLA backend must reach full parity with the CPU scalar reference, the CPU native kernels, the Metal backend, and the CUDA backend when CUDA is completed. XLA must execute real XLA programs through the XLA runtime. Host-side tensor computation wrapped in an XLA backend is not an implementation.

Development on machines without XLA runtime support or NVIDIA hardware must still produce complete XLA source, Go bindings, lowering code, registrations, parity tests, and benchmarks. Runtime verification is deferred until the full implementation exists and runs on a real amd64 machine with NVIDIA hardware and XLA-GPU support. Bugs found during that verification must be fixed after the hardware run.

## Non-Negotiable Rules

1. Do not replace an XLA operation with Go, C, C++, CPU loops, CUDA calls outside XLA, Metal calls, generated constants, Python scripts, shell scripts, or fake buffers.
2. Do not leave `//go:build xla` functions that return `ErrNeedsPlatformSetup`, success without XLA runtime work, or placeholder buffers.
3. Claim an operation is implemented only after it builds an XLA computation, compiles it, executes it through XLA/PJRT, and returns an XLA-resident tensor.
4. Do not compare XLA output against XLA output. Every parity test must compare XLA output against the Go scalar reference for the same operation and dtype.
5. Do not widen tolerances to pass XLA tests. Use bitwise equality or a named tight ULP bound justified by scalar reference operation order.
6. Do not use host callbacks for tensor math.
7. Do not transfer tensors to host between fused XLA operations.
8. Do not shrink `SupportedDTypes()` to avoid implementing dtype paths.
9. Do not mark hardware verification complete on the development machine.
10. Do not depend on CUDA custom kernels for XLA operation bodies. XLA-GPU is an allowed execution target for compiled XLA programs; operation definitions must be expressed in XLA lowering.

## Current State

- `device/xla/backend.go` exposes tensor backend lifecycle, dtype list, dense layout support, upload, download, and close.
- `device/xla/bridge_stub.go` returns `tensor.ErrNeedsPlatformSetup` without the `xla` build tag.
- There is no `//go:build xla` bridge file.
- There is no XLA tensor residency type.
- There is no XLA program builder, lowering layer, compile cache, executable cache, operation registry, parity suite, or benchmark suite.
- `SupportedDTypes()` claims Float64, Float32, Float16, BFloat16, Float8E4M3, Float8E5M2, Int64, Int32, Int16, Int8, Uint64, Uint32, Uint16, Uint8, and Bool. Every listed dtype must be implemented.

## Target Surface

XLA must implement every interface in `device/interface.go`:

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

Every operation/dtype pair implemented for CPU, Metal, and CUDA must have an XLA equivalent.

## Packet 1: Real XLA Runtime Bridge

Modify:
- `device/xla/backend.go`
- `device/xla/bridge_stub.go`
- Create `device/xla/bridge_xla.go`
- Create `device/xla/bridge_xla.cc`
- Create `device/xla/bridge_xla.h`
- Create `device/xla/status.go`
- Create `device/xla/client.go`
- Create `device/xla/tensor.go`
- Create `device/xla/resident.go`
- Create `device/xla/executable.go`

Steps:
1. Implement a `//go:build xla` bridge that opens a real XLA/PJRT client.
2. Query device memory and expose it through `Capabilities().MaxBytes`.
3. Implement host-to-device transfer through the XLA transfer manager.
4. Implement async upload with event-backed pending tensor state.
5. Implement device-to-host transfer for `Download`.
6. Implement `xlaTensor` with shape, dtype, XLA buffer handle, byte count, ready state, close state, and resident token.
7. Implement resident token lookup for unsafe-pointer backend methods. The token must identify an XLA buffer; it must not be a host pointer to tensor data.
8. Implement close semantics that release XLA buffers, compiled executables, and client resources.
9. Preserve `bridge_stub.go` behavior for builds without the `xla` tag.

Acceptance gates:
- `go test ./device/xla` without `-tags xla` must keep passing stub tests.
- `go test -tags xla ./device/xla -run 'Backend|Upload|Download|Resident|Client|Executable'` must pass on the XLA hardware runner.
- `//go:build xla` bridge functions must not return fake success without calling XLA/PJRT APIs.

## Packet 2: XLA Backend Method Surface

Modify:
- `device/xla/backend.go`
- Create `device/xla/device_xla.go`
- Create `device/xla/device_remaining_xla.go`
- Create `device/xla/device_missing_xla.go`
- Create `device/xla/kernels.go`

Steps:
1. Implement every method required by `device.Backend`.
2. Each unsafe-pointer method must resolve resident XLA tensors and panic on invalid resident pointers.
3. Each method must validate `format dtype.DType` against resident tensor dtype before lowering.
4. Each tensor operation must lower to an XLA computation.
5. Each tensor operation must return or write an XLA-resident buffer.
6. Scalar-returning methods must transfer only the scalar output buffer to host.
7. Tensor-output methods must not download tensor contents to host.

Acceptance gates:
- Add `var _ device.Backend = (*Backend)(nil)` in an XLA build-tagged file.
- `go test -tags xla ./device/xla -run 'BackendSurface|DTypeMismatch|Resident'` must pass on the XLA hardware runner.
- XLA backend methods must not silently return after validation, lowering, compile, or execute failure.

## Packet 3: XLA Lowering Framework

Create:
- `device/xla/lowering.go`
- `device/xla/builder.go`
- `device/xla/dtype.go`
- `device/xla/shape.go`
- `device/xla/cache.go`
- `device/xla/program_key.go`

Steps:
1. Implement dtype mapping between `dtype.DType` and XLA element types.
2. Implement tensor shape mapping between `tensor.Shape` and XLA shapes.
3. Implement an operation lowering interface that takes resident buffers and returns an XLA computation.
4. Implement compile cache keys using operation name, dtype list, shape list, scalar parameters, and backend target.
5. Compile each unique computation once per key.
6. Execute cached executables with XLA buffers as arguments.
7. Keep output allocation device-resident.
8. Include scalar parameters in the compiled computation when they affect code shape; pass runtime scalar parameters as XLA scalar arguments when they do not.

Acceptance gates:
- `go test -tags xla ./device/xla -run 'Lowering|Builder|DType|Shape|Cache'`
- Tests must prove that repeated executions reuse the compiled executable for the same key.
- Tests must prove that differing dtype, shape, or operation parameters produce different cache keys.

## Packet 4: Operation Registry Parity

Modify:
- `device/xla/kernels.go`
- `kernels` package registrations that include XLA.
- `fusion/catalog.go`

Steps:
1. Register XLA kernels for every kernel name and signature registered by Metal.
2. Register XLA kernels for every CPU-only kernel that has no Metal registration.
3. Register XLA kernels for every CUDA kernel once CUDA packet completion adds signatures.
4. Use `tensor.XLA` in every XLA location list.
5. Add operation inventory tests that compare CPU, Metal, CUDA, and XLA registry coverage.
6. Fail registry tests when XLA lacks any operation/dtype signature present on CPU, Metal, or CUDA.

Acceptance gates:
- `go test -tags xla ./device/xla ./kernels ./fusion -run 'Registry|Catalog|Coverage'`
- Registry coverage must report zero missing XLA signatures.

## Packet 5: Elementwise, Activation, And GLU Lowerings

Create:
- `device/xla/elementwise.go`
- `device/xla/activation.go`
- `device/xla/activation_glu.go`
- `device/xla/elementwise_test.go`
- `device/xla/activation_test.go`
- `device/xla/activation_glu_test.go`
- Matching benchmark files.

Steps:
1. Lower binary elementwise: add, sub, mul, div, max, min, eq, ne, lt, le, gt, ge, pow, atan2, mod.
2. Lower unary elementwise: abs, neg, sqrt, rsqrt, exp, log, sin, cos, tanh, gelu, sigmoid, silu, swish, softsign, elu, selu, leaky_relu, hard_sigmoid, hard_swish, log1p, expm1, celu, softplus, mish, log_sigmoid, gelu_tanh, hard_tanh, hard_gelu, quick_gelu, tanh_shrink.
3. Lower parametric activations: PReLU, PReLUV, LeakyReLUSlope, ELUAlpha, CELUAlpha, Threshold, HardTanhRange, Snake, SnakeParametric, HardShrink, SoftShrink, RReLU.
4. Lower tensor GLU and packed GLU: GLU, GeGLU, GeGLUTanh, SwiGLU, ReGLU, SiGLU, LinGLU, SeGLU.
5. Implement every dtype in `SupportedDTypes()`.
6. Use XLA operations for tail handling through shape-correct slicing, concatenation, masking, or broadcasting.
7. Do not transfer packed GLU halves to host. Use XLA slice operations and fused math.

Acceptance gates:
- `go test -tags xla ./device/xla -run 'Elementwise|Activation|GLU'`
- `go test -tags xla ./device/xla -bench 'Elementwise|Activation|GLU' -run '^$'`
- Tests must cover `N = {1, 7, 64, 1024, 8192}` for every operation/dtype pair.

## Packet 6: Reductions, Dot, Matmul, Attention

Create:
- `device/xla/reduction.go`
- `device/xla/dot.go`
- `device/xla/matmul.go`
- `device/xla/attention.go`
- Matching test and benchmark files.

Steps:
1. Lower Sum, Prod, ReduceMin, ReduceMax, and L1Norm through XLA reduction operations.
2. Lower Dot through XLA dot or reduction operations with dtype-correct accumulation.
3. Lower dense Matmul through XLA dot_general.
4. Lower reduced-precision matmul for Float16, BFloat16, Float8E4M3, and Float8E5M2 with scalar-reference accumulation semantics.
5. Lower sparse CSR matmul with XLA sparse operations after a parity test proves those primitives match the scalar reference. Implement explicit XLA gather/scatter/reduce lowerings for every sparse semantic not covered by XLA sparse primitives.
6. Lower attention score, weighted value, flash attention, and multi-head attention.
7. Keep attention intermediates in XLA buffers.
8. Do not call CPU matmul, CUDA matmul, Metal matmul, or host BLAS.

Acceptance gates:
- `go test -tags xla ./device/xla -run 'Reduction|Dot|Matmul|Attention'`
- `go test -tags xla ./device/xla -bench 'Reduction|Dot|Matmul|Attention' -run '^$'`
- Large-size benchmarks must include `N = 65536` and `N = 1048576` for reductions and dot.

## Packet 7: Vision, Pooling, Convolution, Shape, Tokenizer

Create:
- `device/xla/pool.go`
- `device/xla/convolution.go`
- `device/xla/shape_ops.go`
- `device/xla/tokenizer.go`
- Matching test and benchmark files.

Steps:
1. Lower MaxPool2D, AvgPool2D, AdaptiveMaxPool2D, and AdaptiveAvgPool2D.
2. Lower Conv1D, Conv2D, Conv3D, and ConvTranspose2D.
3. Lower shape kernels: copy, slice, reshape-compatible copy, concat, split, gather, scatter, where, masked fill, transpose, page_write, and page_gather.
4. Lower tokenizer pack, unpack, and count kernels.
5. Implement every dtype supported by the corresponding CPU, Metal, and CUDA operations.
6. Use XLA convolution and window reduction operations only after parity tests prove their padding, stride, dilation, and accumulation semantics match the scalar reference.
7. Use explicit XLA lowerings when a built-in XLA op does not match the scalar reference exactly.

Acceptance gates:
- `go test -tags xla ./device/xla -run 'Pool|Conv|Shape|Tokenizer|Page'`
- `go test -tags xla ./device/xla -bench 'Pool|Conv|Shape|Tokenizer|Page' -run '^$'`

## Packet 8: Model Operations

Create:
- `device/xla/normalization.go`
- `device/xla/layernorm.go`
- `device/xla/rope.go`
- `device/xla/embedding.go`
- `device/xla/dropout.go`
- `device/xla/loss.go`
- `device/xla/sampling.go`
- `device/xla/optimizer.go`
- Matching test and benchmark files.

Steps:
1. Lower GroupNorm, InstanceNorm, BatchNormEval, LayerNorm, and RMSNorm.
2. Lower RoPE and RoPEPairs.
3. Lower embedding lookup and embedding bag.
4. Lower dropout with deterministic RNG semantics matching the scalar reference seed behavior.
5. Lower MSE, MAE, Huber, BinaryCrossEntropy, KLDivergence, and CrossEntropy.
6. Lower GreedySample, TopKSample, and TopPSample.
7. Lower every optimizer operation exposed by CPU, Metal, and CUDA.
8. Keep all intermediate tensors XLA-resident.
9. Do not use host callbacks for random masks, sampling, reductions, or optimizer state updates.

Acceptance gates:
- `go test -tags xla ./device/xla -run 'Norm|RoPE|Embedding|Dropout|Loss|Sample|Optimizer'`
- `go test -tags xla ./device/xla -bench 'Norm|RoPE|Embedding|Dropout|Loss|Sample|Optimizer' -run '^$'`

## Packet 9: Research And Physics Operations

Create:
- `device/xla/hawkes.go`
- `device/xla/masking.go`
- `device/xla/vsa.go`
- `device/xla/active_inference.go`
- `device/xla/predictive_coding.go`
- `device/xla/causal.go`
- `device/xla/physics.go`
- `device/xla/interpretability.go`
- `device/xla/model_editing.go`
- Matching test and benchmark files.

Steps:
1. Lower Hawkes intensity, kernel matrix, and log likelihood.
2. Lower masking, causal mask, and ALiBi bias.
3. Lower VSA bind, bundle, permute, similarity, and cleanup.
4. Lower active-inference free energy, expected free energy, belief update, and precision weighting.
5. Lower predictive-coding operations.
6. Lower causal operations: backdoor, frontdoor, intervention, CATE, counterfactual, IV estimate, DAG Markov factorization, and Cholesky.
7. Lower physics operations: Laplacian, gradient, divergence, quantum potential, Bohmian velocity, Madelung continuity, and FFT.
8. Lower interpretability and model-editing operations.
9. Use XLA FFT only after a parity test proves result ordering and numeric semantics match the scalar reference. Implement explicit XLA lowerings for every unmatched FFT semantic.

Acceptance gates:
- `go test -tags xla ./device/xla -run 'Hawkes|Mask|VSA|Active|Predictive|Causal|Physics|Interpretability|ModelEditing'`
- `go test -tags xla ./device/xla -bench 'Hawkes|Mask|VSA|Active|Predictive|Causal|Physics|Interpretability|ModelEditing' -run '^$'`

## Packet 10: Sparse Layouts

Modify:
- `device/xla/backend.go`
- `device/xla/kernels.go`
- XLA sparse lowering files.

Steps:
1. Implement `UploadSparse` for CSR and every sparse layout supported by `manifesto/tensor`.
2. Store sparse values and indices as XLA buffers.
3. Lower sparse matmul, sparse embedding, sparse reductions, and sparse shape operations for every sparse operation exposed by the backend.
4. Use XLA sparse primitives only after parity tests prove they exactly match scalar reference semantics.
5. Implement explicit gather/scatter/reduce lowerings for sparse semantics not covered by XLA primitives.
6. Add sparse parity tests against CPU scalar sparse references.

Acceptance gates:
- `go test -tags xla ./device/xla -run 'Sparse|CSR'`
- `go test -tags xla ./device/xla -bench 'Sparse|CSR' -run '^$'`
- `Capabilities().SupportsSparse` must be true after sparse operations pass.

## Packet 11: Fusion And Compile Cache

Modify:
- `device/xla/cache.go`
- `device/xla/kernels.go`
- `fusion/catalog.go`
- Operation lowering files.

Steps:
1. Lower fused operations as one XLA computation when the fusion catalog marks them as fused.
2. Compile each fused computation once per key.
3. Keep intermediate values inside the XLA computation.
4. Add cache metrics for compile count, execute count, cache hit count, and cache miss count.
5. Add tests proving `LogSoftmax`, packed GLU variants, optimizer update chains, and attention subgraphs compile to single fused executables.

Acceptance gates:
- `go test -tags xla ./device/xla ./fusion -run 'Fusion|Cache|LogSoftmax|GLU|Optimizer|Attention'`
- `go test -tags xla ./device/xla -bench 'Fusion|Cache|LogSoftmax|GLU|Optimizer|Attention' -run '^$'`
- Fused XLA operations must not materialize intermediates as host buffers.

## Packet 12: Local Non-Hardware Verification

Run on the development machine:

1. `go test ./device/xla`
2. `go test ./...`
3. `go test ./device/cpu/...`
4. `go test ./device/metal`
5. Static search over `device/xla` for forbidden fake implementation patterns:
   - `ErrNeedsPlatformSetup` in `//go:build xla` operation paths
   - Host tensor math in XLA operation files
   - CPU package imports from XLA operation files, except in tests
   - CUDA or Metal package imports from XLA operation files
   - Python or shell generator scripts under `device/xla`
   - XLA tests comparing XLA output against XLA output

Acceptance gates:
- Non-XLA stub tests must pass.
- Static search must find no fake XLA implementation in `//go:build xla` files.
- This packet does not certify runtime correctness.

## Packet 13: Hardware Verification On amd64 NVIDIA With XLA-GPU

Run only after Packets 1 through 12 are complete.

Hardware:
- amd64 host.
- NVIDIA GPU with CUDA driver installed.
- XLA/PJRT runtime with XLA-GPU backend available.
- Hopper or Blackwell GPU for FP8 verification.

Commands:
1. `go test -tags xla ./device/xla`
2. `go test -tags xla ./device/xla -run 'Parity|DType|Native|Sparse|Registry|Fusion'`
3. `go test -tags xla ./device/xla -bench 'Benchmark' -run '^$'`
4. `go test -tags xla ./...`
5. `go test -tags 'cuda xla' ./...` after CUDA completion on the same hardware class.

Acceptance gates:
- Every command available on the hardware runner must pass.
- Failures discovered on hardware must be fixed in XLA bridge code, lowering code, tests, or scalar references.
- Do not weaken tests after hardware failures.
- Do not replace failing lowerings with host computation after hardware failures.
- Paste full command output into the completion report.

## Final Completion Standard

XLA completion requires all of these facts:

1. `device/xla` implements `device.Backend`.
2. Every operation/dtype pair present on CPU, Metal, and CUDA has an XLA registration.
3. Every XLA registration builds, compiles, and executes a real XLA computation.
4. Every XLA tensor result stays device-resident until the caller downloads it.
5. Every parity test compares XLA against the Go scalar reference.
6. Every benchmark includes scalar and XLA measurements.
7. Non-hardware verification passes on the development machine.
8. Hardware verification passes on amd64 NVIDIA hardware with XLA-GPU.
