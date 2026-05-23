This is the architecture and implementation plan for the execution stack of `puter`. It is designed as a fully committed, deterministic specification optimized for maximum correctness, execution speed, and hardware utilization. 

All runtime dynamic allocations, host-device synchronization bottlenecks, and execution options have been eliminated. This plan specifies an ahead-of-time (AOT) compiler with JIT-fused elementwise kernels, symbolic shape-offset resolution, cache-aware memory layout planning, and out-of-order asynchronous DAG execution.

---

# Architecture & Implementation Specification: `puter` Execution Stack

## 1. Execution Stack

```
manifest assets (YAML)
        ↓
manifesto/compiler          Parse, validate, expand block macros, unify PortTypes
        ↓
manifesto/optimizer         1. Algebraic rewrites (e.g., fold scale into Linear)
                            2. Group elementwise nodes into FusionASTs
                            3. Apply cache-tiling and layout transformations
        ↓
manifesto/codegen           1. JIT-compile FusionASTs into native assembly / GPU kernels
                            2. Generate symbolic stride & offset math formulas
        ↓
manifesto/scheduler         1. Liveness analysis over the DAG
                            2. Map ports to static offsets in a single global workspace
                            3. Resolve execution dependency DAG into concurrent stream paths
        ↓
manifesto/ir                Fully resolved execution instructions (Topology + Static Allocations)
        ↓
execution                   Zero-allocation topological walk -> Dispatch to device queues
        ↓
device.Backend              Direct execution of native static kernels and JIT-compiled binaries
        ↓
ISA targets                 AVX-512, AVX2, SSE2, NEON, Metal (MSL), CUDA (PTX), XLA (HLO → PJRT)
```

Checkpoint tokens (safetensors) attach weight metadata after the graph is compiled. They do not define operations or allocate dynamic memory at runtime.

The manifesto pipeline is target-agnostic. The active `device.Backend` implementation selects how each operation is executed on the chosen hardware.

---

## 2. `device/interface.go` is the Contract

Every compute operation supported by the platform is declared on `device.Backend` and its embedded interfaces in `device/interface.go`. 

### 2.1 The Closed-World Architecture
*   There are no dynamic kernel registries, runtime string lookups, or plugin architectures.
*   If an operation is not defined as a method on `Backend`, it cannot be compiled or executed.
*   All custom operations must be added to `device/interface.go` via pull request and implemented across all target backends: CPU (scalar + SIMD), Metal, CUDA, and XLA.
*   `device.HostBackend` (`PosPop`) is host-only preprocessing on CPU and is excluded from Metal, CUDA, and XLA backend requirements.
*   Implementation contracts and gap tracking for XLA live in `XLA_GAPS.md`. XLA has equal standing with every other backend per `AGENTS.md`.

### 2.2 The Zero-Host-Sync Principle
*   No mathematical operation in `device.Backend` returns a Go scalar type (such as `float32`, `float64`, `int32`, or `bool`) to the CPU host.
*   All operations that reduce tensors to single values (reductions, dot products, loss functions, sampling indices, similarity metrics) must write their output to a destination pointer (`dst unsafe.Pointer`) on the device.
*   The execution pipeline remains entirely non-blocking. Host reads (e.g., for logging metrics or loss values) are treated as explicit, asynchronous synchronization queries scheduled at the end of an execution boundary.

`device.HostBackend` holds host-only preprocessing (`PosPop`) and is not subject to the device workspace or async dispatch model. See §2.3 `pospop/`.

### 2.3 Backend Package Structure

Every `device.Backend` implementation — CPU, Metal, CUDA, XLA — follows the same directory layout. The tree mirrors the embedded interfaces in `device/interface.go`. `device/cpu` is the reference layout for ISA organization and dispatch wiring; `device/metal` uses the same interface-family tree with the **Metal quintet** per family (§2.3.1).

**Principle:** one subdirectory per interface family (`device/interface.go` embed). Within each family, split Go and kernel files by **semantic domain** — not dtype, not arity. The root package holds only `Backend` composition and shared device infrastructure.

#### Top-Level Backend Package

```
device/<backend>/                 # cpu | metal | cuda | xla
├── backend.go
├── backend_config.go             # optional; omit if empty
├── allocate.go                   # GPU/XLA only: buffer pool, queues, residency
├── kernels.go                    # Metal/CUDA only: pipeline / module cache
├── pospop/                       # CPU host only (device.HostBackend); omit on Metal/CUDA/XLA
├── activation/
├── elementwise/
├── reduction/
├── dot/
├── matmul/
├── pool/
├── convolution/
├── dropout/
├── losses/
├── sampling/
├── embedding/
├── normalization/
├── layernorm/
├── rope/
├── hawkes/
├── physics/
├── causal/
├── masking/
├── attention/
├── vsa/
├── active_inference/
├── predictive_coding/
├── dequant/
└── quant/
```

**`backend.go`** — sole root file that touches `device.Backend`:

```go
package cpu // metal | cuda | xla

type Backend struct {
	ctx    context.Context
	cancel context.CancelFunc
	// backend-specific state (pool, device, queue, compile cache, …)

	activation.Activation
	elementwise.Elementwise
	reduction.Reduction
	dot.Dot
	matmul.Matmul
	pool.Pool
	convolution.Convolution
	dropout.Dropout
	losses.Losses
	sampling.Sampling
	embedding.Embedding
	normalization.Normalization
	layernorm.LayerNorm
	rope.RoPE
	hawkes.Hawkes
	physics.Physics
	causal.Causal
	masking.Masking
	attention.Attention
	vsa.VSA
	active_inference.ActiveInference
	predictive_coding.PredictiveCoding
	dequant.Dequant
	quant.Quant
}

// cpu.Backend also embeds pospop.PosPop via device.HostBackend.
// Metal, CUDA, and XLA Backend structs do not.

func NewBackend(ctx context.Context, …) (*Backend, error)
func (backend *Backend) Close() error

var _ device.Backend = (*Backend)(nil)
```

No `backend_<family>.go` forwarding files. Embedded family types promote their methods.

**`backend_config.go`** (optional):

```go
type Config struct { … }
func DefaultConfig() Config
```

**`allocate.go`** (GPU/XLA/CPU workspace):

```go
const workspaceAlign = 64

func (backend *Backend) allocateAligned(size int64) (unsafe.Pointer, error)
// CPU: posix_memalign / _aligned_malloc — base must satisfy uintptr(ptr) % 64 == 0
// CUDA/Metal: device allocator; validate returned pointer alignment at init
func (backend *Backend) release(ptr unsafe.Pointer)
```

**`kernels.go`** (Metal/CUDA):

```go
func (backend *Backend) pipeline(name string) (…, error)
```

---

#### Shared files in every `<family>/` subpackage

Every operation family directory contains this skeleton. **Domain Go files** (listed per family below) sit alongside it. A *domain* is a semantically coherent group of operations (`math`, `gated`, `conv2d`, `free_energy`) — not an arity label (`unary`, `binary`).

```
<family>/
├── <family>.go                   # type definition
├── dispatch.go                   # dtype switch (format dtype.DType) + ISA pick | GPU launch | XLA lower
├── generic.go                    # scalar / host reference; switch format inside each method
├── kernels.go                    # function-pointer table wired by select_*.go
├── select_amd64.go               # CPU: CPUID → AVX-512 | AVX2 | SSE2
├── select_arm64.go               # CPU: NEON
├── select_generic.go             # CPU: portable fallback
├── {domain}.go                   # one file per semantic domain (methods listed below)
├── {domain}_avx512_amd64.s       # kernel stem matches domain name × ISA
├── {domain}_avx2_amd64.s
├── {domain}_sse2_amd64.s
├── {domain}_neon_arm64.s
├── {domain}.metal                # Metal: domain kernels (includes family hub); see §2.3.1
├── {family}_bridge_darwin.go     # Metal: cgo bridge; includes native/*.m
├── native/{domain}.m             # Metal: ObjC dispatch for domain
├── {domain}.cu                   # CUDA: one source file per domain
├── bridge.cu                     # CUDA: driver launch bridge
├── lower.go                      # XLA: HLO lowering; switch format inside
├── <family>_parity_test.go
└── <family>_bench_test.go
```

Public methods take `unsafe.Pointer` and `format dtype.DType` only. **No dtype-prefixed filenames** (`f32_`, `f16_`, `bf16_`). **No arity-based domain names** (`unary`, `binary`) when a semantic name exists (`math`, `arithmetic`). Assembly/kernel stem matches the domain Go file: `{domain}_{isa}_{arch}.s`. Never `remaining`, `missing`, `extra`, `misc`.

On Metal, CPU-only artifacts (`select_*.go`, `*.s`, `dispatch.go`, `generic.go`, `kernels.go`, `lower.go`) are omitted until a family gains a CPU-style reference path; the quintet in §2.3.1 is required instead.

Public method signatures below are the **target** `device/interface.go` contract. All reduction, dot, loss, sampling, and similarity ops write results to `dst unsafe.Pointer` on the device per §2.2 — no Go scalar returns.

---

#### 2.3.1 Metal family quintet (`device/metal`)

Each interface family under `device/metal/` is a self-contained package. The **family hub** and every **semantic domain** follow the same five-role pattern (the quintet):

| Role | Family hub | Per domain `{domain}` |
|------|------------|------------------------|
| Go type | `{family}.go` (`darwin && cgo`) | `{domain}.go` |
| Stub | `{family}_stub.go` | `{domain}_stub.go` |
| C header | `{family}.h` | `{domain}.h` |
| MSL source | `{family}.metal` — storage, templates, kernel `#define` macros | `{domain}.metal` — `#include "{family}.metal"` + instantiations |
| Bridge | `{family}_bridge_darwin.go` — `#include "native/{family}.m"` then each `native/{domain}.m` | (included by family bridge) |
| Native dispatch | `native/{family}.m` — shared status, kernel naming, pipeline prepare | `native/{domain}.m` — `#include "{domain}.h"` + dispatch bodies |

**Layout example (`activation/`):**

```
device/metal/activation/
├── activation.go
├── activation_stub.go
├── activation.h
├── activation.metal
├── activation_bridge_darwin.go
├── native/
│   ├── activation.m
│   ├── standard.m
│   ├── softmax.m
│   ├── parametric.m
│   └── gated.m
├── standard.go
├── standard_stub.go
├── standard.h
├── standard.metal
└── …                               # same quintet per domain
```

**Rules:**

*   **No orphan `.metal` files.** Every `.metal` is `{family}.metal` or a declared domain `{domain}.metal`.
*   **Hub vs domain.** Templates, storage, and `#define` kernel macros live in `{family}.metal`. Domain files include the hub and emit kernel instantiations only.
*   **Native includes.** Use `#include "domain.h"`, not `#include "../domain.h"`. Shared C types: `device/metal/internal/bridge/core.h`, `core_private.h`.
*   **No cross-family bridges.** One `{family}_bridge_darwin.go` per family. Exception: `convolution` may `#include "../../pool/native/pool.m"` so `conv2d` calls `metal_vision_dispatch` from `pool`.
*   **Masking on Metal** lives under `attention/` (`masking.go`, `masking.metal`, `native/masking.m`), not a top-level `masking/` package.
*   **`internal/`** — shared bridge headers, runtime-only MSL (`internal/runtime/*.metal`), and `internal/metallibgen/` (builds `kernels.metallib` from all `**/*.metal` under `device/metal/`).

**Quintet-complete families:** `activation`, `elementwise`, `reduction`, `dot`, `matmul`, `pool`, `convolution`, `dropout`, `losses`, `sampling`, `embedding`, `normalization`, `layernorm`, `rope`, `hawkes`, `physics`, `causal`, `attention`, `vsa`, `active_inference`, `predictive_coding`, `dequant`, `quant`.

---

#### `pospop/` — host-only preprocessing (`device.HostBackend`)

`PosPop` is **not** part of the device execution contract. It operates on host-resident byte buffers and Go strings during tokenizer/preprocessing — never inside the async execution loop on Metal, CUDA, or XLA.

*   `device.Backend` (Metal, CUDA, XLA, CPU compute path) does **not** embed `PosPop`.
*   `device.HostBackend` embeds `PosPop` and is implemented only on the CPU host path (`device/cpu`).
*   Metal, CUDA, and XLA backends omit the `pospop/` subdirectory entirely.

**Files (CPU host only):** `pospop.go`, `count.go`, `string.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `string_avx512_amd64.s`, `string_avx2_amd64.s`, `string_sse2_amd64.s`, `string_neon_arm64.s`, `count8_avx512_amd64.s`, …, `pospop_parity_test.go`, `pospop_bench_test.go`.

**`pospop.go`:** `type PosPop struct { … }`

**`string.go`** — host string scan (uses Go `string`; never dispatched to GPU/XLA):

```go
func (posPop *PosPop) CountString(counts *[8]int, str string)
```

**`count.go`** — raw buffer population counts (`buf` is host-pinned memory, not a Go slice header passed to device code):

```go
func (posPop *PosPop) Count8(counts, buf unsafe.Pointer, byteCount int)
func (posPop *PosPop) Count16(counts, buf unsafe.Pointer, elementCount int)
func (posPop *PosPop) Count32(counts, buf unsafe.Pointer, elementCount int)
func (posPop *PosPop) Count64(counts, buf unsafe.Pointer, elementCount int)
```

---

#### `activation/` — implements `device.Activation`

Nonlinear transforms, softmax, parametric activations, and gated linear units.

**Files:** `activation.go`, `standard.go`, `softmax.go`, `parametric.go`, `gated.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `standard_avx512_amd64.s`, … (per ISA/domain), `standard.metal`, `softmax.metal`, `parametric.metal`, `gated.metal`, `activation.metal`, `activation_bridge_darwin.go`, `native/activation.m`, `native/standard.m`, …, `activation_parity_test.go`, `activation_bench_test.go`, `lower.go`.

**`activation.go`:** `type Activation struct { … }`

**`standard.go`** — fixed-shape nonlinearities (Exp, ReLU, Gelu, …):

```go
func (activation *Activation) Exp(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) Log(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) Log1p(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) Expm1(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) Sigmoid(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) LogSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) Tanh(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) Silu(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) Swish(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) GeluTanh(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) Gelu(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) LeakyReLU(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) ELU(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) CELU(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) SELU(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) Softplus(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) Mish(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) Softsign(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) HardSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) HardSwish(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) HardTanh(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) HardGelu(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) QuickGelu(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) TanhShrink(dst, src unsafe.Pointer, count int, format dtype.DType)
```

**`softmax.go`:**

```go
func (activation *Activation) Softmax(dst, src unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) LogSoftmax(dst, src unsafe.Pointer, count int, format dtype.DType)
```

**`parametric.go`** — activations with scalar/vector parameters:

```go
func (activation *Activation) PReLU(dst, src unsafe.Pointer, count int, format dtype.DType, negativeSlope float32)
func (activation *Activation) PReLUV(dst, src, slopes unsafe.Pointer, count int, format dtype.DType, slopeCount int)
func (activation *Activation) LeakyReLUSlope(dst, src unsafe.Pointer, count int, format dtype.DType, negativeSlope float32)
func (activation *Activation) ELUAlpha(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32)
func (activation *Activation) CELUAlpha(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32)
func (activation *Activation) Threshold(dst, src unsafe.Pointer, count int, format dtype.DType, threshold float32)
func (activation *Activation) HardTanhRange(dst, src unsafe.Pointer, count int, format dtype.DType, minVal, maxVal float32)
func (activation *Activation) Snake(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32)
func (activation *Activation) SnakeParametric(dst, src unsafe.Pointer, count int, format dtype.DType, alpha, beta float32)
func (activation *Activation) HardShrink(dst, src unsafe.Pointer, count int, format dtype.DType, lambda float32)
func (activation *Activation) SoftShrink(dst, src unsafe.Pointer, count int, format dtype.DType, lambda float32)
func (activation *Activation) RReLU(dst, src unsafe.Pointer, count int, format dtype.DType, lower, upper float32)
```

**`gated.go`** — gated linear units on packed tensors:

```go
func (activation *Activation) GLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
func (activation *Activation) GeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
func (activation *Activation) GeGLUTanh(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
func (activation *Activation) SwiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
func (activation *Activation) ReGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
func (activation *Activation) SiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
func (activation *Activation) LinGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
func (activation *Activation) SeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType)
```

**`gated_packed.go`** — gated units on separate gate/up tensors:

```go
func (activation *Activation) GLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) GeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) GeGLUTanhTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) SwiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) ReGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) SiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) LinGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
func (activation *Activation) SeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType)
```

---

#### `elementwise/` — implements `device.Elementwise`

Per-element tensor algebra and math primitives.

**Files:** `elementwise.go`, `arithmetic.go`, `math.go`, `axpy.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `arithmetic_avx512_amd64.s`, `arithmetic_avx2_amd64.s`, `arithmetic_sse2_amd64.s`, `arithmetic_neon_arm64.s`, `math_avx512_amd64.s`, `math_avx2_amd64.s`, `math_sse2_amd64.s`, `math_neon_arm64.s`, `axpy_avx512_amd64.s`, `axpy_avx2_amd64.s`, `axpy_sse2_amd64.s`, `axpy_neon_arm64.s`, `arithmetic.metal`, `math.metal`, `axpy.metal`, `bridge_darwin.m`, `elementwise_parity_test.go`, `elementwise_bench_test.go`, `lower.go`.

**`elementwise.go`:** `type Elementwise struct { … }`

**`arithmetic.go`** — two-operand elementwise ops:

```go
func (elementwise *Elementwise) Add(dst, left, right unsafe.Pointer, count int, format dtype.DType)
func (elementwise *Elementwise) Sub(dst, left, right unsafe.Pointer, count int, format dtype.DType)
func (elementwise *Elementwise) Mul(dst, left, right unsafe.Pointer, count int, format dtype.DType)
func (elementwise *Elementwise) Div(dst, left, right unsafe.Pointer, count int, format dtype.DType)
func (elementwise *Elementwise) Max(dst, left, right unsafe.Pointer, count int, format dtype.DType)
func (elementwise *Elementwise) Min(dst, left, right unsafe.Pointer, count int, format dtype.DType)
```

**`math.go`** — single-operand math primitives:

```go
func (elementwise *Elementwise) Abs(dst, src unsafe.Pointer, count int, format dtype.DType)
func (elementwise *Elementwise) Neg(dst, src unsafe.Pointer, count int, format dtype.DType)
func (elementwise *Elementwise) Sqrt(dst, src unsafe.Pointer, count int, format dtype.DType)
```

**`axpy.go`:**

```go
func (elementwise *Elementwise) Axpy(y, x unsafe.Pointer, count int, alpha float32, format dtype.DType)
func (elementwise *Elementwise) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) // delegates to activation.Activation.ReLU
```

`Elementwise.ReLU` remains on the interface for now; implementation delegates to `activation/` — canonical home for ReLU is `activation/standard.go`.

---

#### `reduction/` — implements `device.Reduction`

**Files:** `reduction.go`, `aggregate.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `aggregate_avx512_amd64.s`, `aggregate_avx2_amd64.s`, `aggregate_sse2_amd64.s`, `aggregate_neon_arm64.s`, `aggregate.metal`, `bridge_darwin.m`, `reduction_parity_test.go`, `reduction_bench_test.go`, `lower.go`.

**`reduction.go`:** `type Reduction struct { … }`

**`aggregate.go`:**

```go
func (reduction *Reduction) Sum(dst, values unsafe.Pointer, count int, format dtype.DType)
func (reduction *Reduction) Prod(dst, values unsafe.Pointer, count int, format dtype.DType)
func (reduction *Reduction) ReduceMin(dst, values unsafe.Pointer, count int, format dtype.DType)
func (reduction *Reduction) ReduceMax(dst, values unsafe.Pointer, count int, format dtype.DType)
func (reduction *Reduction) L1Norm(dst, values unsafe.Pointer, count int, format dtype.DType)
```

---

#### `dot/` — implements `device.Dot`

**Files:** `dot.go`, `inner_product.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `inner_product_avx512_amd64.s`, `inner_product_avx2_amd64.s`, `inner_product_sse2_amd64.s`, `inner_product_neon_arm64.s`, `inner_product.metal`, `bridge_darwin.m`, `dot_parity_test.go`, `dot_bench_test.go`, `lower.go`.

**`dot.go`:** `type Dot struct { … }`

**`inner_product.go`:**

```go
func (dot *Dot) Dot(dst, left, right unsafe.Pointer, count int, format dtype.DType)
```

---

#### `matmul/` — implements `device.Matmul`

**Files:** `matmul.go`, `product.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `product_avx512_amd64.s`, `product_avx2_amd64.s`, `product_sse2_amd64.s`, `product_neon_arm64.s`, `product.metal`, `bridge_darwin.m`, `matmul_parity_test.go`, `matmul_bench_test.go`, `lower.go`.

**`matmul.go`:** `type Matmul struct { … }`

**`product.go`:**

```go
func (matmul *Matmul) Matmul(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
	format dtype.DType,
)
```

---

#### `pool/` — implements `device.Pool`

**Files:** `pool.go`, `maxpool.go`, `avgpool.go`, `adaptive.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `maxpool_avx512_amd64.s`, `avgpool_avx512_amd64.s`, `adaptive_avx512_amd64.s`, (per ISA: `_avx2_amd64.s`, `_sse2_amd64.s`, `_neon_arm64.s`), `maxpool.metal`, `avgpool.metal`, `adaptive.metal`, `bridge_darwin.m`, `pool_parity_test.go`, `pool_bench_test.go`, `lower.go`.

**`pool.go`:** `type Pool struct { … }`

**`maxpool.go`:**

```go
func (pool *Pool) MaxPool2D(
	config device.PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
)
```

**`avgpool.go`:**

```go
func (pool *Pool) AvgPool2D(
	config device.PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
)
```

**`adaptive.go`:**

```go
func (pool *Pool) AdaptiveMaxPool2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
)
func (pool *Pool) AdaptiveAvgPool2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
)
```

---

#### `convolution/` — implements `device.Convolution`

**Files:** `convolution.go`, `conv2d.go`, `conv1d.go`, `conv3d.go`, `transpose.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `conv2d_avx512_amd64.s`, `conv1d_avx512_amd64.s`, `conv3d_avx512_amd64.s`, `transpose_avx512_amd64.s`, (per ISA: `_avx2_amd64.s`, `_sse2_amd64.s`, `_neon_arm64.s`), `conv2d.metal`, `conv1d.metal`, `conv3d.metal`, `transpose.metal`, `bridge_darwin.m`, `convolution_parity_test.go`, `convolution_bench_test.go`, `lower.go`.

**`convolution.go`:** `type Convolution struct { … }`

**`conv2d.go`:**

```go
func (convolution *Convolution) Conv2D(
	config device.Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType,
)
```

**`conv1d.go`:**

```go
func (convolution *Convolution) Conv1D(
	config device.Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
	format dtype.DType,
)
```

**`conv3d.go`:**

```go
func (convolution *Convolution) Conv3D(
	config device.Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
	format dtype.DType,
)
```

**`transpose.go`:**

```go
func (convolution *Convolution) ConvTranspose2D(
	config device.Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType,
)
```

---

#### `dropout/` — implements `device.Dropout`

**Files:** `dropout.go`, `mask.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `mask_avx512_amd64.s`, (per ISA), `mask.metal`, `bridge_darwin.m`, `dropout_parity_test.go`, `dropout_bench_test.go`, `lower.go`.

**`dropout.go`:** `type Dropout struct { … }`

**`mask.go`:**

```go
func (dropout *Dropout) Dropout(
	dst, src unsafe.Pointer,
	count int,
	config device.DropoutConfig,
	format dtype.DType,
)
```

---

#### `losses/` — implements `device.Losses`

**Files:** `losses.go`, `regression.go`, `classification.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `regression_avx512_amd64.s`, `classification_avx512_amd64.s`, (per ISA), `regression.metal`, `classification.metal`, `bridge_darwin.m`, `losses_parity_test.go`, `losses_bench_test.go`, `lower.go`.

**`losses.go`:** `type Losses struct { … }`

**`regression.go`:**

```go
func (losses *Losses) MSE(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType)
func (losses *Losses) MAE(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType)
func (losses *Losses) Huber(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType)
```

**`classification.go`:**

```go
func (losses *Losses) BinaryCrossEntropy(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType)
func (losses *Losses) KLDivergence(dst, predictions, targets unsafe.Pointer, count int, format dtype.DType)
func (losses *Losses) CrossEntropy(
	dst unsafe.Pointer,
	logits unsafe.Pointer,
	targets unsafe.Pointer,
	batchSize, classes int,
	format dtype.DType,
)
```

---

#### `sampling/` — implements `device.Sampling`

**Files:** `sampling.go`, `greedy.go`, `nucleus.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `greedy_avx512_amd64.s`, `nucleus_avx512_amd64.s`, (per ISA), `greedy.metal`, `nucleus.metal`, `bridge_darwin.m`, `sampling_parity_test.go`, `sampling_bench_test.go`, `lower.go`.

**`sampling.go`:** `type Sampling struct { … }`

**`greedy.go`:**

```go
func (sampling *Sampling) GreedySample(dst, logits unsafe.Pointer, vocabSize int, format dtype.DType)
```

**`nucleus.go`** — top-k / top-p (nucleus) sampling:

```go
func (sampling *Sampling) TopKSample(
	dst, logits unsafe.Pointer,
	vocabSize int,
	config device.SamplingConfig,
	format dtype.DType,
)
func (sampling *Sampling) TopPSample(
	dst, logits unsafe.Pointer,
	vocabSize int,
	config device.SamplingConfig,
	format dtype.DType,
)
```

---

#### `embedding/` — implements `device.Embedding`

**Files:** `embedding.go`, `lookup.go`, `bag.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `lookup_avx512_amd64.s`, `bag_avx512_amd64.s`, (per ISA), `lookup.metal`, `bag.metal`, `bridge_darwin.m`, `embedding_parity_test.go`, `embedding_bench_test.go`, `lower.go`.

**`embedding.go`:** `type Embedding struct { … }`

**`lookup.go`:**

```go
func (embedding *Embedding) Lookup(
	table, indices, output unsafe.Pointer,
	vocab, hidden, indexCount int,
	format dtype.DType,
)
```

**`bag.go`:**

```go
func (embedding *Embedding) Bag(
	table, indices, offsets, output unsafe.Pointer,
	vocab, hidden, bagCount, indexCount int,
	format dtype.DType,
)
```

---

#### `normalization/` — implements `device.Normalization`

**Files:** `normalization.go`, `groupnorm.go`, `instancenorm.go`, `batchnorm.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `groupnorm_avx512_amd64.s`, `instancenorm_avx512_amd64.s`, `batchnorm_avx512_amd64.s`, (per ISA), `groupnorm.metal`, `instancenorm.metal`, `batchnorm.metal`, `bridge_darwin.m`, `normalization_parity_test.go`, `normalization_bench_test.go`, `lower.go`.

**`normalization.go`:** `type Normalization struct { … }`

**`groupnorm.go`:**

```go
func (normalization *Normalization) GroupNorm(
	config device.GroupNormConfig,
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
)
```

**`instancenorm.go`:**

```go
func (normalization *Normalization) InstanceNorm(
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
)
```

**`batchnorm.go`:**

```go
func (normalization *Normalization) BatchNormEval(
	input, scale, bias, mean, variance, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
)
```

---

#### `layernorm/` — implements `device.LayerNorm`

**Files:** `layernorm.go`, `layer.go`, `rms.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `layer_avx512_amd64.s`, `rms_avx512_amd64.s`, (per ISA), `layer.metal`, `rms.metal`, `bridge_darwin.m`, `layernorm_parity_test.go`, `layernorm_bench_test.go`, `lower.go`.

**`layernorm.go`:** `type LayerNorm struct { … }`

**`layer.go`:**

```go
func (layernorm *LayerNorm) LayerNorm(
	input, scale, bias, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
)
```

**`rms.go`:**

```go
func (layernorm *LayerNorm) RMSNorm(
	input, scale, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
)
```

---

#### `rope/` — implements `device.RoPE`

**Files:** `rope.go`, `rotate.go`, `pairs.go`, `config.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `rotate_avx512_amd64.s`, `pairs_avx512_amd64.s`, (per ISA), `rotate.metal`, `pairs.metal`, `bridge_darwin.m`, `rope_parity_test.go`, `rope_bench_test.go`, `lower.go`.

**`rope.go`:** `type RoPE struct { … }`

**`rotate.go`:**

```go
func (rope *RoPE) RoPE(
	config device.RoPEConfig,
	input, output unsafe.Pointer,
	seqLen, numHeads, headDim int,
	format dtype.DType,
)
```

**`pairs.go`:**

```go
func (rope *RoPE) RoPEPairs(
	output, input, cosBuffer, sinBuffer unsafe.Pointer,
	halfDim int,
	format dtype.DType,
)
```

---

#### `hawkes/` — implements `device.Hawkes`

**Files:** `hawkes.go`, `intensity.go`, `kernel.go`, `likelihood.go`, `markov.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `intensity_avx512_amd64.s`, `kernel_avx512_amd64.s`, `likelihood_avx512_amd64.s`, `markov_avx512_amd64.s`, (per ISA), `intensity.metal`, `kernel.metal`, `likelihood.metal`, `markov.metal`, `bridge_darwin.m`, `hawkes_parity_test.go`, `hawkes_bench_test.go`, `lower.go`.

**`hawkes.go`:** `type Hawkes struct { … }`

**`intensity.go`:**

```go
func (hawkes *Hawkes) HawkesIntensity(
	eventTimes, queryTimes, output unsafe.Pointer,
	eventCount, queryCount int,
	mu, alpha, beta float32,
	format dtype.DType,
)
```

**`kernel.go`:**

```go
func (hawkes *Hawkes) HawkesKernelMatrix(
	eventTimes, output unsafe.Pointer,
	eventCount int,
	alpha, beta float32,
	format dtype.DType,
)
```

**`likelihood.go`:**

```go
func (hawkes *Hawkes) HawkesLogLikelihood(
	eventTimes unsafe.Pointer,
	eventCount int,
	totalT, mu, alpha, beta float32,
	output unsafe.Pointer,
	format dtype.DType,
)
```

**`markov.go`:**

```go
func (hawkes *Hawkes) MarkovMutualInformation(
	joint, output unsafe.Pointer,
	xCount, yCount int,
	format dtype.DType,
)
func (hawkes *Hawkes) MarkovBlanketPartition(
	adjacency, internal, output unsafe.Pointer,
	nodeCount, internalCount int,
	format dtype.DType,
)
```

---

#### `physics/` — implements `device.Physics`

**Files:** `physics.go`, `differential.go`, `spectral.go`, `quantum.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `differential_avx512_amd64.s`, `spectral_avx512_amd64.s`, `quantum_avx512_amd64.s`, (per ISA), `differential.metal`, `spectral.metal`, `quantum.metal`, `bridge_darwin.m`, `physics_parity_test.go`, `physics_bench_test.go`, `lower.go`.

**`physics.go`:** `type Physics struct { … }`

**`differential.go`** — spatial differential operators:

```go
func (physics *Physics) Laplacian(input, output unsafe.Pointer, dims []int, spacing float32, format dtype.DType)
func (physics *Physics) Laplacian4(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
func (physics *Physics) Grad1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
func (physics *Physics) Divergence1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
```

**`spectral.go`:**

```go
func (physics *Physics) FFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType)
func (physics *Physics) IFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType)
```

**`quantum.go`:**

```go
func (physics *Physics) QuantumPotential(density, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
func (physics *Physics) BohmianVelocity(phase, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
func (physics *Physics) MadelungContinuity(
	density, velocity, residual unsafe.Pointer,
	count int,
	spacing float32,
	format dtype.DType,
)
```

---

#### `causal/` — implements `device.Causal`

**Files:** `causal.go`, `matrix.go`, `adjustment.go`, `intervention.go`, `dag.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `matrix_avx512_amd64.s`, `adjustment_avx512_amd64.s`, `intervention_avx512_amd64.s`, `dag_avx512_amd64.s`, (per ISA), `matrix.metal`, `adjustment.metal`, `intervention.metal`, `dag.metal`, `bridge_darwin.m`, `causal_parity_test.go`, `causal_bench_test.go`, `lower.go`.

**`causal.go`:** `type Causal struct { … }`

**`matrix.go`:**

```go
func (causal *Causal) Cholesky(input, output unsafe.Pointer, matrixOrder int, format dtype.DType)
```

**`adjustment.go`** — backdoor / frontdoor identification:

```go
func (causal *Causal) BackdoorAdjustment(
	conditional, marginalZ, output unsafe.Pointer,
	xCount, zCount, yCount int,
	format dtype.DType,
)
func (causal *Causal) FrontdoorAdjustment(
	mediatorGivenX, outcomeGivenXM, marginalX, output unsafe.Pointer,
	xCount, mediatorCount, yCount int,
	format dtype.DType,
)
```

**`intervention.go`** — do-calculus and treatment effects:

```go
func (causal *Causal) DoIntervene(
	adjacency, intervened, output unsafe.Pointer,
	nodeCount, intervenedCount int,
	format dtype.DType,
)
func (causal *Causal) CATE(treated, control, output unsafe.Pointer, count int, format dtype.DType)
func (causal *Causal) Counterfactual(
	observedY, observedX, counterfactualX, output unsafe.Pointer,
	count int,
	slope float32,
	format dtype.DType,
)
func (causal *Causal) IVEstimate(
	instrument, treatment, outcome unsafe.Pointer,
	count int,
	output unsafe.Pointer,
	format dtype.DType,
)
```

**`dag.go`:**

```go
func (causal *Causal) DAGMarkovFactorization(
	conditionals unsafe.Pointer,
	conditionalCount int,
	output unsafe.Pointer,
	format dtype.DType,
)
func (causal *Causal) MarkovFlowActive(
	mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	format dtype.DType,
)
func (causal *Causal) MarkovFlowInternal(
	mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	format dtype.DType,
)
```

---

#### `masking/` — implements `device.Masking`

**Files:** `masking.go`, `apply.go`, `causal.go`, `alibi.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `apply_avx512_amd64.s`, `causal_avx512_amd64.s`, `alibi_avx512_amd64.s`, (per ISA), `apply.metal`, `causal.metal`, `alibi.metal`, `bridge_darwin.m`, `masking_parity_test.go`, `masking_bench_test.go`, `lower.go`.

**`masking.go`:** `type Masking struct { … }`

**`apply.go`:**

```go
func (masking *Masking) ApplyMask(input, mask, output unsafe.Pointer, count int, format dtype.DType)
```

**`causal.go`:**

```go
func (masking *Masking) CausalMask(output unsafe.Pointer, seqQ, seqK int, format dtype.DType)
```

**`alibi.go`:**

```go
func (masking *Masking) ALiBiBias(scores, slope, output unsafe.Pointer, seqQ, seqK int, format dtype.DType)
```

---

#### `attention/` — implements `device.Attention`

**Files:** `attention.go`, `scaled_dot_product.go`, `flash.go`, `multihead.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `scaled_dot_product_avx512_amd64.s`, `flash_avx512_amd64.s`, `multihead_avx512_amd64.s`, (per ISA), `scaled_dot_product.metal`, `flash.metal`, `multihead.metal`, `bridge_darwin.m`, `attention_parity_test.go`, `attention_bench_test.go`, `lower.go`.

**`attention.go`:** `type Attention struct { … }`

**`scaled_dot_product.go`:**

```go
func (attention *Attention) ScaledDotProductAttention(
	config device.FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
	format dtype.DType,
)
```

**`flash.go`:**

```go
func (attention *Attention) FlashAttention(
	config device.FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
	format dtype.DType,
)
```

**`multihead.go`:**

```go
func (attention *Attention) MultiHeadAttention(
	config device.MultiHeadAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
)
```

---

#### `vsa/` — implements `device.VSA`

**Files:** `vsa.go`, `bind.go`, `bundle.go`, `permute.go`, `similarity.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `bind_avx512_amd64.s`, `bundle_avx512_amd64.s`, `permute_avx512_amd64.s`, `similarity_avx512_amd64.s`, (per ISA), `bind.metal`, `bundle.metal`, `permute.metal`, `similarity.metal`, `bridge_darwin.m`, `vsa_parity_test.go`, `vsa_bench_test.go`, `lower.go`.

**`vsa.go`:** `type VSA struct { … }`

**`bind.go`:**

```go
func (vsa *VSA) Bind(left, right, output unsafe.Pointer, count int, format dtype.DType)
```

**`bundle.go`:**

```go
func (vsa *VSA) Bundle(left, right, output unsafe.Pointer, count int, format dtype.DType)
```

**`permute.go`:**

```go
func (vsa *VSA) Permute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType)
func (vsa *VSA) InversePermute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType)
```

**`similarity.go`:**

```go
func (vsa *VSA) Similarity(dst, left, right unsafe.Pointer, count int, format dtype.DType)
```

---

#### `active_inference/` — implements `device.ActiveInference`

**Files:** `active_inference.go`, `free_energy.go`, `belief.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `free_energy_avx512_amd64.s`, `belief_avx512_amd64.s`, (per ISA), `free_energy.metal`, `belief.metal`, `bridge_darwin.m`, `active_inference_parity_test.go`, `active_inference_bench_test.go`, `lower.go`.

**`active_inference.go`:** `type ActiveInference struct { … }`

**`free_energy.go`:**

```go
func (activeInference *ActiveInference) FreeEnergy(
	likelihood, posterior, prior, output unsafe.Pointer,
	count int,
	format dtype.DType,
)
func (activeInference *ActiveInference) ExpectedFreeEnergy(
	predictedObs, preferredObs, predictedState, output unsafe.Pointer,
	obsCount, stateCount int,
	format dtype.DType,
)
```

**`belief.go`:**

```go
func (activeInference *ActiveInference) BeliefUpdate(likelihood, prior, output unsafe.Pointer, count int, format dtype.DType)
func (activeInference *ActiveInference) PrecisionWeight(errors, precision, output unsafe.Pointer, count int, format dtype.DType)
```

---

#### `predictive_coding/` — implements `device.PredictiveCoding`

**Files:** `predictive_coding.go`, `forward.go`, `learning.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `forward_avx512_amd64.s`, `learning_avx512_amd64.s`, (per ISA), `forward.metal`, `learning.metal`, `bridge_darwin.m`, `predictive_coding_parity_test.go`, `predictive_coding_bench_test.go`, `lower.go`.

**`predictive_coding.go`:** `type PredictiveCoding struct { … }`

**`forward.go`:**

```go
func (predictiveCoding *PredictiveCoding) Prediction(
	weights, representation, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType,
)
func (predictiveCoding *PredictiveCoding) PredictionError(
	observed, predicted, output unsafe.Pointer,
	count int,
	format dtype.DType,
)
```

**`learning.go`:**

```go
func (predictiveCoding *PredictiveCoding) UpdateRepresentation(
	config device.PredictiveCodingConfig,
	weights, representation, predictionError, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType,
)
func (predictiveCoding *PredictiveCoding) UpdateWeights(
	config device.PredictiveCodingConfig,
	weights, representation, predictionError, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType,
)
```

---

#### `dequant/` — implements `device.Dequant`

**Files:** `dequant.go`, `int8.go`, `int4.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `int8_avx512_amd64.s`, `int4_avx512_amd64.s`, (per ISA), `int8.metal`, `int4.metal`, `bridge_darwin.m`, `dequant_parity_test.go`, `dequant_bench_test.go`, `lower.go`.

**`dequant.go`:** `type Dequant struct { … }`

**`int8.go`:**

```go
func (dequant *Dequant) Dequant(dst, src unsafe.Pointer, count int, config device.DequantInt8Config, dstFormat, srcFormat dtype.DType)
```

**`int4.go`:**

```go
func (dequant *Dequant) Dequant4(dst, src unsafe.Pointer, pairCount int, config device.DequantInt4Config, dstFormat, srcFormat dtype.DType)
```

Note: `int8`/`int4` here name the **quantization scheme**, not a floating dtype — distinct from banned `f32_`/`f16_`/`bf16_` prefixes.

---

#### `quant/` — implements `device.Quant`

**Files:** `quant.go`, `int8.go`, `dispatch.go`, `generic.go`, `kernels.go`, `select_amd64.go`, `select_arm64.go`, `select_generic.go`, `int8_avx512_amd64.s`, (per ISA), `int8.metal`, `bridge_darwin.m`, `quant_parity_test.go`, `quant_bench_test.go`, `lower.go`.

**`quant.go`:** `type Quant struct { … }`

**`int8.go`:**

```go
func (quant *Quant) Quant(dst, src unsafe.Pointer, count int, config device.DequantInt8Config, dstFormat, srcFormat dtype.DType)
```

---

#### Backend-Specific Kernel Artifacts

| Backend | Kernel files live in | Bridge / runtime |
|---------|---------------------|------------------|
| CPU     | `<family>/*.s` + `select_*.go` | none (direct assembly) |
| Metal   | `<family>/{family}.metal` + `<family>/{domain}.metal`; runtime MSL in `internal/runtime/` | `<family>/{family}_bridge_darwin.go` + `native/*.m`; `internal/metallibgen` → `kernels.metallib` |
| CUDA    | `<family>/*.cu` | `<family>/bridge.cu` or single bridge per family |
| XLA     | `<family>/lower.go` | HLO builder + PJRT compile cache in `device/xla/` |

GPU backends may keep a small amount of top-level infrastructure (`allocate.go`, pipeline cache, metallib build) that is not tied to a single interface family. That infrastructure must not contain operation implementations.

#### Metal anti-patterns (prohibited)

The following must not reappear in `device/metal/`:

*   **Catch-all backend files:** `device_missing_darwin.go`, `device_remaining_darwin.go`, `device_dispatch_darwin.go`, `device_backend_stub_ops.go`.
*   **Root forwarding shims:** `backend_activation.go`, `backend_elementwise.go`, and similar one-line delegators.
*   **Flat root sprawl:** per-op `*_darwin.go` or `bridge_*.m` files at `device/metal/` root instead of inside the matching family subpackage.
*   **Cross-family bridges:** one `bridge_*.m` spanning Attention, RoPE, and Embedding. Split by interface family (§2.3.1).
*   **Monolithic MSL:** single `{family}.metal` or `vision.metal` / `research.metal` holding multiple semantic domains — split into hub + domain `.metal` files.
*   **Orphan kernels:** `.metal` / `.m` / `.h` / `.go` without a matching quintet sibling (e.g. `.metal` with no `{domain}.go` + `native/{domain}.m`).
*   **Dtype-prefixed filenames** and **arity-based domain names** (same rules as §2.3 CPU layout).

Adding a method to `device/interface.go` requires: subpackage entry in **every** backend, kernel(s) for **every** dtype and target, parity test, benchmark — in the canonical paths above. Not a line in `device_remaining`.

---

## 3. Kernel Dispatches and the DType Multiplexer

Each static `Backend` method exposes a single entry point. Tensor data arrives as `unsafe.Pointer`. Extents arrive as explicit dimension or stride arguments. The active precision arrives as `format dtype.DType`.

```go
func (activation *Activation) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	switch format {
	case dtype.Float32, dtype.Float16, dtype.BFloat16:
		activation.standardKernel.relu(dst, src, count, format) // domain: standard; ISA picked at init
	default:
		panic("unsupported dtype format")
	}
}
```

Rules:
*   Methods must not be split into precision-specific variants at the public interface level (e.g., no `ReLUF32` vs `ReLUBF16`).
*   Every supported `dtype.DType` must have a deterministic execution path implemented.
*   `Matmul` and `Linear` are separate operations. `Linear` carries weight representations and projection parameters, allowing the optimizer to fold downstream biases or scales directly into the weight matrices before execution.

### 3.1 XLA Backend Target

XLA is the fifth execution backend. It is not a bypass around the manifesto compiler, the closed-world `Backend` contract, or `PortType` unification. It is how operations run when the active device is `tensor.XLA`.

**Responsibilities:**

*   Every `device.Backend` method lowers to an XLA computation (HLO), compiles through PJRT, and reads/writes XLA-resident buffers.
*   Parity is measured against the Go scalar reference for the same operation and dtype — never XLA-vs-XLA.
*   No host-side tensor math, no CUDA custom kernels inside XLA operation bodies, and no host transfers between fused subgraphs on the XLA device.

**Relation to manifesto codegen:**

The manifesto pipeline (unify → optimize → schedule → static workspace) is unchanged. What changes is the **codegen backend** selected at execution target:

| Target | Elementwise fusion clusters | Heavy ops (Matmul, Conv, Attention, …) |
|--------|----------------------------|----------------------------------------|
| CPU    | LLVM JIT → AVX-512 / AVX2 / SSE2 / NEON | Static SIMD kernels or scalar reference |
| Metal  | MSL JIT via `MTLLibrary` | MSL kernels |
| CUDA   | PTX JIT via `NVRTC` | CUDA kernels |
| XLA    | Lower FusionAST to HLO; XLA compiler fuses | Lower each `Backend` method to HLO; XLA compiler fuses |

On an XLA target, do **not** also emit the LLVM/MSL/CUDA elementwise JIT for the same fused cluster. Lower once to HLO and let the XLA compiler perform device fusion. The manifesto optimizer still forms `FusionAST`s and applies algebraic rewrites; XLA is the codegen and execution path for that target.

**Zero-host-sync alignment:** XLA naturally matches §2.2. Reductions, losses, dot products, and metrics write to `dst unsafe.Pointer` on the device. Host reads happen only at explicit execution boundaries.

**Build tag:** Real XLA runtime integration lives behind `//go:build xla`. Builds without the tag keep the stub bridge (`device/xla/bridge_stub.go`) returning `tensor.ErrNeedsPlatformSetup`. Complete implementation packets are specified in `XLA_GAPS.md`.

**Workspace vs. PJRT buffers:** §5 defines a flat byte workspace with static offsets for CPU, Metal, and CUDA. XLA/PJRT does not accept raw pointer arithmetic into that slab. PJRT owns device-resident memory as opaque `PjRtBuffer` handles.

When XLA is the active target, the executor keeps the **same virtual offset layout** in `Topology.Workspace`, but `device/xla` acts as a translation layer:

1.  At workspace init (before the execution loop), the XLA backend pre-allocates one `PjRtBuffer` per workspace slot. Slots are indexed by aligned byte offset (`slotIndex = offset >> 6` for 64-byte slots).
2.  At compile/session init, the executor builds a **flat pre-resolved slice** `[]*PjRtBuffer` (or `[]xlaBuffer`) mapping each slot index to its handle. **No map lookups, no locks at dispatch time.**
3.  During the execution loop, each node indexes the pre-resolved slice directly:
    ```go
    slotIndex := port.BaseOffset >> 6
    buffer := backend.resolvedBuffers[slotIndex]
    ```
4.  HLO lowering reads/writes through those handles. Host upload/download happens only at execution boundaries, not between ops in the loop.
5.  **In-place aliasing is not forced on XLA.** The host-side in-place optimizer (§5.1) is disabled for XLA targets. Output ports receive unique virtual offsets; XLA's internal allocation pass decides buffer reuse during HLO compilation.

```go
// device/xla/workspace.go — populated once at init, indexed during dispatch
type xlaWorkspace struct {
	resolvedBuffers []*xlaBuffer // len == workspaceSize >> 6
}
```

---

## 4. PortType, Composition, and Advanced Optimization

### 4.1 PortType Definition
A port represents a typed logical edge in the graph. It is defined as:

```go
type PortType struct {
    DType       dtype.DType
    ShapeSchema ShapeSchema        // Statically bound dimensions with dynamic symbols (e.g., [B, T, D])
    Layout      LayoutSchema       // Contiguous, Strided, Tiled, or Channel-First/Last
    Kind        SemanticKind       // e.g., HiddenState, Logits, AdjacencyMatrix, BeliefState
    Constraints []Constraint       // e.g., LastDim % 8 == 0, BatchSymbol == Parent.BatchSymbol
}
```

### 4.2 Composition via Adaptor Synthesis
When output port $\tau_1$ of Node A connects to input port $\tau_2$ of Node B, the compiler evaluates their unification:
1.  **Direct Unification:** If $\tau_1 == \tau_2$, the connection is made directly.
2.  **Adaptor Synthesis:** If the types do not unify, the compiler searches its adaptor database and synthesizes an explicit chain of transformation operations (e.g., `Transpose`, `Cast`, or `Reshape`).
3.  **Compilation Failure:** If no valid adaptor sequence can reconcile $\tau_1$ and $\tau_2$, compilation halts with a static type unification error. No implicit or silent runtime coercions are permitted.

### 4.3 Elementwise Fusion & JIT Compilation
The optimizer groups sequences of elementwise operations (e.g., `x -> Add -> Scale -> ReLU -> Mul`) into a single `FusionAST` node. 

Instead of executing these as separate loops, the `manifesto/codegen` backend compiles the `FusionAST` at startup:
*   **CPU Backends:** Generates optimized assembly loops using target-specific vector registers (AVX-512, AVX2, SSE2, NEON) via an internal assembler or LLVM, minimizing memory round-trips to CPU caches.
*   **GPU Backends (Metal / CUDA):** Generates raw Metal Shading Language (MSL) or CUDA PTX source code, JIT-compiles it using the native driver APIs (`MTLLibrary` or `NVRTC`), and loads the binary executable directly into the execution environment.
*   **XLA Backend:** Lowers the `FusionAST` (or the equivalent fused subgraph) to HLO. Compilation and kernel fusion happen inside the XLA/PJRT runtime. Do not duplicate this work with LLVM or GPU JIT on the same target.

### 4.4 Cache-Tiling Transformations
For matrix operations (such as `Matmul` and `Conv2D`), the optimizer applies static loop tiling based on target hardware limits:
*   Tensors are restructured into blocks matching the hardware cache sizes (L1/L2 on CPU, Shared Memory/SRAM on GPU).
*   The memory planning stage organizes these tiles to ensure high cache locality, preventing intermediate tiles from being written back to global memory during fused computation.

---

## 5. Execution and Static Memory Planning

All runtime allocations and dynamic tracking are eliminated during the execution loop. The workspace is a single pre-allocated region on the target device, **outside the Go GC heap** (see §5.2).

### 5.1 Static Liveness Analysis & Offset Allocation
The compiler performs a liveness analysis over the topological execution order:
1.  **Lifetime Tracking:** Each port's active lifetime is calculated as the interval $[S, E]$, where $S$ is the step index of the producing node and $E$ is the step index of the final consuming node.
2.  **Interval Coloring Allocator:** The compiler maps intervals to physical byte offsets within the workspace. Intervals that do not overlap share the same physical memory space.
3.  **Hardware Alignment Padding:** Every interval size and base offset is rounded to the target's maximum vector alignment (64 bytes on amd64 AVX-512; 64 bytes minimum on all targets). Sub-byte packed layouts (e.g. int4 dequant pair packing) still occupy whole aligned slots in the workspace; the kernel unpacks inside the aligned region.
4.  **In-Place Validation (CPU / Metal / CUDA only):** If a node represents an elementwise operation (such as `ReLU` or `Exp`) and its input tensor has no downstream consumers beyond this node, the compiler maps the output port to the exact same memory offset as the input port (`dst == src`), eliminating allocation overhead. In-place is permitted only when alignment and dtype/layout constraints match. **Disabled for XLA** — PJRT buffers are not aliased by the host planner; XLA's compiler performs its own in-place decisions during HLO lowering.
5.  **Symbolic Stride Resolution:** For dynamic shape symbols (such as batch size $B$ or sequence length $T$), the compiler generates algebraic offset expressions (e.g., `offset = base_ptr + (b * stride_b) + (t * stride_t)`). These expressions are calculated at launch time, ensuring zero heap allocation during execution.

Relative interval alignment is insufficient unless the workspace base itself is aligned. The top-level allocator must guarantee:

```go
// Required at workspace init — before any interval.Offset is applied
if uintptr(workspaceBase)%workspaceAlign != 0 {
	panic("workspace base not 64-byte aligned")
}
```

*   **CPU:** allocate via `posix_memalign` / `_aligned_malloc` (64-byte minimum).
*   **CUDA / Metal:** use device allocators; assert `uintptr(devicePtr) % 64 == 0` at init.
*   **XLA:** slot indexing assumes 64-byte aligned virtual offsets (`offset >> 6`).

Alignment rule applied during planning:

```go
const workspaceAlign = 64

func alignUp(size int64) int64 {
	return (size + workspaceAlign - 1) &^ (workspaceAlign - 1)
}
// interval.Size = alignUp(rawByteSize); interval.Offset = alignUp(proposedOffset)
```

### 5.2 Asynchronous DAG Scheduler & GC-Safe Memory
Rather than a single-threaded linear loop, the executor walks a dependency DAG and dispatches workloads asynchronously across concurrent hardware queues:
*   **Stream Mapping:** Independent branches in the DAG are mapped to separate hardware streams (e.g., multiple CUDA streams or Metal Command Queues).
*   **Asynchronous Semaphores:** Where parallel paths merge (e.g., in a residual addition), the compiler inserts hardware-level wait events/semaphores directly into the queue pipelines.
*   **Zero Host Wait:** The CPU dispatches execution instructions non-blockingly. The hardware coordinates execution barriers natively. Completion is observed only at explicit execution boundaries.
*   **GC-Safe Workspace:** The execution workspace must **never** be Go heap memory (`make([]byte, …)`, slices backing `[]T`, or any GC-managed object). Allocate exclusively through:
    *   Native **device** memory (`cudaMalloc`, `MTLBuffer`, PJRT `PjRtBuffer` pool), or
    *   OS-native **host** memory outside the Go heap (`posix_memalign`, `mmap`, C `malloc` via `cgo`) when the CPU backend executes in-place on host.
*   **No `runtime.Pinner` in the async loop:** Do not pass Go heap pointers to async GPU/XLA queues and attempt to unpin from driver completion callbacks (`cudaLaunchHostFunc`, Metal `addCompletedHandler`). Those callbacks run on non-Go threads; calling into the Go runtime from them risks deadlocks during stop-the-world GC. If memory is not on the Go heap, pinning is unnecessary.
*   **Pre-resolved pointers:** Before entering the execution loop, the executor materializes `workspaceBase + offset` (CPU/Metal/CUDA) or `resolvedBuffers[offset >> 6]` (XLA) into each `ExecutionNode`. The loop itself performs index/load only — no allocation, no map lookup, no pinning.

---

## 6. IR Specification

```go
type Topology struct {
    Nodes      []Node
    Edges      []Edge
    Workspace  WorkspaceLayout
    InputPorts  map[string]int32 // Maps input names to workspace offsets
    OutputPorts map[string]int32 // Maps output names to workspace offsets
}

type Node struct {
    ID          int32
    Name        string
    Op          ir.Operation     // Maps 1:1 to device.Backend interface methods
    JitKernel   unsafe.Pointer   // Points to compiled JIT binary if fused, otherwise nil
    Inputs      []PortAllocation
    Outputs     []PortAllocation
    WeightToken *types.Token     // Checkpoint weight metadata, if applicable
    StreamID    int32            // Hardware stream index for parallel execution
    SyncBarriers []SyncEvent     // Synchronization events to wait on before dispatch
}

type PortAllocation struct {
    PortID       int32
    BaseOffset   int64           // Static byte offset in the global workspace
    StrideExprs  []StrideFormula // Symbolic stride math for dynamic dimensions
    PortType     PortType
}

type StrideFormula struct {
    Symbol      string           // e.g., "B", "T"
    Multiplier  int64
}
```

---

## 7. Banned Patterns

To maintain peak execution speed and deterministic correctness, the following patterns are prohibited:
*   Any host-device memory copies (`cudaMemcpy`, `Buffer.Contents()`) inside the primary execution loop.
*   Dynamic heap allocations (`malloc`, `new`, `make()`, `cudaMalloc`, `MTLDevice.newBuffer`) during the primary execution loop.
*   Go heap-backed workspace buffers (`make([]byte, workspaceSize)`) or any GC-managed memory passed to async GPU/XLA queues.
*   `runtime.Pinner` or Go runtime calls from native driver completion callbacks.
*   Dynamic map lookups (`map[int64]*PjRtBuffer`) during the execution loop — resolve handles into flat slices at init.
*   Unaligned workspace base pointer (`uintptr(base) % 64 != 0`) or unaligned interval offsets.
*   String-keyed kernel lookups or string parameter parsing at execution time.
*   Implicit or silent type conversions (e.g., automatically casting `Float32` to `Float16` without an explicit adaptor node).
*   Returning Go scalars (`float32`, `int32`) from mathematical execution methods.
*   Single-threaded, blocking topological execution when parallel streams are available on the target hardware.
*   Catch-all backend source files (`device_missing`, `device_remaining`, `*_stub_ops`, `*_extra`) that aggregate unrelated interface methods.
*   Passing Go slice headers (`[]uint8`) or `string` to Metal, CUDA, or XLA backend methods. Host-only ops (`PosPop`) live on `device.HostBackend`.
*   Operation implementations or kernel bridges at the backend package root when a matching subpackage directory exists per §2.3.
*   Root forwarding shims (`backend_activation.go`, …), dtype-prefixed filenames, and arity-based domain names per §2.3 anti-patterns.

## 8. Implementation Plan

```
PHASE 1: Core Interface and Types
  │
  ├── 1.1 Finalize device/interface.go: zero-host-sync dst signatures; split PosPop to device.HostBackend
  ├── 1.2 Define the PortType struct, LayoutSchema, and SemanticKind enums
  ├── 1.3 Write unit tests verifying that all methods in device.Backend accept unsafe.Pointer
  └── 1.4 Reorganize device/metal (and device/cuda) to match device/cpu layout per §2.3
        ├── 1.4a Metal: interface-family tree + quintet per family (§2.3.1) — **done**
        └── 1.4b CUDA: mirror §2.3 family subpackages — **pending**
  │
PHASE 2: Compiler Front-End and Unification
  │
  ├── 2.1 Write YAML manifest parser (converts blocks into raw flat nodes)
  ├── 2.2 Implement Hindley-Milner type unification algorithm over PortTypes
  └── 2.3 Create the Adaptor Synthesis Engine (inserts Transpose, Cast, and Reshape)
  │
PHASE 3: Optimizer and JIT Codegen Engine
  │
  ├── 3.1 Write the Fusion Engine (clusters elementwise nodes into FusionASTs)
  ├── 3.2 Implement target codegen backends:
  │     ├── CPU: LLVM IR → AVX-512 / AVX2 / SSE2 / NEON
  │     ├── GPU: Metal Shading Language & CUDA PTX (runtime compilation)
  │     └── XLA: HLO lowering for FusionASTs and static Backend ops (PJRT compile cache)
  └── 3.3 Implement Cache-Tiling optimizer for Matmul and Convolution
  │
PHASE 4: Static Memory Planner and Scheduler
  │
  ├── 4.1 Write Liveness Analysis engine over DAG (calculates interval arrays)
  ├── 4.2 Implement Interval Coloring Allocator (generates static offsets)
  ├── 4.3 Write Symbolic Stride Solver for dynamic symbols (B, T, D)
  └── 4.4 Implement Out-of-Order DAG Scheduler (maps nodes to streams and sync events)
  │
PHASE 5: Runtime Executor
  │
  ├── 5.1 Implement the flat workspace memory allocator (allocates one block on device startup)
  ├── 5.2 Build the Asynchronous Dispatcher (calls device.Backend via generated pointers)
  └── 5.3 Write host-side synchronization primitives for end-of-step metric collection
  │
PHASE 6: XLA Backend (`device/xla`)
  │
  ├── 6.1 PJRT bridge, pre-resolved buffer slot table (offset >> 6), no runtime map (see §3.1)
  ├── 6.2 `device.Backend` method surface — every interface method dispatches through lowering (Packet 2)
  ├── 6.3 Lowering framework: dtype/shape mapping, builder, compile cache, executable cache (Packet 3)
  ├── 6.4 Full operation coverage: elementwise, activation, matmul, attention, conv, … (Packets 4–N)
  └── 6.5 Parity and benchmarks vs scalar reference on XLA hardware runner (`-tags xla`)
  │
PHASE 7: Verification and Validation
  │
  ├── 7.1 Set up numeric parity verification pipeline (CPU scalar vs AVX-512 vs AVX2 vs SSE2 vs NEON vs Metal vs CUDA vs XLA)
  └── 7.2 Execute performance benchmarks under scale constraints (N ∈ {1, 7, 64, 1024, 8192})
```

### Detailed Phase Tasks

#### Phase 1: Core Interface and Types
*   **Task 1.1:** Finalize `device/interface.go`: all reduction/dot/loss/sampling/similarity methods write to `dst unsafe.Pointer`; split `PosPop` onto `device.HostBackend` (CPU only).
*   **Task 1.2:** Implement validation structures for `PortType`. Write strict checking logic that prevents execution if any dimension mismatch, layout mismatch, or semantic data-type violation is detected.
*   **Task 1.3:** Write unit tests verifying that all methods in `device.Backend` accept `unsafe.Pointer` and adhere to the zero-host-sync principle.
*   **Task 1.4:** Restructure GPU backends to mirror §2.3.
    *   **1.4a (Metal, complete):** `device/metal/` uses the interface-family tree with the quintet per family (§2.3.1). Monolithic MSL and cross-family bridges eliminated. `internal/metallibgen` walks `**/*.metal` under `device/metal/` when building `kernels.metallib`.
    *   **1.4b (CUDA, pending):** Delete `device_missing`, `device_remaining`, and other catch-all files. Move each operation family into its subdirectory with methods on a family type, embedded from root `Backend`, colocated kernels, bridges, and parity tests.
    *   Remove root `backend_*.go` forwarding shims from `device/cpu` as subpackages gain embedded types.

#### Phase 2: Compiler Front-End and Unification
*   **Task 2.1:** Implement the compiler front-end to read manifest YAML configurations. Expand block macros (e.g., active inference blocks or causal heads) into low-level primitives.
*   **Task 2.2:** Build the type unification pass. Every connection between an output port and an input port must satisfy shape, layout, and precision constraints.
*   **Task 2.3:** Write the adaptor synthesis pass. When a mismatch is found (such as a channel-first output feeding a channel-last input), search for a valid transformation sequence and append the corresponding adaptor nodes to the topology.

#### Phase 3: Optimizer and JIT Codegen Engine
*   **Task 3.1:** Write the fusion pass. Iterate through the topology, cluster contiguous elementwise operations, and substitute them with a single `FusedOp` node containing the consolidated math operations in an AST.
*   **Task 3.2:** Write the target codegen layer:
    *   For CPU, write a code generator that translates elementwise ASTs into LLVM IR, compiles them to machine code, and retrieves execution function pointers. Support AVX-512, AVX2, SSE2, and NEON via CPUID-selected LLVM target features.
    *   For GPU (CUDA and Metal), implement string-based kernel builders that generate MSL or CUDA C++ kernels, compile them using driver APIs, and load the executable functions into memory.
    *   For XLA, implement HLO lowering for `FusionAST` clusters and for each static `Backend` method. Cache compiled executables per signature; execute through PJRT without host round-trips between fused ops.
*   **Task 3.3:** Implement cache-tiling logic. For operations like `Matmul` and `Conv2D`, the compiler must divide the calculation into micro-tiles based on the detected hardware cache sizes, structuring memory access patterns to maximize local memory reuse.

#### Phase 4: Static Memory Planner and Scheduler
*   **Task 4.1:** Build the liveness analysis engine. Walk the execution graph to find the exact node step index where each port is initialized and where its final read occurs.
*   **Task 4.2:** Implement the register-allocation-style coloring algorithm to assign intermediate ports to non-overlapping byte-offsets inside a unified memory block. Apply 64-byte alignment padding to every interval size and base offset (§5.1). Disable in-place offset reuse when compile target is XLA.
*   **Task 4.3:** Develop the algebraic stride compiler. Convert dynamic shape variables into simple linear offset math calculations that run on the CPU just before submitting the GPU commands.
*   **Task 4.4:** Write the stream partitioner. Group independent subgraphs into concurrent streams, inserting GPU execution semaphores at the merge points.

#### Phase 5: Runtime Executor
*   **Task 5.1:** Implement the runtime workspace allocator with **64-byte aligned base** (`posix_memalign` on CPU; validate device pointer alignment on CUDA/Metal). Never `make([]byte)`. No `runtime.Pinner` — workspace lives entirely outside the Go heap (§5.2).
*   **Task 5.2:** Build the non-blocking execution loop. Pre-materialize all port pointers / `PjRtBuffer` slot indices before the loop. No map lookups or pinning during dispatch.
*   **Task 5.3:** Implement the host query engine. Create host-side wait mechanisms to copy output metrics from the device back to host memory only when requested.

#### Phase 6: XLA Backend
*   **Task 6.1:** Implement the PJRT bridge, pre-resolved `[]*PjRtBuffer` slot table (`offset >> 6`), and virtual-offset layout. No runtime map. In-place host planner disabled for XLA (§3.1, §5.1).
*   **Task 6.2:** Implement every `device.Backend` method on `device/xla`. Each method resolves resident buffers, lowers to HLO, compiles, and executes. Add `var _ device.Backend = (*xla.Backend)(nil)`.
*   **Task 6.3:** Build the lowering framework (builder, dtype/shape mapping, program key, compile cache, executable cache) per `XLA_GAPS.md` Packet 3.
*   **Task 6.4:** Implement all operation families with parity tests against the scalar reference. Do not compare XLA output to XLA output. Do not widen tolerances.
*   **Task 6.5:** Run verification on an amd64 machine with NVIDIA hardware and XLA-GPU support. Development machines without XLA defer runtime verification but must still land complete source, tests, and benchmarks.

#### Phase 7: Verification and Validation
*   **Task 7.1:** Build a automated test harness that executes every operation across all target backends (scalar CPU, AVX-512, AVX2, SSE2, NEON, Metal, CUDA, XLA) using random input values, verifying that the output matches the scalar reference within strict precision tolerances.
*   **Task 7.2:** Profile execution time and memory footprint under varying dimensions to confirm zero dynamic allocations and complete utilization of hardware streams.

What was presented was the high-level structural contract and the execution specification. To provide a complete, non-delegated implementation guide, we must define the concrete compiler code, algorithm implementations, and low-level execution machinery.

The following sections provide the low-level code patterns, mathematical frameworks, and algorithms required to build the compiler, the JIT engine, the static memory planner, and the asynchronous DAG dispatcher.

---

# Detailed Implementation Blueprints

## 1. The Fusion AST Representation

To allow the optimizer to group elementwise operations and compile them into a single loop, we require a concrete Abstract Syntax Tree (AST) structure. This represents the algebraic operations before they are translated into target assembly or GPU kernels.

```go
package compiler

import "github.com/theapemachine/manifesto/dtype"

type NodeType int

const (
	NodeInput NodeType = iota
	NodeConstant
	NodeAdd
	NodeSub
	NodeMul
	NodeDiv
	NodeReLU
	NodeExp
	NodeSigmoid
)

// FusionAST Node represents an algebraic operation in a fused loop
type ASTNode struct {
	Type       NodeType
	Value      float64       // Used if Type == NodeConstant
	InputIndex int           // Maps to the input pointer array if Type == NodeInput
	Children   []*ASTNode
	DType      dtype.DType
}

// FusionAST represents a consolidated elementwise mathematical expression
type FusionAST struct {
	Root       *ASTNode
	InputPorts []int32 // Global workspace offsets of inputs
	OutputPort int32   // Global workspace offset of the destination
	CountExpr  string  // Dynamic symbol expression representing element count
}
```

---

## 2. JIT Kernel Generation (CUDA PTX, Metal Shading Language, and XLA HLO)

The following generator parses the `FusionAST` and dynamically constructs valid GPU kernel source strings. This code is passed directly to the runtime compiler API (`nvrtc` for CUDA or `MTLDevice` for Metal). On XLA targets, the same `FusionAST` lowers to HLO instead — see §3.1 and `XLA_GAPS.md`.

```go
package codegen

import (
	"fmt"
	"strings"
	"compiler"
)

type ShaderGenerator struct{}

func (g *ShaderGenerator) GenerateMetal(ast *compiler.FusionAST) (string, error) {
	var builder strings.Builder

	builder.WriteString("#include <metal_stdlib>\n")
	builder.WriteString("using namespace metal;\n\n")
	builder.WriteString("kernel void fused_kernel(\n")

	// Declare inputs dynamically
	for i := range ast.InputPorts {
		builder.WriteString(fmt.Sprintf("    device const float* in%d [[buffer(%d)]],\n", i, i))
	}
	// Declare output
	builder.WriteString(fmt.Sprintf("    device float* out [[buffer(%d)]],\n", len(ast.InputPorts)))
	builder.WriteString("    uint id [[thread_position_in_grid]]\n")
	builder.WriteString(") {\n")
	
	// Add bound checking
	builder.WriteString("    if (id >= 1024) return; // TODO: Bind to dynamic CountExpr\n")

	// Generate the mathematical expression recursively
	expr := g.generateExpression(ast.Root)
	builder.WriteString(fmt.Sprintf("    out[id] = %s;\n", expr))
	builder.WriteString("}\n")

	return builder.String(), nil
}

func (g *ShaderGenerator) generateExpression(node *compiler.ASTNode) string {
	switch node.Type {
	case compiler.NodeInput:
		return fmt.Sprintf("in%d[id]", node.InputIndex)
	case compiler.NodeConstant:
		return fmt.Sprintf("%f", node.Value)
	case compiler.NodeAdd:
		return fmt.Sprintf("(%s + %s)", g.generateExpression(node.Children[0]), g.generateExpression(node.Children[1]))
	case compiler.NodeSub:
		return fmt.Sprintf("(%s - %s)", g.generateExpression(node.Children[0]), g.generateExpression(node.Children[1]))
	case compiler.NodeMul:
		return fmt.Sprintf("(%s * %s)", g.generateExpression(node.Children[0]), g.generateExpression(node.Children[1]))
	case compiler.NodeDiv:
		return fmt.Sprintf("(%s / %s)", g.generateExpression(node.Children[0]), g.generateExpression(node.Children[1]))
	case compiler.NodeReLU:
		return fmt.Sprintf("max(0.0, %s)", g.generateExpression(node.Children[0]))
	case compiler.NodeExp:
		return fmt.Sprintf("exp(%s)", g.generateExpression(node.Children[0]))
	case compiler.NodeSigmoid:
		return fmt.Sprintf("(1.0 / (1.0 + exp(-%s)))", g.generateExpression(node.Children[0]))
	default:
		return "0.0"
	}
}
```

---

## 3. Static Memory Allocation: Interval Coloring Algorithm

This algorithm performs liveness analysis over the topological execution steps and maps every intermediate buffer to a non-overlapping offset within a single, global workspace block.

```go
package scheduler

import (
	"sort"
)

type Interval struct {
	PortID int32
	Start  int // Step index of production
	End    int // Step index of final consumption
	Size   int64 // Byte size of the allocation
	Offset int64 // Calculated during allocation
}

type MemoryPlanner struct{}

const workspaceAlign int64 = 64

func alignUp(size int64) int64 {
	return (size + workspaceAlign - 1) &^ (workspaceAlign - 1)
}

func (memoryPlanner *MemoryPlanner) AllocateOffsets(intervals []*Interval) int64 {
	// Sort intervals by start step to process in topological order
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i].Start < intervals[j].Start
	})

	type ActiveAllocation struct {
		Offset int64
		End    int
		Size   int64
	}

	var active []ActiveAllocation
	var maxWorkspaceSize int64 = 0

	for _, interval := range intervals {
		// Pad interval size to 64-byte alignment for SIMD/GPU load requirements
		interval.Size = alignUp(interval.Size)

		// Remove expired allocations whose End step is less than the current interval's Start step
		var remaining []ActiveAllocation
		for _, act := range active {
			if act.End >= interval.Start {
				remaining = append(remaining, act)
			}
		}
		active = remaining

		// Find the lowest available offset that does not overlap with any active allocation
		var proposedOffset int64 = 0
		for {
			collision := false
			for _, act := range active {
				// Check if the proposed memory region overlaps with an active allocation
				if proposedOffset < act.Offset+act.Size && proposedOffset+interval.Size > act.Offset {
					proposedOffset = act.Offset + act.Size // Push offset past the colliding allocation
					collision = true
					break
				}
			}
			if !collision {
				break
			}
		}

		// Align proposed offset to workspace boundary
		proposedOffset = alignUp(proposedOffset)

		// Assign offset to interval
		interval.Offset = proposedOffset
		active = append(active, ActiveAllocation{
			Offset: interval.Offset,
			End:    interval.End,
			Size:   interval.Size,
		})

		// Track maximum high-water mark of the workspace allocation
		if interval.Offset+interval.Size > maxWorkspaceSize {
			maxWorkspaceSize = interval.Offset + interval.Size
		}
	}

	return maxWorkspaceSize
}
```

---

## 4. Symbolic Shape & Offset Solver

When handling dynamic dimension symbols (such as batch size $B$ or sequence length $T$), the compiler generates algebraic stride evaluation trees to calculate target memory locations at launch time without dynamic heap allocation.

```go
package scheduler

type SymbolMap map[string]int64

type Dimension struct {
	Symbol     string // e.g., "B", "T", "D"
	StaticVal  int64  // Used if Symbol is empty
}

type LayoutSchema struct {
	Dims []Dimension
}

// ResolveOffset calculates the byte offset for a specific tensor coordinate
func ResolveOffset(layout LayoutSchema, coords []int64, symbols SymbolMap, elementSize int64) int64 {
	var offset int64 = 0
	var currentStride int64 = 1

	// Iterate backward from the innermost dimension (row-major default)
	for i := len(layout.Dims) - 1; i >= 0; i-- {
		coord := coords[i]
		offset += coord * currentStride

		// Compute dimension size to calculate the stride of the next outer dimension
		dimSize := layout.Dims[i].StaticVal
		if layout.Dims[i].Symbol != "" {
			dimSize = symbols[layout.Dims[i].Symbol]
		}
		currentStride *= dimSize
	}

	return offset * elementSize
}
```

---

## 5. The Asynchronous DAG Execution Loop

The runtime executor processes fully resolved nodes out-of-order, bypassing CPU overhead by scheduling computations directly onto target hardware command queues (streams) with hardware semaphores.

```go
package runtime

import (
	"unsafe"
	"device"
)

type ExecutionNode struct {
	Op           device.Backend
	Inputs       []unsafe.Pointer // Pre-calculated offset pointers
	Outputs      []unsafe.Pointer // Pre-calculated offset pointers
	StreamID     int              // Maps to native execution queue
	WaitEvents   []uintptr        // Hardware semaphores to wait on
	SignalEvents []uintptr        // Hardware semaphores to trigger on completion
}

type Executor struct {
	workspaceBase    unsafe.Pointer   // 64-byte aligned; outside Go heap
	workspaceSlots   []*xlaBuffer     // XLA only: pre-resolved, indexed by offset >> 6
	streams          []uintptr        // Target queue pointers (e.g., MTLCommandQueue or CUstream)
}

func NewExecutor(base unsafe.Pointer, streams []uintptr) *Executor {
	if uintptr(base)%64 != 0 {
		panic("workspace base not 64-byte aligned")
	}

	return &Executor{
		workspaceBase: base,
		streams:       streams,
	}
}

// Execute submits operations to the hardware asynchronously
func (e *Executor) Execute(nodes []ExecutionNode) {
	for _, node := range nodes {
		// 1. Enqueue wait dependencies directly into the target stream
		for _, event := range node.WaitEvents {
			e.enqueueStreamWait(node.StreamID, event)
		}

		// 2. Dispatch computation (completely non-blocking)
		e.dispatchKernel(node)

		// 3. Enqueue completion signals directly into the target stream
		for _, event := range node.SignalEvents {
			e.enqueueStreamSignal(node.StreamID, event)
		}
	}
}

func (e *Executor) enqueueStreamWait(streamID int, event uintptr) {
	// Native binding: cudaStreamWaitEvent(e.streams[streamID], event, 0)
	// Or Metal equivalent: [commandBuffer encodeWaitForEvent:event value:1]
}

func (e *Executor) enqueueStreamSignal(streamID int, event uintptr) {
	// Native binding: cudaEventRecord(event, e.streams[streamID])
	// Or Metal equivalent: [commandBuffer encodeSignalEvent:event value:1]
}

func (executor *Executor) dispatchKernel(node ExecutionNode) {
	// Direct execution call to device.Backend or JIT-compiled function.
	// Inputs/Outputs are pre-resolved before the loop:
	//   CPU/Metal/CUDA: workspaceBase + port.BaseOffset
	//   XLA: workspaceSlots[port.BaseOffset >> 6]
}
```

---

## 6. Domain-Specific Mathematical Port Mapping

This section defines the precise semantic `PortType` models used by the compiler to unify mathematical interfaces for specific advanced models.

### 6.1 Active Inference Block
*   **Goal:** Calculate Expected Free Energy (EFE) and update policy beliefs ($q$).
*   **Mathematical Formulations:**
    $$\text{EFE}_a = \sum_o q(o \mid a) \left[ \ln q(o \mid a) - \ln P(o) \right] + \sum_s q(s \mid a) H(A \mid s)$$
*   **PortType Signatures:**

```go
var ActiveInferenceEFESignature = struct {
	PredictedObs    PortType // Shape: [B, A, O], Kind: BeliefState (q(o|a))
	PreferredObs    PortType // Shape: [B, O],    Kind: PreferenceDistribution (P(o))
	PredictedState  PortType // Shape: [B, A, S], Kind: BeliefState (q(s|a))
	EntropyA        PortType // Shape: [S],       Kind: SystemEntropy (H(A|s))
	OutputEFE       PortType // Shape: [B, A],    Kind: ExpectedFree Energy (EFE)
}{
	PredictedObs:    PortType{DType: dtype.Float32, Kind: "BeliefState"},
	PreferredObs:    PortType{DType: dtype.Float32, Kind: "PreferenceDistribution"},
	PredictedState:  PortType{DType: dtype.Float32, Kind: "BeliefState"},
	EntropyA:        PortType{DType: dtype.Float32, Kind: "SystemEntropy"},
	OutputEFE:       PortType{DType: dtype.Float32, Kind: "ExpectedFreeEnergy"},
}
```

### 6.2 Hawkes Process Block
*   **Goal:** Compute the intensity function $\lambda(t)$ over query timestamps.
*   **Mathematical Formulation:**
    $$\lambda(t) = \mu + \alpha \sum_{t_i < t} e^{-\beta (t - t_i)}$$
*   **PortType Signatures:**

```go
var HawkesIntensitySignature = struct {
	EventTimes PortType // Shape: [B, N], Kind: EventTimestamps
	QueryTimes PortType // Shape: [B, M], Kind: QueryTimestamps
	Output     PortType // Shape: [B, M], Kind: IntensityValues
}{
	EventTimes: PortType{DType: dtype.Float32, Kind: "EventTimestamps"},
	QueryTimes: PortType{DType: dtype.Float32, Kind: "QueryTimestamps"},
	Output:     PortType{DType: dtype.Float32, Kind: "IntensityValues"},
}
```

---

## 7. Concrete Testing & Validation Verification

To ensure numerical correctness and strict behavioral parity across all hardware targets (scalar reference CPU vs. vectorized assembly vs. GPU JIT vs. XLA HLO execution), compilation output is run through a deterministic differential verification loop.

```go
package verification

import (
	"math"
	"testing"
	"unsafe"
	"device"
	"github.com/theapemachine/manifesto/dtype"
)

// VerifyParity checks that CPU reference outputs match GPU execution outputs precisely
func VerifyParity(t *testing.T, backendCPU, backendGPU device.Backend, size int, opName string) {
	// 1. Allocate host reference buffers
	inputData := make([]float32, size)
	outputRef := make([]float32, size)
	outputTest := make([]float32, size)

	// Initialize inputs with deterministic values
	for i := 0; i < size; i++ {
		inputData[i] = float32(i) * 0.1
	}

	inPtr := unsafe.Pointer(&inputData[0])
	outRefPtr := unsafe.Pointer(&outputRef[0])
	outTestPtr := unsafe.Pointer(&outputTest[0])

	// 2. Execute on reference CPU backend
	if opName == "ReLU" {
		backendCPU.ReLU(outRefPtr, inPtr, size, dtype.Float32)
		backendGPU.ReLU(outTestPtr, inPtr, size, dtype.Float32)
	}

	// 3. Perform pairwise delta validation
	const epsilon = 1e-5
	for i := 0; i < size; i++ {
		delta := math.Abs(float64(outputRef[i] - outputTest[i]))
		if delta > epsilon {
			t.Fatalf("Numerical parity violation at index %d in %s. CPU: %f, GPU: %f", 
				i, opName, outputRef[i], outputTest[i])
		}
	}
}
```

No, you should **not** drop AVX2 and SSE2 support. 

The reference to AVX-512 in the JIT compilation phase was meant to illustrate the highest vector path on x86, but restricting your compiler to AVX-512 alone would severely limit the deployment surface of the platform.

Here is why you must retain AVX2 and SSE2, along with how to implement them deterministically within your JIT compilation architecture.

---

### 1. Hardware Distribution & Compatibility

While AVX-512 provides high compute density, it is not universally supported:
*   **The Hybrid-Core Mismatch:** Intel disabled AVX-512 on several generations of consumer hybrid CPUs (such as Alder Lake, Raptor Lake, and Arrow Lake) because their E-cores did not support 512-bit vector registers. While newer architectures (like Intel's Nova Lake featuring AVX10 and AMD's Zen 4/Zen 5) support 512-bit vectors, millions of consumer machines in active service are capped at AVX2.
*   **The x86-64 Baseline:** SSE2 is the fundamental baseline instruction set for all 64-bit x86 processors. Keeping SSE2 ensures that your stack will execute on any x86-64 CPU, serving as your final deterministic fallback when no AVX capability is present.

If you drop AVX2 and SSE2, your execution stack will fail with invalid instruction faults on a significant portion of consumer hardware.

---

### 2. How the JIT Handles Multiple ISAs Deterministically

Using a JIT compiler (such as LLVM) does not mean you have to write three completely separate code generators. Instead, you design your code generator to emit vector-width-agnostic intermediate representation (IR), and let LLVM lower it to the specific hardware targets based on a startup query.

#### The Implementation Strategy:
1.  **CPUID Startup Detection:** When the runtime starts, it queries the host CPUID once to determine the highest supported instruction set (AVX-512 / AVX10, AVX2, SSE2).
2.  **Target Feature Flagging:** When initializing the LLVM Execution Engine, you pass the corresponding CPU target features:
    *   *AVX-512 Target:* Set LLVM features to `+avx512f` (and any desired sub-extensions).
    *   *AVX2 Target:* Set LLVM features to `+avx2`.
    *   *SSE2 Target:* Set LLVM features to `+sse2`.
3.  **Vector Width Selection:** The compiler maps its AST operations to vector widths supported by the host hardware:
    *   AVX-512 uses `v64f32` (16 elements of 32-bit floats).
    *   AVX2 uses `v32f32` (8 elements of 32-bit floats).
    *   SSE2 uses `v16f32` (4 elements of 32-bit floats).

---

### 3. Revised Compiler Codegen Specification (Phase 3 Update)

To reflect this deterministic fallback path, the codegen section of Phase 3 is updated as follows:

```
PHASE 3: Optimizer and JIT Codegen Engine
  │
  ├── 3.1 Write the Fusion Engine (clusters elementwise nodes into FusionASTs)
  ├── 3.2 Implement JIT compiler backends:
  │     ├── CPU (LLVM): Query host CPUID at startup to target the highest available path:
  │     │     ├── Path A: AVX-512/AVX10 (512-bit vectors / 16 float elements)
  │     │     ├── Path B: AVX2 (256-bit vectors / 8 float elements)
  │     │     └── Path C: SSE2 (128-bit vectors / 4 float elements)
  │     ├── GPU: Metal Shading Language & CUDA PTX generators with runtime compilation
  │     └── XLA: HLO lowering + PJRT compile cache (no LLVM/MSL/CUDA JIT on the same target)
  └── 3.3 Implement Cache-Tiling optimizer for Matmul and Convolution
```

By keeping all three paths, you guarantee absolute correctness and portability across the x86 ecosystem, while automatically unlocking maximum execution speed on modern hardware.