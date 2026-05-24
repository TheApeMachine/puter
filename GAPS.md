# Implementation Gaps vs. ARCHITECTURE.md

Audit date: 2026-05-24. Scope: `puter`, `manifesto`, `hf`, `caramba` against `puter/ARCHITECTURE.md`.

This document is a punch list, not a redesign. Section numbers refer to `ARCHITECTURE.md` unless noted otherwise.

---

## 0. Executive summary

The four repos together implement roughly half of the architecture by surface area, but the half that's missing is the half that turns kernels into a system: the manifesto compiler/optimizer/codegen/scheduler pipeline (Phases 2–4 of §8) is essentially absent. `puter`'s kernel surface is broad but its top-level interface contract still violates the zero-host-sync rule. `hf` is in the best shape relative to its scope. `caramba` is a thin orchestrator that depends on the missing pieces.

What works end-to-end today: load a HuggingFace checkpoint, resolve a recipe, build a minimal IR, and dispatch ops sequentially via the `tensor.Backend` interface that delegates to puter's CPU/Metal kernels. What doesn't exist: PortType unification, adaptor synthesis, fusion AST, JIT codegen for any target, liveness analysis, static interval-coloring allocator, symbolic stride solver, stream-partitioned DAG executor, parity verification harness.

The single most damaging gap is **§2.2 zero-host-sync** — the `device.Backend` interface itself still returns Go scalars from 16 reduction/dot/loss/sampling methods. Until that's fixed, the rest of the architecture (async streams, non-blocking dispatch, XLA buffer aliasing) cannot land.

---

## 1. Repository roles

| Repo | Actual role | Spec section it owns |
|---|---|---|
| **puter** | `device.Backend` implementations: CPU (scalar + AVX-512/AVX2/SSE2/NEON), Metal, CUDA, XLA. Owns the family tree under `device/<backend>/`. | §2 (`device/interface.go`), §2.3 (backend layout), §3.1 (XLA target). |
| **manifesto** | HuggingFace checkpoint compiler today. YAML manifest parser, recipe expander, checkpoint binder, minimal sequential executor. Should own the full compiler/optimizer/codegen/scheduler pipeline. | §1 (manifesto stack), §4 (PortType + composition), §5 (memory planning), §6 (IR), §8 Phases 2–5. |
| **hf** | HuggingFace Hub client, safetensors parser, tokenizer, model config → manifest generator. Implements manifesto's `Hub` and `Host` interfaces. | §1 (checkpoint tokens / safetensors), §6 (`WeightToken *types.Token`). |
| **caramba** | Application/orchestrator. Wires puter + manifesto + hf together, exposes HTTP API, CLI (`program`, `chat`, `research`, `serve`), tuner, research project scaffolding. | Consumer of §1 stack; partial owner of fusion catalog and Phase 7 verification. |

---

## 2. puter — kernel backends

### 2.1 Backend completeness matrix

Families listed in §2.3 of the spec. Cell = source present (not necessarily correct or parity-tested).

| Family | CPU scalar | AVX-512 | AVX2 | SSE2 | NEON | Metal | CUDA | XLA |
|---|---|---|---|---|---|---|---|---|
| activation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| elementwise | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| reduction | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| dot | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| matmul | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| pool | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| convolution | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| dropout | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| losses | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| sampling | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| embedding | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| normalization | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| layernorm | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| rope | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| hawkes | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| physics | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| causal | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| masking | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (under attention/) | ✓ | partial |
| attention | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| vsa | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| active_inference | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| predictive_coding | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| dequant | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| quant | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | partial |
| pospop *(host only)* | ✓ | ✓ | ✓ | ✓ | ✓ | n/a | n/a | n/a |

XLA is marked "partial" across the board because the build is gated behind `//go:build xla`, `bridge_stub.go` returns `ErrNeedsPlatformSetup`, and the spec's pre-resolved `[]*PjRtBuffer` slot table indexed by `offset >> 6` (§3.1 lines 1223–1237) is not in place. Coverage cannot be verified without an XLA-enabled build environment.

### 2.2 Families present in puter but not in the spec

These directories exist under `device/cpu/` (and partially elsewhere) and are not enumerated in §2.3. They need to either be added to the architecture or folded into existing families:

- `optimizer/` — likely SGD/Adam/etc. updates. Not in spec.
- `math/` — appears to overlap with `elementwise/` and `activation/` (`convert.go`, `activation.go`, `softmax.go`, `constant.go`). Possible duplication.
- `shape/` — reshape/transpose adaptors. Should arguably move to manifesto adaptor synthesis (§4.2), not be a device family.
- `checkpoint/` — checkpoint-related kernels. Spec puts checkpoints in `hf`/manifesto, not on `device.Backend`.
- `interpretability/`, `model_editing/` — research families not declared in §2.3.
- `neon/` — appears to be a shared NEON helpers directory rather than a family. Should be `internal/`.

**Action:** decide for each whether to (a) add to `device/interface.go` and §2.3, (b) move to manifesto/compiler, or (c) delete.

### 2.3 `device/interface.go` conformance — **P0a COMPLETE**

The spec (§2.2) is unambiguous: "All operations that reduce tensors to single values must write their output to a destination pointer (`dst unsafe.Pointer`) on the device."

**P0a contract migration status: all 16 methods migrated, `make check §1 = 0`.**

| Family | Methods | Contract migrated? | Kernel writes on device? |
|---|---|---|---|
| Reduction | Sum, Prod, ReduceMin, ReduceMax, L1Norm | ✓ done | ✗ P0b — Metal/CUDA/XLA still do internal device→host scalar transfer |
| Dot | Dot | ✓ done | ✗ P0b |
| Losses | MSE, MAE, Huber, BinaryCrossEntropy, KLDivergence, CrossEntropy | ✓ done | ✗ P0b |
| Sampling | GreedySample, TopKSample, TopPSample | ✓ done | ✗ P0b |
| VSA | Similarity | ✓ done | ✗ P0b |

**P0a — Interface contract.** Public method signatures take `dst unsafe.Pointer` and return nothing. CPU implementation writes `*(*float32)(dst) = computedValue`. Metal/CUDA/XLA implementations temporarily store the host-returned scalar from `host.ReductionScalar(...)` into `*(*float32)(dst)` — same observable behavior as the old code, but the *interface* is now compatible with async dispatch.

**P0b — Kernel-side device writes.** The Metal/CUDA/XLA reduction kernels already compute their result on device into a temporary buffer, then `host.ReductionScalar` reads it back to host and returns a Go scalar. The end-state per §2.2 is: the kernel writes directly into the caller's `dst` workspace slot, no read-back. This requires the static memory planner (P1, §3.1) to resolve workspace pointers to backend-specific buffer handles (`MetalBufferRef`, `cuMem`, `PjRtBuffer`). Sequenced after the planner lands.

`scripts/check_banned.sh §1` enforces P0a. P0b is enforced by code review against AGENTS.md §11.2 and by the absence of device→host scalar transfers in the async execution path once the executor is in place.

**Caller updates needed:** the public `Reduction.Sum/Prod/...` callers are mostly the `*Native` host-side convenience helpers in `device/cpu/reduction/select_{amd64,arm64,generic}.go` plus a small number of parity tests (`reduction_reduced_prod_parity_test.go`, `xla/reduction/reduction_parity_test.go`, `xla/dot/dot_parity_test.go`). All have been migrated for the Reduction family. **No manifesto or caramba callers** — Reduction is well-isolated.

### 2.4 Anti-pattern violations (§2.3, §7)

**Dtype-prefixed filenames** (§2.3 line 202 forbids them; only int4/int8 in `dequant`/`quant` get a pass as they name a *quantization scheme*, not a dtype):

Spotted across `device/cpu/`: `f32_*`, `bf16_*`, `fp16_*`, `f64_*`, `int8_*` filenames in `activation/`, `dot/`, `dropout/`, `causal/`, `matmul/`, `optimizer/`, `layernorm/`, `rope/`, `vsa/`, `pool/`, `losses/`, `embedding/`, `normalization/`, `sampling/`, `shape/`, `masking/`, `math/`, `hawkes/`, `physics/`, `active_inference/`, `predictive_coding/`, `convolution/`, `reduction/`, `elementwise/`, `attention/`, `checkpoint/`, `interpretability/`, `model_editing/`, `neon/`. Likely 80+ files.

The §3 dispatch pattern is: one public method, switch on `format dtype.DType` inside `dispatch.go`, route to a single domain kernel that handles all dtypes. Current layout encodes the dtype in the path instead.

**Action:** rename or merge per domain. This is mechanical but touches a lot of files. Do it incrementally per family alongside §2.3 conformance.

**Other anti-patterns** — spot checks did **not** find:
- Catch-all `device_missing*`, `device_remaining*`, `*_stub_ops*`, `*_extra*` files.
- Root forwarding shims `backend_<family>.go`.
- Monolithic MSL files spanning multiple semantic domains.
- Orphan Metal kernels without a matching quintet sibling.

The Metal quintet (§2.3.1) appears compliant across all families. CUDA `.cu` per-domain + `bridge.cu` per family is in place.

### 2.5 XLA — `XLA_GAPS.md` missing

§2.1 references `XLA_GAPS.md` as "implementation contracts and gap tracking for XLA". The file does not exist in the repo. Either:
- The spec is aspirational and `XLA_GAPS.md` was never created, or
- It was removed.

§8 Phase 6 lists 5 sub-tasks (PJRT bridge with pre-resolved slot table, full `device.Backend` method surface, lowering framework, full op coverage with parity, hardware verification). None of these can be tracked without the file.

**Action:** create `XLA_GAPS.md` with the per-packet structure §3.1/§8.6 describes. Even an empty skeleton + a current-status checklist makes the gap visible.

### 2.6 XLA — pre-resolved buffer slot table

§3.1 lines 1223–1237 require a flat pre-resolved `[]*PjRtBuffer` slice indexed by `offset >> 6`, populated once at workspace init, indexed directly during the execution loop with no map lookups or locks. Current XLA path likely uses dynamic resolution. Once the executor exists, this must land at the same time.

---

## 3. manifesto — compiler/optimizer/codegen/scheduler

Manifesto today is a **HuggingFace checkpoint compiler with a sequential interpreter**, not the multi-stage AOT compiler the architecture describes. There is no `codegen/`, `optimizer/`, or `scheduler/` top-level directory.

### 3.1 Pipeline stage coverage

| Spec stage | Phase | Status | Where (if any) | Notes |
|---|---|---|---|---|
| YAML parser + block macro expansion | 2.1 | ✓ partial | `parse/`, `expand/`, `lower/`, `ast/`, `catalog/`, `registry/` | Parses, expands `extends`/`repeat`, lowers shapes. No semantic-level macro expansion (active inference blocks, causal heads) yet. |
| `PortType` (DType, ShapeSchema, LayoutSchema, SemanticKind, Constraints) | 2.2 | ✗ | — | Type doesn't exist. `Port` in `ir/` is just `Tensor *tensor.Tensor`. |
| Hindley-Milner port unification | 2.2 | ✗ | — | Not implemented. |
| Adaptor synthesis (Transpose/Cast/Reshape insertion) | 2.3 | ✗ | — | Not implemented. |
| `FusionAST` + elementwise clustering | 3.1 | ✗ | — | No fusion type, no fusion pass. |
| CPU JIT codegen (LLVM → AVX-512/AVX2/SSE2/NEON) | 3.2a | ✗ | — | No LLVM bindings, no IR builder. |
| GPU JIT codegen (MSL / PTX) | 3.2b | ✗ | — | Kernel source generators not present. |
| XLA HLO lowering + PJRT compile cache | 3.2c | ✗ | — | Not present in manifesto; stub bridge in `puter/device/xla`. |
| Cache-tiling for Matmul/Conv | 3.3 | ✗ | — | Not implemented. |
| Liveness analysis (`[Start, End]` per port) | 4.1 | ✗ | — | Not implemented. |
| Interval-coloring allocator + 64-byte alignment | 4.2 | ✗ | — | Not implemented. |
| Symbolic stride solver (`StrideFormula`, `ResolveOffset`) | 4.3 | ✗ | — | Not implemented. |
| DAG scheduler (stream partition + semaphores) | 4.4 | ✗ partial | `runtime/plan.go` `ExecutionPlan.Layers [][]string` | Topological layering exists. No stream mapping, no `StreamID`/`SyncBarriers` on nodes, no hardware-queue dispatch. |
| Flat 64-byte aligned workspace | 5.1 | ✗ | `tensor/arena.go`, `tensor/slab.go` | Native allocators exist (mmap on Linux/Darwin), but not driven by a static interval allocator. |
| Async non-blocking executor | 5.2 | ✗ | `runtime/executor.go` | Sequential host-side dispatch; map-based input lookups (compile-time, but no streams). |
| Parity harness (CPU scalar vs SIMD vs GPU vs XLA) | 7.1 | ✗ | — | Not present in manifesto. |

### 3.2 IR conformance vs §6

Spec IR (§6):

```
Topology { Nodes, Edges, Workspace, InputPorts, OutputPorts }
Node     { ID, Name, Op, JitKernel, Inputs, Outputs, WeightToken, StreamID, SyncBarriers }
PortAllocation { PortID, BaseOffset, StrideExprs, PortType }
StrideFormula  { Symbol, Multiplier }
```

Current (`manifesto/ir/`):

```
Topology { Kind, Name, Description, Created, Updated, Nodes, Edges }
Node     { Kind, Name, Description, Created, Updated, Operation, Weight, Inputs, Outputs }
Port     { Tensor *tensor.Tensor }
Edge     { From string, To string }
```

Missing: `Workspace`, `InputPorts`, `OutputPorts` on Topology; `ID`, `JitKernel`, `StreamID`, `SyncBarriers` on Node; entire `PortAllocation` / `StrideFormula` / `PortType` family. Without these types the rest of Phase 4–5 has nothing to read or write.

### 3.3 Banned-pattern audit

Within manifesto's current scope (host-side compilation + sequential dispatch) the §7 violations are mostly N/A — there is no async loop to leak Go pointers into. The two notable items:

- `runtime/executor.go` uses `map[string]any` for input/state lookups. Compile-time, so not a §7 violation today, but when the executor is rewritten to dispatch async, all such maps must be pre-resolved into flat slices (§5.2 "Pre-resolved pointers").
- `tensor/` builds a `Backend` abstraction with allocation arenas. Acceptable today; must remain off-Go-heap when feeding device queues.

### 3.4 Biggest gaps in manifesto

1. **Entire PortType + unification layer absent.** This is the foundational type system. Until it exists, no fusion, no adaptor synthesis, no static planning.
2. **Zero JIT codegen.** Spec describes LLVM, MSL, PTX, and HLO emit paths. None exist. Today, ops execute by `Backend` method call dispatched through the `tensor.Backend` interface.
3. **Static memory planner absent.** Liveness, interval coloring, symbolic strides, stream partition — none implemented. The IR types to hold their output don't exist either.
4. **IR types incomplete.** Node has no `StreamID`/`SyncBarriers`/`JitKernel`. Port has no offset, stride, or type metadata. Topology has no workspace.
5. **No parity harness in manifesto.** Each backend in puter has its own per-family `*_parity_test.go`, but there is no cross-backend differential test driven from manifesto's compiled topology.

---

## 4. hf — HuggingFace I/O

### 4.1 Role and spec alignment

`hf` is the cleanest of the four relative to its declared scope. It is the safetensors / Hub / tokenizer / config / runtime-Host layer manifesto depends on. The architecture references its outputs primarily in §1 ("checkpoint tokens attach weight metadata after the graph is compiled") and §6 (`WeightToken *types.Token` on `Node`).

| Spec touchpoint | Status |
|---|---|
| Safetensors → `types.Token` with `Span{Offset, Length}` | ✓ |
| Hub client with revision pinning, ETag/SHA caching, parallel snapshot | ✓ |
| Tokenizer load + encode/decode + chat templates | ✓ |
| Model config → manifest YAML | ✓ Llama only |
| `manifesto/runtime.Host` (ReadLine, Encode, EmitToken, WriteImage) | ✓ |
| `manifesto/resolve.Hub` adapter | ✓ |
| Dtype coverage (F32, F16, BF16, I8, I4) | ✓ |
| No Go-heap leakage into device workspace | ✓ |

### 4.2 Gaps

- **Config generator is Llama-only.** `config/generator.go` has one template. Anything else (BERT, GPT-2, diffusion, ViT) needs an explicit template or fails.
- **`OperationLookup` is heuristic.** Substring matching on tensor names (`to_q` → Matmul, `.norm` → RMSNorm) with rank-based fallback. No validation that the matched op's expected shape matches the actual tensor shape — silent miscompile risk if a checkpoint name pattern collides.
- **No offset-alignment validation.** Parser doesn't check that safetensors archive offsets land on 64-byte boundaries before being handed to device code (spec §5.1 requires `uintptr(workspaceBase) % 64 == 0` and all interval offsets aligned). Today the alignment is enforced device-side; that's fine, but a fast-fail check in the parser would catch malformed archives earlier.
- **`dataset/` is partially scaffolded.** `server.go` and stream helpers are incomplete. Lower priority — training data isn't a Phase 1–5 concern.

`hf` does not import `puter` or `caramba` — clean leaf in the dependency graph.

---

## 5. caramba — orchestrator

### 5.1 Role

Caramba is the application layer: imports `puter`, `manifesto`, `hf`, and `qpool`; instantiates device pools; passes a compiled program through `manifesto/runtime.Orchestrator`; exposes a Fiber HTTP API; ships a CLI (`program`, `chat`, `research`, `serve`); maintains a fusion catalog, a multi-armed-bandit tuner with EFE arm selection, research project scaffolding, and a React/Flume frontend for node editing.

It does not implement any of §2–§5 itself. It consumes what's there.

### 5.2 Spec touchpoints owned by caramba

| Area | Status |
|---|---|
| End-to-end program load → compile → execute glue (`cmd/program.go`) | ✓ |
| Device backend wiring (`pkg/backend/compute/backend.go`) | ✓ |
| Sharding mesh declaration | ✓ |
| Fusion catalog (`pkg/backend/compute/fusion/catalog.go`) — `matmul+bias+gelu`, `layernorm+residual`, `dequant+matmul` | ⚠️ seeded, **no parity tests** |
| Distributed compute (`distributed/`, `collective/`: process group, AllReduce, AllGather) | ✗ stubs |
| Compute HTTP handlers (`pkg/backend/compute/service.go` `operation`, `optimizer`, `block`) | ✗ all return nil stubs |
| Phase 7 verification harness | ✗ |
| Model checkpoint → device memory (`modelscope/`) | ✗ listing only |
| Network/DHT (`pkg/network/dht/`) | ⚠️ scaffolded, never invoked |
| Notary/ledger (`pkg/notary/`) | ⚠️ defined, unused |
| TUI (`pkg/tui/`) | ⚠️ coded, not wired into any cmd |

### 5.3 Notable bit of debt

`pkg/backend/compute/backend.go` keeps `devices map[DeviceID]device.Backend` and resolves via `Device(id)` map lookup. Compile-time today, but if `PickDevice` ends up called per node in the execution DAG, that map becomes a §7 hot-path violation. Pre-resolve to a flat slice when the async executor lands.

### 5.4 Biggest gaps

1. **Fusion catalog has zero parity tests.** Per AGENTS.md and §7 of the spec, fused ops must match the unfused reference at N ∈ {1, 7, 64, 1024, 8192}.
2. **Compute HTTP service has no actual handlers.** All three (`operation`, `optimizer`, `block`) return nil.
3. **No distributed execution.** Process group, AllReduce, AllGather all stubs.
4. **Modelscope doesn't load into device memory.** Lists files only.

---

## 6. Cross-cutting punch list — prioritized

This is the order I'd suggest. Each item is roughly self-contained.

### P0 — unblocks everything else
1. **Fix `device/interface.go` zero-host-sync (§2.2). ✓ DONE.** All 16 methods across 5 families (Reduction, Dot, Losses, Sampling, VSA) migrated to take `dst unsafe.Pointer`. CPU/Metal/CUDA/XLA backends updated, *Native host-side helpers bridged, parity tests adapted. `scripts/check_banned.sh §1 = 0`. Total violations 395 → 379.
   - **P0b — kernel-side device writes.** Metal/CUDA/XLA implementations now have the correct interface signature, but internally they still call `host.ReductionScalar(...)` / equivalent, which does a device→host scalar transfer. Once the static memory planner lands (P1), migrate these to write directly into the caller's workspace slot on device. Tracked as task #17.
2. **Define the spec's IR types in `manifesto/ir`.** `PortType`, `LayoutSchema`, `SemanticKind`, `Constraint`, `PortAllocation`, `StrideFormula`, `WorkspaceLayout`, plus the missing fields on `Topology` and `Node` (§6). Everything in Phases 2–5 reads or writes these.
3. **Create `puter/XLA_GAPS.md`.** Even as a skeleton checklist. §2.1 says it exists; today it doesn't.

### P1 — foundational compiler work
4. **Excise diffusion-specific Go (§6.5).** Delete `manifesto/diffusion/` package and `manifesto/runtime/{scheduler,latents,executor_diffusion,pipeline_scheduler}*.go`. Remove `scheduler.*` and `diffusion.prepare_latents` step dispatch from `executor.go`. Requires landing RNG / shape-inference / scalar-arithmetic atomics first (see §6.5 sequencing).
5. **PortType unification + adaptor synthesis (§4.1–§4.2).** Without this, the compiler can't validate connections or auto-insert Transpose/Cast/Reshape.
6. **FusionAST + elementwise fusion pass (§4.3).** This is the data structure all JIT codegen lowers from. The blueprint code (`compiler.FusionAST`, `compiler.ASTNode`) in the doc's bottom half is concrete enough to drop in.
7. **Liveness analysis + interval-coloring allocator (§5.1).** The blueprint code (`scheduler.MemoryPlanner.AllocateOffsets`) is also concrete. 64-byte alignment is non-negotiable.
8. **Symbolic stride solver (§5.1 + §4 of blueprints).** Required for dynamic batch/seq dimensions.

### P2 — codegen and execution
8. **CPU JIT codegen via LLVM** with CPUID-driven AVX-512/AVX2/SSE2/NEON paths (the spec's later addendum is unambiguous: keep all three x86 paths plus NEON).
9. **GPU JIT codegen** — MSL source generator (Metal), PTX/CUDA C++ via NVRTC. The `ShaderGenerator.GenerateMetal` blueprint is the starting template.
10. **Async DAG executor with stream partitioning and hardware semaphores (§5.2).** Maps `StreamID`/`SyncBarriers` to native queues. No `runtime.Pinner` in the loop. Workspace allocated via `posix_memalign` / `cudaMalloc` / `MTLBuffer`, off the Go heap.

### P3 — XLA path
11. **PJRT bridge + pre-resolved `[]*PjRtBuffer` slot table (§3.1 lines 1223–1237).** Indexed by `offset >> 6`. Disable host-side in-place aliasing for XLA targets.
12. **HLO lowering for every `device.Backend` method.** Parity vs scalar reference, never XLA-vs-XLA.

### P4 — house cleaning in puter
13. **Decide the fate of unplanned families** (`optimizer`, `math`, `shape`, `checkpoint`, `interpretability`, `model_editing`, `neon`). Add to spec, fold into existing families, or remove.
14. **Drop dtype-prefixed filenames** (`f32_*`, `bf16_*`, `fp16_*`, `f64_*`, `int8_*` outside `dequant`/`quant`). Mechanical rename per family.

### P5 — verification and caramba
15. **Parity harness driven from manifesto's compiled topology** — cross-backend differential testing.
16. **Fusion parity tests in caramba** at N ∈ {1, 7, 64, 1024, 8192} per AGENTS.md.
17. **Wire caramba's compute HTTP handlers** (`operation`, `optimizer`, `block`) or remove them.
18. **Distributed compute in caramba** if multi-device is a goal — process group, AllReduce, AllGather.

---

## 6.5 Diffusion contamination — model-specific Go that shouldn't exist

While trying to get HuggingFace-driven YAML manifests compiling, the codebase accreted **diffusion-specific Go code** that bypasses the "compile from atomics" principle. Every model architecture — Llama, FLUX, SD3, BERT, ViT — should decompose into the same atomic ops via a YAML recipe. Today, FLUX-style diffusion has its own Go fast-path in manifesto, and the executor knows what `diffusion.prepare_latents` and `scheduler.timesteps` are. That has to come out.

### What exists today

**`manifesto/diffusion/` — entire package, delete:**

- `manifesto/diffusion/layout.go` — `LatentLayout` struct (`LatentDownsample`, `SnappedHeight`, `PackedHeight`, `ImageSeqLen`, `VAESpatial`, `MidAttnTokens`, `PackedChannels`) and `ComputeLatentLayout`. This is FLUX latent grid arithmetic in Go. Should be manifest-side dimension math (constants, or `manifesto/lower` shape inference reading from a recipe).
- `manifesto/diffusion/latents.go` — `PackLatents` (NCHW → N·HW·C reshape/permute), `SamplePackedLatents` (Gaussian sampling + packing), `PositionID`. The reshape is a generic permute atomic; the Gaussian sampling is a generic RNG atomic. Neither belongs in a `diffusion/` package.

**`manifesto/runtime/` — diffusion-coupled files, delete:**

- `manifesto/runtime/scheduler.go` — `FlowMatchEulerDiscrete` (FLUX flow-match Euler noise scheduler), `SchedulerConfig`, `Timesteps()`, `Delta()`, sigma schedule construction. **This is not the spec's §4.4 DAG scheduler — it's a diffusion noise scheduler.** Confusing naming made it look like compiler work; it's model-architecture work.
- `manifesto/runtime/scheduler_test.go` — tests for the above.
- `manifesto/runtime/pipeline_scheduler.go` and `pipeline_scheduler_test.go` — glue for the above. Audit before deletion to confirm there's nothing salvageable, but expected to go.
- `manifesto/runtime/latents.go` and `latents_test.go` — runtime-side latent handling.
- `manifesto/runtime/executor_diffusion.go` — `runPrepareLatents` implementing the `diffusion.prepare_latents` step. Imports `manifesto/diffusion`.

**`manifesto/runtime/executor.go` — surgical edits, not full delete:**

The step-name dispatch (around lines 152–159) has cases for:
- `"scheduler.timesteps"` → `runSchedulerTimesteps`
- `"scheduler.bind_latents"` → `runSchedulerBindLatents`
- `"scheduler.delta"` → `runSchedulerDelta`
- `"diffusion.prepare_latents"` → `runPrepareLatents`

The `Executor` struct holds `schedulers map[string]*FlowMatchEulerDiscrete` and `Schedulers` in `Options`. All of this needs to go. The remaining executor (axpy step, host ops, generic step dispatch) stays.

### What's OK to keep

- **The YAML manifests for diffusion models** are not the problem. Recipes like `manifesto/asset/template/model/diffusion/flux-1-dev.yml`, `sd3-medium.yml`, `flux-2-klein-4b.yml`, and `model/architecture/flux2.yml` should remain — *provided* they decompose into atomic ops. If they reference step kinds like `diffusion.prepare_latents` or `scheduler.delta`, those references die with the Go code. The recipes will need to be rewritten to use atomics (RNG, permute, elementwise math, matmul, etc.) once the atomics exist.
- **`manifesto/asset/template/runtime/diffusion.yml` and `diffusion-diagnose.yml`** are the runtime pipelines that orchestrate the diffusion loop. Same rule — keep the file if it can be expressed as a normal program over atomics; delete it if it only works against the soon-to-die step kinds.
- **`hf/config/generator.go:161`** references `subfolder + "/diffusion_pytorch_model.safetensors"` — this is just the on-disk filename convention HuggingFace ships diffusion weights with. It's a string literal for I/O, not modeling logic. Keep.

### What's missing that the diffusion code was filling in

Before deletion, the corresponding atomics need to exist somewhere reachable from a YAML recipe. Cross-reference with §2 of this doc:

| Diffusion code does | Atomic that should replace it | Status |
|---|---|---|
| `SamplePackedLatents` Gaussian draw | A general `RandomNormal` / `Gaussian` device op (or a host-side `RNG` host op writing to the workspace at session start) | **Missing** — no RNG op in `device/interface.go`. Either add one to `device.Backend` or treat noise as a host-prepared input tensor. |
| `PackLatents` NCHW → N·HW·C | A general `Permute` or `Reshape` atomic | Partial — `device/cpu/shape/` exists but is unplanned (§2.2 above); the shape adaptor really wants to live in manifesto's adaptor-synthesis pass (§4.2 of spec), not as a device family. |
| `FlowMatchEulerDiscrete.Timesteps()` sigma schedule | Either a host-prepared constant tensor (computed once when the manifest is compiled) or a YAML expression of `arange` + scalar math over atomic elementwise ops. | **Missing** primitives — no `arange`, no scalar-broadcast math primitive declared at manifest level. |
| `Delta(timestep)` scheduler step | YAML recipe over the same elementwise atomics | **Missing** — no manifest-level recipe primitives. |
| `LatentLayout` dimension math | Manifest-level shape inference (`manifesto/lower`) reading scalar fields from the program declaration | Partial — `lower/` does shape inference; needs to know how to derive `packed_height = (height / latent_downsample) / 2 * 2` etc. from declared variables. That's arithmetic, not a new feature — but the manifest format needs to support it. |

### Removal sequencing

This is the order that keeps the tree green:

1. **Land the missing atomics first.** At minimum: an RNG op (either a `device.Backend` method or a host-prepared input), a manifest-level expression of `arange` + scalar-broadcast arithmetic, and shape inference for the latent layout derivations. Without these, deleting the Go fast-paths breaks every diffusion YAML in the repo.
2. **Rewrite the diffusion YAMLs** (`runtime/diffusion.yml`, `runtime/diffusion-diagnose.yml`, the per-model files) to use only atomic step kinds.
3. **Verify** at least one diffusion model end-to-end (a FLUX or SD3 generation) before deletion — establishes the YAML-only path actually works.
4. **Delete** in this order to keep imports clean:
   1. `manifesto/runtime/executor_diffusion.go`
   2. Remove `diffusion.prepare_latents` and `scheduler.*` cases from `manifesto/runtime/executor.go`; delete `schedulers` field and `Schedulers` option.
   3. `manifesto/runtime/scheduler.go` + `scheduler_test.go`
   4. `manifesto/runtime/pipeline_scheduler.go` + `pipeline_scheduler_test.go` (after confirming nothing else imports them)
   5. `manifesto/runtime/latents.go` + `latents_test.go`
   6. `manifesto/diffusion/` (entire package)
5. **Re-run** the diffusion YAML pipelines. They should pass via atomics only.

### The same anti-pattern at the loader layer: `hf/config/generator.go`

The FLUX scheduler in `manifesto/runtime/` and the Llama-only template in `hf/config/generator.go` are **the same anti-pattern at different layers of the stack**, and they both have the same root cause.

The platform's value proposition is single-shaped: every model — language, vision, audio, diffusion, multimodal, or a research architecture nobody has named yet — is a manifest of atomic ops. There are exactly two ways a manifest enters the system:

1. **Hand-authored** — a researcher writes YAML directly.
2. **Synthesized from a HuggingFace checkpoint** — the loader reads the remote repo's `model_index.json` / `config.json` / `*.safetensors` headers and *emits a manifest* from that metadata.

Both paths converge into the same compiler over the same atomic op set. The HF path is not a privileged shortcut; it's just a manifest-generation step that happens to read its inputs from a remote URL instead of a local file.

`hf/config/generator.go` today violates this. It carries one hardcoded Llama template (per the §4 audit) and special-cases it in Go. A BERT model? Add a `BertGenerator`. A FLUX model? Add a `FluxGenerator`. Same precedent as the diffusion scheduler: once one model has a Go fast-path, every model gets one, and the closed-world manifest contract dies. A researcher who wants to load a brand-new architecture from the Hub has to fork the loader.

The right design is symmetric with the rest of the platform: the HF loader composes a manifest by **including/extending atomic sub-manifests**, the same way a hand-authored YAML does. `config.json` tells you `architectures: ["LlamaForCausalLM"]` and the relevant hyperparameters (hidden_size, num_layers, num_attention_heads, vocab_size, …); the loader uses those to instantiate a top-level recipe that `include`s the relevant sub-manifests with the right variables. The generator stops being one Go function per architecture and becomes one Go function: **map config.json fields → manifest variables → include the architecture's sub-manifest**.

### Where the sub-manifests live

`/Users/theapemachine/go/src/github.com/theapemachine/manifesto/asset/` is already the canonical location and **it's already mostly populated**:

| Asset directory | Purpose | Count |
|---|---|---|
| `template/operation/` | Atomic op definitions — `math/matmul.yml`, `math/rmsnorm.yml`, `attention/sdpa.yml`, `attention/gqa.yml`, `shape/transpose.yml`, `activation/swiglu.yml`, `positional/`, `tokenizer/`, `state/`, etc. | ~110 files |
| `template/block/` | Composite blocks — `active_inference/`, `causal/`, `energy/`, `hawkes/`, `markov_blanket/`, `memory/`, `predictive_coding/`, `vsa/` | ~15 files |
| `template/optimizer/` | Optimizer recipes — `adagrad/`, `adam/`, `hebbian/`, `lars/`, `lbfgs/`, `lion/`, `rmsprop/`, `sgd/` | ~10 files |
| `template/model/architecture/` | Full architecture topologies — `flux2.yml` is the worked example, comment-headed with the variables the include caller supplies | small |
| `template/model/{audio,diffusion,llm,vision}/` | Model-family-specific recipes that compose architectures | small |
| `template/manifest/`, `template/runtime/`, `template/devteam/`, `template/latex/` | Meta-templates and program scaffolds | small |

**174 YAML files total. 161 already use the new format (top line `kind: ...`). 13 are still old format and need migration:**

```
template/devteam/agent.yml
template/manifest/architecture.yml
template/manifest/edge.yml
template/manifest/experiment_bench.yml
template/manifest/experiment_train.yml
template/manifest/experiment_tune.yml
template/manifest/node.yml
template/manifest/person.yml
template/manifest/project.yml
template/manifest/team.yml
template/manifest/topology.yml
template/model/architecture/flux2.yml      ← FLUX-2 transformer topology
template/model/architecture/registry.yml
```

The FLUX-2 file is notable: it's already structured as data, the file header comment literally states *"This file is data, not code. Adding a new architecture means writing another sibling YAML, not modifying any Go source."* That comment is the design intent we want to honor everywhere. Migrating it to `kind:` format and verifying it decomposes cleanly into atomics is the proof point that this approach scales to diffusion.

### What this changes about the removal sequencing

Updating the §6.5 removal plan with this in mind:

1. **Migrate the 13 old-format YAMLs to `kind:` format.** Most are meta-templates and trivial. `flux2.yml` is the substantive one — confirm it composes cleanly from `template/operation/` atomics.
2. **Land the missing atomic op recipes** in `template/operation/`:
   - `template/operation/random/normal.yml` (Gaussian sampling) — backed by a new `RandomNormal` device op or a host-prepared input.
   - `template/operation/math/arange.yml`, `linspace.yml` — used by the sigma schedule.
   - `template/operation/math/scalar_broadcast.yml` (or a generic mechanism in elementwise) — used by every schedule-shaped recipe.
   - Verify `template/operation/shape/transpose.yml` + `reshape.yml` cover the NCHW ↔ N·HW·C permute that `PackLatents` was doing in Go.
3. **Rewrite the diffusion runtime/recipe YAMLs** (`template/runtime/diffusion.yml`, `template/runtime/diffusion-diagnose.yml`, `template/model/diffusion/{flux-1-dev,flux-2-klein-4b,sd3-medium}.yml`) to reference only atomic sub-manifests and the migrated `flux2.yml` architecture.
4. **Refactor `hf/config/generator.go`** from "one Go template per architecture" to "map `config.json` fields → variables → include the matching `template/model/architecture/*.yml`". The generator's job becomes purely metadata translation; there is no model-specific Go.
5. **Delete the diffusion Go** in the order from §6.5 above.
6. **Add the missing atomics to `device/interface.go`** (RNG at minimum) so step 2's YAMLs have something real to dispatch to.

### Why this matters

This is the spine of the value proposition. The whole point of the manifesto pipeline (§1, §4.3) is that every model — language, vision, audio, diffusion, multimodal — compiles from the same atomic op set, and the HF loader is a manifest synthesizer that doesn't get to bypass that. If FLUX needs a Go fast-path, the next AI will write one for SD3, then DiT, then video, then audio; if Llama needs its own generator template, the next AI writes BertGenerator and FluxGenerator. The atomic surface stops being closed, the type system can't unify across model families, and the parity claim across CPU/Metal/CUDA/XLA narrows from "any manifest" to "the manifests we hand-tuned." Researchers who can't express their idea by editing YAML go back to PyTorch.

This is currently the **second-most-damaging** thing in the codebase, after the §2.2 zero-host-sync violation in `device/interface.go`. It should be a P1 alongside the IR types work.

---

## 6.6 Known pre-existing test failures (not P0-related)

These exist in the tree from before the P0 contract migration and are documented here so the next agent running `make verify` recognises them and doesn't waste time hunting their cause through recent changes.

### 6.6.1 Duplicate CGo symbols — `device/metal.test` and `pool.test` link errors

`go test ./...` fails to link the `device/metal` and `puter/pool` test binaries with errors like:

```
duplicate symbol '_metal_transformer_status_set' in: …000011.o, …000027.o, …000054.o
duplicate symbol '_metal_dispatch_pool2d'        in: …000015.o, …000045.o
duplicate symbol '_metal_vision_kernel_name'     in: …000015.o, …000045.o
duplicate symbol '_metal_dispatch_timestep_embedding' in: …000003.o, …000005.o
duplicate symbol '_metal_vision_dispatch'        in: …000015.o, …000045.o
duplicate symbol '_metal_vision_status_set'      in: …000015.o, …000045.o
duplicate symbol '_metal_transformer_kernel_name' in: …000011.o, …000027.o, …000054.o
duplicate symbol '_metal_transformer_dispatch'    in: …000011.o, …000027.o, …000054.o
ld: 8 duplicate symbols
```

These are CGo-level — multiple `.m` files under `device/metal/.../native/` define the same C function (`metal_transformer_dispatch`, `metal_vision_dispatch`, etc.). Probably a copy-paste between two family bridges plus a leftover transformer/vision bridge that wasn't cleaned up when work moved into family quintets (§2.3.1). The Metal quintet rules forbid cross-family bridges, but these symbol names suggest pre-quintet leftovers that still get linked in.

**Diagnostic next step:** `grep -rn "metal_transformer_dispatch\|metal_vision_dispatch\|metal_dispatch_pool2d\|metal_dispatch_timestep_embedding" device/metal/`. The first match per symbol that lives in a family quintet stays; any siblings in other locations should be deleted or renamed.

**Action priority:** P3. Blocks `make verify` from going fully green but does not affect the `device.Backend` contract or the kernels themselves (they still build and pass parity individually — only the test-binary link step fails when both family object files get pulled in together).

### 6.6.2 `device/cpu/allocate_test.go:24` — GoConvey type-strict assertion ✓ FIXED

Was: `So(uintptr(pointer)%workspaceAlign, ShouldEqual, 0)` — GoConvey's `ShouldEqual` is type-strict and bare `0` is `int` while the left side is `uintptr`. One-line fix landed: `ShouldEqual, uintptr(0)`.

### 6.6.3 Metal ULP precision drift — `activation`, `layernorm`, `normalization`

Three Metal parity tests fail by 1–3 ULPs over their declared tolerance:

```
TestActivationStandardUnaryMetalParity: lane 26 got=-0.058651008 want=-0.058651045 ulp=10 max=8
TestLayerNormMetalParity:               lane 16 got= 0.026889674 want= 0.026889682 ulp= 4 max=3
TestLayerNormMetalApplyParity:          lane 16 got= 0.026889674 want= 0.026889682 ulp= 4 max=3
TestGroupNormMetalParity:               lane 30 got=-0.1459909   want=-0.1459908   ulp= 6 max=3
```

These are Metal-vs-scalar differences in the low-order bits at specific lanes — most likely fused-multiply-add ordering or sub-ULP rounding in the MSL kernels' Welford / softmax / exp paths. Per AGENTS.md §1, the fix is to make the Metal kernel bitwise-match the scalar reference (or get within the declared ULP budget), NOT to widen the tolerance.

**Diagnostic next step:** for each failing test, narrow the input that produces the failing lane, then trace which MSL operation introduces the drift. Likely candidates: replace `exp(x)` with the same polynomial the scalar uses, force FMA on/off consistently, or rework the reduction order in normalization variance.

**Action priority:** P3. Real correctness issue but isolated to three Metal kernels; does not block contract work, planner work, or diffusion excision.

---

## 7. What's *not* a gap

Worth recording, since the audit also confirmed several things are in good shape:

- Metal quintet (§2.3.1) compliance across all 24 families: hub + per-domain `{family}.go`/`_stub.go`/`.h`/`.metal`/`_bridge_darwin.go` + `native/*.m`. No monolithic MSL, no orphan kernels, no cross-family bridges.
- CUDA family layout mirrors §2.3 — `.cu` per domain, `bridge.cu` per family, dispatch wiring in place.
- PosPop correctly isolated to `device/cpu/pospop/` as a `device.HostBackend`-only family. Metal/CUDA/XLA backends do not embed PosPop.
- No catch-all backend source files anywhere (`device_missing*`, `device_remaining*`, `*_stub_ops*`, `*_extra*`).
- No root forwarding shims (`backend_<family>.go`).
- `hf` does not leak Go heap slices into device code paths.
- All four x86 ISA paths (AVX-512, AVX2, SSE2) plus NEON exist for CPU. The spec's late addendum about not dropping AVX2/SSE2 is already honored.

---

*This document is a snapshot. Once any P0–P1 item lands, re-run the audit — the matrix shifts.*
