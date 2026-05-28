# Implementation Gaps vs. ARCHITECTURE.md

Audit date: 2026-05-24. Scope: `puter`, `manifesto`, `hf`, `caramba` against `puter/ARCHITECTURE.md`.

This document is a punch list, not a redesign. Section numbers refer to `ARCHITECTURE.md` unless noted otherwise.

---

## 0. Executive summary

The four repos together implement roughly half of the architecture by surface area, but the half that's missing is the half that turns kernels into a system: the manifesto compiler/optimizer/codegen/scheduler pipeline (Phases 2–4 of §8) is essentially absent. `puter`'s kernel surface is broad, and its scalar-output `device.Backend` methods now use device-resident `dst unsafe.Pointer` outputs. `hf` is in the best shape relative to its scope. `caramba` is a thin orchestrator that depends on the missing pieces.

What works end-to-end today: load a HuggingFace checkpoint, resolve a recipe, build a minimal IR, and dispatch ops sequentially via the `tensor.Backend` interface that delegates to puter's CPU/Metal kernels. What doesn't exist: PortType unification, adaptor synthesis, fusion AST, JIT codegen for any target, liveness analysis, static interval-coloring allocator, symbolic stride solver, stream-partitioned DAG executor, parity verification harness.

The single most damaging remaining gap is the executor/compiler half of the architecture: stream-aware scheduling, static workspace planning, and full JIT/codegen integration. The old §2.2 scalar-return blocker in `device.Backend` has been migrated to `dst unsafe.Pointer` outputs.

---

## 1. Repository roles

| Repo          | Actual role                                                                                                                                                                                      | Spec section it owns                                                                             |
|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| **puter**     | `device.Backend` implementations: CPU (scalar + AVX-512/AVX2/SSE2/NEON), Metal, CUDA, XLA. Owns the family tree under `device/<backend>/`.                                                       | §2 (`device/interface.go`), §2.3 (backend layout), §3.1 (XLA target).                            |
| **manifesto** | HuggingFace checkpoint compiler today. YAML manifest parser, recipe expander, checkpoint binder, minimal sequential executor. Should own the full compiler/optimizer/codegen/scheduler pipeline. | §1 (manifesto stack), §4 (PortType + composition), §5 (memory planning), §6 (IR), §8 Phases 2–5. |
| **hf**        | HuggingFace Hub client, safetensors parser, tokenizer, model config → manifest generator. Implements manifesto's `Hub` and `Host` interfaces.                                                    | §1 (checkpoint tokens / safetensors), §6 (`WeightToken *types.Token`).                           |
| **caramba**   | Application/orchestrator. Wires puter + manifesto + hf together, exposes HTTP API, CLI (`program`, `chat`, `research`, `serve`), tuner, research project scaffolding.                            | Consumer of §1 stack; partial owner of fusion catalog and Phase 7 verification.                  |

---

## 2. puter — kernel backends

### 2.1 Backend completeness matrix

Families listed in §2.3 of the spec. Cell = source present (not necessarily correct or parity-tested).

| Family               | CPU scalar | AVX-512 | AVX2 | SSE2 | NEON | Metal                | CUDA | XLA     |
|----------------------|------------|---------|------|------|------|----------------------|------|---------|
| activation           | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| elementwise          | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| reduction            | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| dot                  | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| matmul               | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| pool                 | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| convolution          | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| dropout              | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| losses               | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| sampling             | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| embedding            | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| normalization        | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| layernorm            | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| rope                 | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| hawkes               | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| physics              | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| causal               | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| masking              | ✓          | ✓       | ✓    | ✓    | ✓    | ✓ (under attention/) | ✓    | partial |
| attention            | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| vsa                  | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| active_inference     | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| predictive_coding    | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| dequant              | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| quant                | ✓          | ✓       | ✓    | ✓    | ✓    | ✓                    | ✓    | partial |
| pospop *(host only)* | ✓          | ✓       | ✓    | ✓    | ✓    | n/a                  | n/a  | n/a     |

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

| Family    | Methods                                                         | Contract migrated? | Kernel writes to caller `dst` slot? |
|-----------|-----------------------------------------------------------------|--------------------|-------------------------------------|
| Reduction | Sum, Prod, ReduceMin, ReduceMax, L1Norm                         | ✓ done             | ✓ done                              |
| Dot       | Dot                                                             | ✓ done             | ✓ done                              |
| Losses    | MSE, MAE, Huber, BinaryCrossEntropy, KLDivergence, CrossEntropy | ✓ done             | ✓ done                              |
| Sampling  | GreedySample, TopKSample, TopPSample                            | ✓ done             | ✓ done                              |
| VSA       | Similarity                                                      | ✓ done             | ✓ done                              |

**P0a/P0b — Interface and backend write path.** Public method signatures take `dst unsafe.Pointer` and return nothing. CPU stores scalar results using dtype-aware storage. Metal/CUDA/XLA dispatch now resolves the caller's backend-resident `dst` buffer and passes it into the real device submission path rather than materializing the scalar on the host.

`scripts/check_banned.sh §1` enforces the scalar-return side of this contract. Code review and scalar-output parity tests enforce the absence of inline device-to-host scalar materialization.

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

| Spec stage                                                               | Phase | Status    | Where (if any)                                                 | Notes                                                                                                                             |
|--------------------------------------------------------------------------|-------|-----------|----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| YAML parser + block macro expansion                                      | 2.1   | ✓ partial | `parse/`, `expand/`, `lower/`, `ast/`, `catalog/`, `registry/` | Parses, expands `extends`/`repeat`, lowers shapes. No semantic-level macro expansion (active inference blocks, causal heads) yet. |
| `PortType` (DType, ShapeSchema, LayoutSchema, SemanticKind, Constraints) | 2.2   | ✗         | —                                                              | Type doesn't exist. `Port` in `ir/` is just `Tensor *tensor.Tensor`.                                                              |
| Hindley-Milner port unification                                          | 2.2   | ✗         | —                                                              | Not implemented.                                                                                                                  |
| Adaptor synthesis (Transpose/Cast/Reshape insertion)                     | 2.3   | ✗         | —                                                              | Not implemented.                                                                                                                  |
| `FusionAST` + elementwise clustering                                     | 3.1   | ✓         | `manifesto/optimizer/fusion_ast.go`, `fuse.go`, `constant_fold.go`, `rewrite.go` | FusionAST/ASTNode types match spec blueprint §1. `optimizer.Fuse` clusters DAG-shaped subgraphs (chains + sibling fan-in like SwiGLU and residual Add). Constant folding + identity elim + scale-into-Linear flag also land. Tests cover the patterns. |
| CPU JIT codegen (LLVM → AVX-512/AVX2/SSE2/NEON)                          | 3.2a  | ✗         | `manifesto/codegen/cpu.go`                                     | What exists: a tree-walking Go interpreter (`CPUKernel.Run` calls `evalCPU` per element). No LLVM bindings, no IR builder, no SIMD. The interpreter is a reference evaluator for correctness — it is **not** the spec's JIT codegen path. Phase 3.2a needs LLVM + CPUID-driven AVX-512/AVX2/SSE2/NEON emission, both still missing. |
| GPU JIT codegen (MSL / PTX)                                              | 3.2b  | ✗ partial | `manifesto/codegen/metal.go`                                   | MSL **source generator** ships (`MetalKernel.Source` returns `kernel void` source per blueprint §2). What's missing: `MTLLibrary` compilation, pipeline state object caching, and `puter/device/metal` integration that loads + dispatches the compiled pipeline. No CUDA/PTX path either. |
| XLA HLO lowering + PJRT compile cache                                    | 3.2c  | ✗         | —                                                              | Not present in manifesto; stub bridge in `puter/device/xla`.                                                                      |
| Cache-tiling for Matmul/Conv                                             | 3.3   | ✗ partial | `manifesto/optimizer/tiling.go`                                | `optimizer.Tile` attaches a `TileConfig` struct as metadata on every Matmul/Conv node, sized to fit a default L1 budget. **No consumer reads it.** The tiled matmul loops themselves don't exist — codegen for heavy ops is unwritten. The annotation is wiring only. |
| Liveness analysis (`[Start, End]` per port)                              | 4.1   | ✗         | —                                                              | Not implemented.                                                                                                                  |
| Interval-coloring allocator + 64-byte alignment                          | 4.2   | ✗         | —                                                              | Not implemented.                                                                                                                  |
| Symbolic stride solver (`StrideFormula`, `ResolveOffset`)                | 4.3   | ✗         | —                                                              | Not implemented.                                                                                                                  |
| DAG scheduler (stream partition + semaphores)                            | 4.4   | ✗ partial | `runtime/plan.go` `ExecutionPlan.Layers [][]string`            | Topological layering exists. No stream mapping, no `StreamID`/`SyncBarriers` on nodes, no hardware-queue dispatch.                |
| Flat 64-byte aligned workspace                                           | 5.1   | ✗         | `tensor/arena.go`, `tensor/slab.go`                            | Native allocators exist (mmap on Linux/Darwin), but not driven by a static interval allocator.                                    |
| Async non-blocking executor                                              | 5.2   | ✗         | `runtime/executor.go`                                          | Sequential host-side dispatch; map-based input lookups (compile-time, but no streams).                                            |
| Parity harness (CPU scalar vs SIMD vs GPU vs XLA)                        | 7.1   | ✗         | —                                                              | Not present in manifesto.                                                                                                         |

### 3.1.b Static memory planner (Phase 4.1–4.2) — **LANDED**

Liveness analysis + interval-coloring allocator implemented in `manifesto/ir/planner.go`:

- `AssignPortIDs(topology)` assigns sequential unique IDs to every Port in topological order. Idempotent.
- `AnalyzeLiveness(nodes, bindings)` walks the topologically-ordered node list and produces one Interval per distinct output Port. Unproduced inputs (graph inputs, weights) are treated as live from step 0. Each interval's byte size is computed via `PortByteSize`, which resolves symbolic shape dimensions through the bindings map.
- `AllocateOffsets(intervals, align)` runs the interval-coloring algorithm from ARCHITECTURE.md §3 blueprint: sorts intervals by Start, maintains a live set, evicts expired entries, finds the lowest non-conflicting offset for each new entry. Sizes and offsets are rounded up to `align` (defaults to 64 bytes per §5.1).
- `PlanWorkspace(topology, options)` combines them: assigns IDs, runs liveness, allocates, populates `Topology.Workspace` and each `Port.Allocation` in place.

Tests cover: ID assignment uniqueness and idempotency, linear chain liveness, multi-consumer extension to last use, unbound symbol rejection, bound symbol resolution, disjoint-intervals-share-offset (coloring works), overlapping-intervals-distinct-offsets, alignment rounding, scalar port byte size, default alignment.

**What this unblocks:**
- Scalar-output device writes — Metal/CUDA/XLA backends use backend-resident `dst` buffers for reduction/dot/loss/sampling/VSA scalar outputs.
- The async DAG executor — needs `Topology.Workspace` populated before the dispatch loop can pre-materialize pointers.
- Cross-stream sync (Phase 4.4) — the planner is the prerequisite for the stream-partitioner pass that emits SyncEvents.

**Still missing for end-to-end planning:**
- The PortType typer pass (referenced in §3.1.a) — populates Port.Type from the recipe so the planner can compute sizes. Currently the planner requires callers to set Port.Type manually.
- Symbol-binding pipeline — bindings come from Unify in §3.1.a but no pass yet runs Unify across all node connections to collect a global SymbolMap.

---

### 3.1.a PortType unification (Phase 2.2) — **FIRST PASS LANDED**

Direct unification per ARCHITECTURE.md §4.2 is implemented in `manifesto/ir/unify.go`:

- `Unify(producer, consumer PortType) → UnificationResult` returns a unified type plus a `SymbolMap` of bindings (e.g. when producer `[B, 768]` meets consumer `[4, 768]`, Bindings holds `{"B": 4}`).
- DType mismatches return a `UnificationError` with `AdaptorHint: "cast"`. Layout mismatches return `AdaptorHint: "transpose"`. The adaptor-synthesis pass that consumes these hints is Phase 2.3 — still pending.
- Shape unification handles all four cases: static/static, static/symbolic (binds), symbolic/symbolic (same name OK, distinct names add a `SymbolEqualityConstraint`).
- Kind unification: `SemanticGeneric` acts as a wildcard; otherwise strict equality, so a `Logits` port can't be silently consumed where `HiddenState` was expected.
- Constraint validation runs against the unified shape + bindings: `DivisibilityConstraint`, `RangeConstraint`, and `SymbolEqualityConstraint` all enforce at unification time when their inputs are bound.

Tests (`unify_test.go`) cover: identical types, static/static, symbol binding from either side, same/distinct symbols, conflicting bindings, rank mismatch, dtype mismatch with cast hint, layout mismatch with transpose hint, `LayoutUnspecified` deferral, Kind wildcards and mismatches, divisibility enforcement, range enforcement, and constraint deduplication.

**What's still missing in the pipeline:** the unifier ingests two `PortType` values it doesn't yet have any compiler stage producing. The next pieces are:
- A typer pass that walks the recipe and assigns `PortType` to every `Port` (currently `Port.Type` is zero-valued because no pass writes to it).
- The adaptor synthesis pass that consumes `UnificationError.AdaptorHint` and inserts Transpose/Cast/Reshape nodes.
- Connection of these passes into the `manifesto/lower` pipeline.

---

### 3.2 IR conformance vs §6 — **TYPES NOW DEFINED**

**Update:** The foundational types from ARCHITECTURE.md §6 and §4.1 are now declared in `manifesto/ir/` (additive changes only — existing fields preserved for backward compat):

- `port_type.go` — `PortType`, `ShapeSchema`, `Dimension`, `LayoutSchema` enum, `SemanticKind`, `Constraint` interface + three concrete implementations (`DivisibilityConstraint`, `SymbolEqualityConstraint`, `RangeConstraint`).
- `port_allocation.go` — `PortAllocation`, `StrideFormula`, `SymbolMap` with `Resolve()`.
- `workspace.go` — `WorkspaceLayout`, `Interval`, `Interval.Overlaps()`.
- `sync_event.go` — `SyncEvent`.
- `topology.go` updated to add `Workspace`, `InputPorts`, `OutputPorts`.
- `node.go` updated to add `ID`, `JitKernel`, `StreamID`, `SyncBarriers`.
- `port.go` updated to add `ID`, `Type`, `Allocation`.

Unit tests cover construction, stride resolution, interval overlap semantics, and field preservation. **What remains is the *implementation* of the passes that populate these fields** — PortType unification (Phase 2.2), liveness analysis + coloring allocator (Phase 4.1–4.2), DAG scheduler with stream partitioning (Phase 4.4), JIT codegen (Phase 3.2). The types they read/write now exist; their implementations are still pending per the GAPS.md §3.1 pipeline matrix.

---

### 3.2-legacy IR conformance (before types landed)

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

| Spec touchpoint                                                       | Status       |
|-----------------------------------------------------------------------|--------------|
| Safetensors → `types.Token` with `Span{Offset, Length}`               | ✓            |
| Hub client with revision pinning, ETag/SHA caching, parallel snapshot | ✓            |
| Tokenizer load + encode/decode + chat templates                       | ✓            |
| Model config → manifest YAML                                          | ✓ Llama only |
| `manifesto/runtime.Host` (ReadLine, Encode, EmitToken, WriteImage)    | ✓            |
| `manifesto/resolve.Hub` adapter                                       | ✓            |
| Dtype coverage (F32, F16, BF16, I8, I4)                               | ✓            |
| No Go-heap leakage into device workspace                              | ✓            |

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

| Area                                                                                                                  | Status                           |
|-----------------------------------------------------------------------------------------------------------------------|----------------------------------|
| End-to-end program load → compile → execute glue (`cmd/program.go`)                                                   | ✓                                |
| Device backend wiring (`pkg/backend/compute/backend.go`)                                                              | ✓                                |
| Sharding mesh declaration                                                                                             | ✓                                |
| Fusion catalog (`pkg/backend/compute/fusion/catalog.go`) — `matmul+bias+gelu`, `layernorm+residual`, `dequant+matmul` | ⚠️ seeded, **no parity tests**   |
| Distributed compute (`distributed/`, `collective/`: process group, AllReduce, AllGather)                              | ✗ stubs                          |
| Compute HTTP handlers (`pkg/backend/compute/service.go` `operation`, `optimizer`, `block`)                            | ✗ all return nil stubs           |
| Phase 7 verification harness                                                                                          | ✗                                |
| Model checkpoint → device memory (`modelscope/`)                                                                      | ✗ listing only                   |
| Network/DHT (`pkg/network/dht/`)                                                                                      | ⚠️ scaffolded, never invoked     |
| Notary/ledger (`pkg/notary/`)                                                                                         | ⚠️ defined, unused               |
| TUI (`pkg/tui/`)                                                                                                      | ⚠️ coded, not wired into any cmd |

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
   - **P0b — kernel-side device writes. ✓ DONE.** Metal/CUDA/XLA scalar-output implementations resolve and pass the caller's backend-resident `dst` buffer into device dispatch. CPU stores through dtype-aware scalar helpers.
2. **Define the spec's IR types in `manifesto/ir`.** `PortType`, `LayoutSchema`, `SemanticKind`, `Constraint`, `PortAllocation`, `StrideFormula`, `WorkspaceLayout`, plus the missing fields on `Topology` and `Node` (§6). Everything in Phases 2–5 reads or writes these.
3. **Create `puter/XLA_GAPS.md`.** Even as a skeleton checklist. §2.1 says it exists; today it doesn't.

### P1 — foundational compiler work
4. **Excise diffusion-specific Go (§6.5).** Delete `manifesto/diffusion/` package and `manifesto/runtime/{scheduler,latents,executor_diffusion,pipeline_scheduler}*.go`. Remove `scheduler.*` and `diffusion.prepare_latents` step dispatch from `executor.go`. Requires landing RNG / shape-inference / scalar-arithmetic atomics first (see §6.5 sequencing).
5. **PortType unification + adaptor synthesis (§4.1–§4.2).** Without this, the compiler can't validate connections or auto-insert Transpose/Cast/Reshape.
6. **FusionAST + elementwise fusion pass (§4.3).** ✓ **partial — front-end only.** What landed 2026-05-25: `manifesto/optimizer/` with the spec's `FusionAST`/`ASTNode` types, a recursive DAG-shaped fusion pass (`optimizer.Fuse`) covering chains and tree patterns (SwiGLU `Mul(Sigmoid(gate), up)`, residual `Add(Add(a, b), c)`), `optimizer.ConstantFold`, `optimizer.Rewrite` (identity elim + scale-into-Linear flag), and `optimizer.Tile` which **attaches a TileConfig struct as metadata only — no consumer reads it yet**. `manifesto/codegen` ships two things that are **not** the spec's Phase 3.2 JIT codegen: (a) `codegen.CPUKernel` is a tree-walking Go interpreter that evaluates the AST per element — no LLVM, no SIMD, no JIT; it's a reference evaluator only. (b) `codegen.MetalKernel` emits an MSL *source string* but nothing compiles it — no `MTLLibrary` invocation, no pipeline state object, no integration with `puter/device/metal`. `compiler.CompileAssets` runs `optimizer.Run → codegen.AttachKernels` and attaches the interpreter to each `FuseOp`, which is enough to execute fused subgraphs correctly through the new dispatcher, but it does not satisfy Phase 3.2a (CPU LLVM JIT) or Phase 3.2b (Metal/CUDA pipeline compilation). Those two remain **✗**.
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

## 6.5 Diffusion contamination — **GO EXCISION COMPLETE**

**Update:** All diffusion-specific Go has been removed from `manifesto/`. The `manifesto/diffusion/` package is gone; `manifesto/runtime/{executor_diffusion,scheduler,scheduler_test,pipeline_scheduler,pipeline_scheduler_test,latents,latents_test}.go` are gone; all `FlowMatchEulerDiscrete`, `Schedulers` map fields, and step-name dispatch cases (`scheduler.timesteps`, `scheduler.bind_latents`, `scheduler.delta`, `diffusion.prepare_latents`) have been stripped from `runtime/executor.go`, `runtime/session.go`, and `runtime/session_callgraph.go`. The two diffusion runtime YAMLs (`asset/template/runtime/diffusion.yml`, `diffusion-diagnose.yml`) that referenced the deleted step kinds have also been deleted.

`check_banned.sh` for manifesto now reports zero diffusion-related violations (§1-4). The only remaining flags are the 13 old-format YAMLs that pre-date this work and are tracked separately as a migration task.

What still exists (deliberately):
- `ast/program.go` retains `Schedulers map[string]SchedulerDeclaration` for AST parse fidelity. The parser still consumes the `schedulers:` block in YAML programs, but the runtime never reads the parsed data — it's harmless dead-data until the AST is also cleaned up (P3 follow-up).
- `asset/template/model/diffusion/*.yml` (FLUX, SD3 architecture templates) — these are model topology definitions over atomics, not Go shortcuts. They're the targets the YAML rewrite will fill in.
- `caramba/cmd/diffusion*.go` CLI subcommands and the corresponding `Makefile` targets — they'll fail at runtime now that the step kinds are gone. They become "broken until atomics + YAML rewrite" and are documented as such.

The historical analysis below is preserved for context.

---

## 6.5-historical Diffusion contamination — model-specific Go that shouldn't exist

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

| Diffusion code does                                 | Atomic that should replace it                                                                                                                                    | Status                                                                                                                                                                                                                                        |
|-----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `SamplePackedLatents` Gaussian draw                 | A general `RandomNormal` / `Gaussian` device op (or a host-side `RNG` host op writing to the workspace at session start)                                         | **Missing** — no RNG op in `device/interface.go`. Either add one to `device.Backend` or treat noise as a host-prepared input tensor.                                                                                                          |
| `PackLatents` NCHW → N·HW·C                         | A general `Permute` or `Reshape` atomic                                                                                                                          | Partial — `device/cpu/shape/` exists but is unplanned (§2.2 above); the shape adaptor really wants to live in manifesto's adaptor-synthesis pass (§4.2 of spec), not as a device family.                                                      |
| `FlowMatchEulerDiscrete.Timesteps()` sigma schedule | Either a host-prepared constant tensor (computed once when the manifest is compiled) or a YAML expression of `arange` + scalar math over atomic elementwise ops. | **Missing** primitives — no `arange`, no scalar-broadcast math primitive declared at manifest level.                                                                                                                                          |
| `Delta(timestep)` scheduler step                    | YAML recipe over the same elementwise atomics                                                                                                                    | **Missing** — no manifest-level recipe primitives.                                                                                                                                                                                            |
| `LatentLayout` dimension math                       | Manifest-level shape inference (`manifesto/lower`) reading scalar fields from the program declaration                                                            | Partial — `lower/` does shape inference; needs to know how to derive `packed_height = (height / latent_downsample) / 2 * 2` etc. from declared variables. That's arithmetic, not a new feature — but the manifest format needs to support it. |

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

| Asset directory                                                                   | Purpose                                                                                                                                                                                               | Count      |
|-----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| `template/operation/`                                                             | Atomic op definitions — `math/matmul.yml`, `math/rmsnorm.yml`, `attention/sdpa.yml`, `attention/gqa.yml`, `shape/transpose.yml`, `activation/swiglu.yml`, `positional/`, `tokenizer/`, `state/`, etc. | ~110 files |
| `template/block/`                                                                 | Composite blocks — `active_inference/`, `causal/`, `energy/`, `hawkes/`, `markov_blanket/`, `memory/`, `predictive_coding/`, `vsa/`                                                                   | ~15 files  |
| `template/optimizer/`                                                             | Optimizer recipes — `adagrad/`, `adam/`, `hebbian/`, `lars/`, `lbfgs/`, `lion/`, `rmsprop/`, `sgd/`                                                                                                   | ~10 files  |
| `template/model/architecture/`                                                    | Full architecture topologies — `flux2.yml` is the worked example, comment-headed with the variables the include caller supplies                                                                       | small      |
| `template/model/{audio,diffusion,llm,vision}/`                                    | Model-family-specific recipes that compose architectures                                                                                                                                              | small      |
| `template/manifest/`, `template/runtime/`, `template/devteam/`, `template/latex/` | Meta-templates and program scaffolds                                                                                                                                                                  | small      |

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

## 6.7 Diffusion-via-YAML rewrite plan

After the Go excision (§6.5), the platform has the closed-world atomic contract but no working diffusion. This section is the design for getting diffusion running again — using only YAML recipes over atomic ops, no model-family Go.

### 6.7.1 Atomic ops required (and their status)

| Op                                             | Family       | Status                                                                                             | Used for                                                              |
|------------------------------------------------|--------------|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `random.normal`                                | random (new) | Runtime setup step landed in `manifesto/runtime` as a seeded host-prepared tensor; puter has CPU scalar + NEON Philox + Metal random code, but no `device.Backend` contract wiring yet; ✗ AMD64, ✗ CUDA, ✗ XLA | Initial noise tensor                                                  |
| `math.arange`                                  | math         | ✗ missing                                                                                          | Timestep array generation                                             |
| `math.linspace`                                | math         | Runtime setup step landed in `manifesto/runtime`; graph/device op YAML still missing                | Alternative scheduler base (sigma schedules from `linspace(1, 0, N)`) |
| `math.scalar_broadcast`                        | elementwise  | Runtime setup step landed in `manifesto/runtime` for schedule tensors; graph/device broadcast semantics still missing | `sigma * x` style operations                                          |
| `shape.permute`                                | shape        | partial — `transpose`/`reshape` exist; need general N-d permute                                    | NCHW ↔ packed-latent grid                                             |
| `math.add`, `math.mul`, `math.sub`, `math.div` | math         | ✓ exist                                                                                            | Sigma scaling, residual update                                        |
| `control.loop_each`                            | control      | ✓ exists                                                                                           | The denoising loop body                                               |
| `state.update`                                 | state        | ✓ exists; scalar state init + bare state-name target support landed in `manifesto/runtime`          | Tracking step_index across the loop                                   |

**Net new YAML to write under `manifesto/asset/template/operation/`:**
- `random/normal.yml` — references the device.Backend RandomNormal method (step 6 of the RandomNormal sequence)
- `math/arange.yml` — `arange(start, stop, step) → vector`
- `math/linspace.yml` — `linspace(start, stop, count) → vector`
- `math/scalar_broadcast.yml` — `(scalar, tensor) → tensor with scalar broadcast`

Each of those YAMLs declares the op interface (inputs, outputs, config); the underlying implementation either already exists as a `device.Backend` method (arange/linspace need adding) or composes from existing ops via a block recipe.

### 6.7.2 Architecture YAML structure (FLUX-2 worked example)

A diffusion model YAML at `template/model/architecture/flux2.yml` (after `kind:` migration) declares:

```yaml
kind: Architecture
name: FLUX-2
op: model.architecture.flux2
description: FLUX-2 transformer denoiser.

# Variables the include caller supplies (already documented in the
# existing flux2.yml header comment).
variables:
  hidden_size:        { type: int }
  num_layers:         { type: int }
  num_single_layers:  { type: int }
  num_attention_heads:{ type: int }
  attention_head_dim: { type: int }
  joint_attention_dim:{ type: int }
  rope_theta:         { type: int }
  eps:                { type: float }
  context_seq_len:    { type: int }
  latent_seq_len:     { type: int }
  in_channels:        { type: int }
  # ...

# Inputs the denoiser network takes per call (NOT the diffusion loop —
# this is just the prediction model).
inputs:
  - { name: latents,       type: tensor, shape: [B, latent_seq_len, in_channels] }
  - { name: text_embedding,type: tensor, shape: [B, context_seq_len, joint_attention_dim] }
  - { name: timestep,      type: tensor, shape: [B] }
outputs:
  - { name: noise_pred,    type: tensor, shape: [B, latent_seq_len, in_channels] }

# The denoiser topology over atomics (the existing flux2.yml content,
# already shaped right per its header comment: "This file is data, not
# code"). Each node is an atomic op already in template/operation/.
system:
  topology:
    nodes:
      - { id: …, op: math.rmsnorm, … }
      - { id: …, op: attention.multihead_joint, … }
      - { id: …, op: activation.swiglu, … }
      # … etc, exactly as flux2.yml already encodes
```

### 6.7.3 Runtime program structure (the diffusion loop)

A diffusion runtime program at `template/runtime/diffusion.yml` (after rewrite) declares:

```yaml
kind: Program
name: diffusion
description: Generate an image by iteratively denoising random noise.

state:
  - { name: step_index, type: int }
  - { name: latents,    type: tensor }

steps:
  # 1) Initialize noise as latents
  - op: random.normal
    config:
      shape: [1, latent_seq_len, in_channels]
      seed:  $config.seed
      dtype: $config.dtype
    out: { dst: latents }

  # 2) Compute sigma schedule (replaces FlowMatchEulerDiscrete.Timesteps)
  - op: math.linspace
    config: { start: 1.0, stop: 0.0, count: $config.num_inference_steps }
    out: { dst: sigmas }

  # 3) Compute timesteps from sigmas (sigma * num_train_timesteps, possibly
  #    with the flow-match shift transform — but that's just arithmetic)
  - op: math.scalar_broadcast
    config: { op: mul, scalar: $config.num_train_timesteps }
    in:  { x: sigmas }
    out: { dst: timesteps }

  # 4) The denoising loop
  - op: control.loop_each
    loop: { over: timesteps, as: timestep }
    body:
      # 4a) Predict noise via the architecture
      - op: model.architecture.flux2
        in:
          latents:        latents
          text_embedding: $input.text_embedding
          timestep:       timestep
        out: { noise_pred: noise_pred }
        weights: $config.weights_uri  # safetensors source

      # 4b) Compute sigma delta for this step
      - op: math.scheduler_delta
        config: { schedule: $sigmas, step_index: step_index }
        out: { dst: sigma_delta }

      # 4c) Update latents: latents = latents - sigma_delta * noise_pred
      - op: math.axpy
        config: { alpha: -1.0 }  # latents <- latents + alpha * (sigma_delta * noise_pred)
        in: { y: latents, x: noise_pred, scale: sigma_delta }
        out: { dst: latents }

      # 4d) Increment step counter
      - op: state.update
        config: { target: step_index, update: increment }

  # 5) Decode latents → image (VAE decoder)
  - op: model.architecture.vae_decoder
    in:  { latents: latents }
    out: { image: image }
    weights: $config.vae_weights_uri

outputs:
  - image
```

The "Flow-Match Euler Discrete" scheduler that used to live in Go becomes just `linspace + scalar_broadcast + per-step delta arithmetic` — all expressible as a manifest.

The `math.scheduler_delta` op above is the only piece that smells model-family. It's actually generic ("given a 1-D schedule tensor and a step index, return the delta to the next entry") and belongs in `math/`. If we don't want a dedicated op, it's `sigmas[step_index] - sigmas[step_index+1]` expressed as `shape.slice + math.sub`.

### 6.7.4 Sequencing

1. **Land the missing atomics:**
   - `random.normal`: runtime setup step landed for host-prepared initial latents; still needs YAML at `template/operation/random/normal.yml` and full `device.Backend` wiring when RNG becomes a graph op.
   - `math.arange`: device-side reference + YAML
   - `math.linspace`: runtime setup step landed; still needs device-side reference + YAML
   - `math.scalar_broadcast`: runtime setup step landed; still needs either a device op or composition from `math.mul` with broadcast semantics

2. **Migrate `template/model/architecture/flux2.yml` to `kind:` format.** It's already structured as the desired data; just needs the `kind: Architecture` header + variables block.

3. **Write the runtime program** `template/runtime/diffusion.yml` per §6.7.3 above. Initial FLUX-2 Klein runtime YAML exists and is parser-covered for variable substitution; next work is correcting graph inputs/scheduler delta semantics and executing through Metal.

4. **Add a VAE decoder architecture template** (the existing diffusion model YAMLs like `template/model/diffusion/flux-1-dev.yml` likely already reference a VAE component; just confirm it's expressed as a sub-manifest over atomics).

5. **End-to-end smoke test:** caramba CLI `diffusion` subcommand re-enabled, loads the rewritten runtime YAML, runs one denoising step, produces a non-noise image. Doesn't need to match prior output bit-for-bit; just needs to produce a recognizable diffusion artifact.

6. **Delete the diffusion CLI plumbing in caramba** if it has any model-specific paths that survived. The `caramba diffusion <prompt>` command becomes a generic "load this YAML, run it, save the image output" — same shape as `caramba chat`.

### 6.7.5 Acceptance criteria

- `manifesto/scripts/check_banned.sh` continues to report zero diffusion violations (already true after §6.5).
- `caramba run diffusion "<prompt>"` produces an image without invoking any Go code that special-cases diffusion.
- Adding a new diffusion variant (say, SD3) requires writing only YAML — no Go.
- The rewritten YAMLs all start with `kind:` and pass `check_banned.sh §5`.

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

## 8. End-to-end runnability gap — `caramba --program runtime/chat.yml`

Audit date: 2026-05-25. Re-checked the path from `caramba/cmd/root.go` → `chat.yml` → first emitted token. The kernels and the parser pieces work in isolation, but four sequential links between them are missing or stubbed. Each one is by itself small; the chain has to be completed end-to-end before any program YAML, FLUX or otherwise, can run.

The current state of the runtime path, in execution order:

| # | Stage                                                                                                                 | File                                                         | Status                                                                                                                                                                                                                                                                                                                                    |
|---|-----------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | `runtime/chat.yml` parsed into `*ast.Program`, `program.Includes["model"] = "hf://meta-llama/Llama-3.2-1B-Instruct"`  | `manifesto/parse/parser.go`                                  | ✓ works                                                                                                                                                                                                                                                                                                                                   |
| 2 | `compiler.CompileAssets` resolves the include and produces `Graphs["model"]` + `ComputeGraphs["model"]`               | `manifesto/compiler/program.go`, `includes.go`               | ✓ **LANDED 2026-05-25** — `ProgramCompiler` now accepts a `compiler.IncludeResolver` (via `WithIncludeResolver`), walks `program.Includes`, and routes each entry through the resolver. `OrchestratorOptions.IncludeResolver` carries one through from caramba.                                                                           |
| 3 | The `hf://…` reference is resolved into a model block YAML via `hfconfig.GenerateYAML(config, source)`                | `hf/program/include_resolver.go`, `hf/config/generator.go`   | ✓ **LANDED 2026-05-25** — `program.NewIncludeResolver` is wired into `caramba/cmd/program.go`. For each `hf://repo-id[#component]` include it downloads `config.json` via `resolve.Hub`, decodes via `hfconfig.ParseConfig`, and emits block YAML via `hfconfig.GenerateYAML`.                                                            |
| 4 | The parsed `ast.Topology` is lowered into a verified `dag.Graph` with proper input/output wiring                      | `manifesto/compiler/topology_lower.go`, `topology_expand.go` | ✓ **LANDED 2026-05-25** — `LowerTopology(*ast.Topology) → *LoweredGraph` materializes both an `ast.Graph` (carrying Op, Attributes, Weights per node) and a parallel `dag.Graph` (carrying scheduling layout). Handles `control.repeat` template substitution (`${i}`, `${i+1}`) over template bodies. Tests in `topology_lower_test.go`. |
| 5 | `ProgramSession` builds an `ExecutionPlan` per compute graph                                                          | `manifesto/runtime/plan.go`, `session.go`                    | ✓ works — now driven by stage 2's populated `ComputeGraphs`.                                                                                                                                                                                                                                                                              |
| 6 | `execution.Backend.CallGraph` dispatches the plan's layers through `device.Backend` methods on the active device pool | `puter/execution/backend.go`, `dispatch.go`, `dispatch_table.go`, `values.go`, `weights.go` | ✓ **LANDED 2026-05-25** — dispatcher walks `request.Plan.Layers`, resolves each node ID against `request.Graph.Nodes`, and routes to one of three paths: (a) `Op == optimizer.FuseOp` invokes the codegen-attached `CPUKernel`; (b) known device ops route through a narrow `executionDevice` interface — `embedding.token`→`Lookup`, `math.rmsnorm`→`RMSNorm`, `math.layernorm`→`LayerNorm`, `projection.linear`/`math.matmul`→`Matmul`, elementwise add/sub/mul/div, activation relu/sigmoid/tanh/gelu/swish; (c) unknown ops surface a clear "unsupported op" error. Weights resolve via an injected `WeightStore` (ErrWeightNotFound is a clean diagnostic). Three unit tests cover the fused-node happy path, the unsupported-op error, and the missing-weight error.                                                                                          |
| 7 | The CPU and Metal `device.Backend` families execute the kernels                                                       | `puter/device/cpu/*`, `puter/device/metal/*`                 | ✓ kernels exist (per §2 matrix) and are now called by stage 6.                                                                                                                                                                                                                                                                            |

The diffusion path (`asset/template/runtime/diffusion.yml` for FLUX Klein-2 4B) hits the same chain — it just has a different include (`hf://black-forest-labs/FLUX.2-Klein-4B`) and a denoising loop instead of a decode loop. Nothing about the chain is model-family-specific, which is good: fixing it once unlocks both runtimes.

### 8.1 What "minimally working" looks like

For CPU + Metal only (XLA and CUDA are out of scope per the user's current direction):

1.  `compiler.CompileAssets` in `manifesto/compiler/program.go`:
    *   Add a `Pool` field (already there) wired to `resolve.Hub` (passed through from the orchestrator).
    *   Walk `program.Includes`. For each `name: source` entry:
        *   `hf://…` → fetch `config.json` via the hub, decode into `hfconfig.Config`, call `hfconfig.GenerateYAML(config, source)` to get block YAML.
        *   Local asset path (no scheme) → read via the catalog FS.
    *   Parse the block YAML through `parse.BlockModelFromYAML` → `ast.Topology` (via `block.TopologyAST()`).
    *   Lower the topology into a `dag.Graph` (stage 4 below).
    *   Build an `ast.Graph` wrapper (`Outputs`, `InputShapes`, `Topology`) and stash it under `Graphs[name]` and `ComputeGraphs[name]`.
2.  New `manifesto/lower` package (or expand existing `lower/` location):
    *   `LowerTopology(*ast.Topology) (*dag.Graph, error)` — for each topology node, create a `dag.Node` with a stable ID, attach `Inputs` by looking up their producer node IDs, validate via `graph.Verify()`.
    *   This is structurally similar to the existing `manifesto/compiler/node_draft.go` but operates on `ast.Topology` rather than on checkpoint tokens.
3.  `execution.Backend.CallGraph` in `puter/execution/backend.go`:
    *   Walk `request.Plan.Layers` (topologically layered node IDs).
    *   For each node, dispatch by op kind to the device backend chosen by `devicePool` (CPU or Metal):
        *   `embedding.token` → `Embedding.Lookup`
        *   `math.rmsnorm` → `LayerNorm.RMSNorm`
        *   `projection.linear` → `Matmul.Matmul`
        *   `positional.rope` → `RoPE.RoPE`
        *   `attention.sdpa` / `attention.flash` → `Attention.ScaledDotProductAttention` / `Attention.FlashAttention`
        *   `attention.kv_cache.write` / `read_concat` → `attention/` Metal-only ops; CPU equivalents exist under `device/cpu/attention/`
        *   `activation.swiglu` → `Activation.SwiGLU` (or `SwiGLUTensors`)
        *   `math.add` → `Elementwise.Add`
    *   Materialize per-node weight pointers via the HF safetensors token table (already produced by `hf/safetensors/parser.go`).
    *   Materialize per-node output tensors via `tensor.Backend.Upload`/`Allocate`.
    *   Write the graph's declared output ports back into `GraphCallResult.Outputs`.
4.  CPU and Metal device pickers (`puter/pool`) already do CPU+Metal discovery. No XLA/CUDA dependency.

### 8.2 Why this matters for §6.7 (FLUX-via-YAML)

The §6.7 rewrite plan assumes the chain in §8 already works. Adding atomic ops, migrating YAMLs, and writing a diffusion runtime program are all downstream of compiler stages 2–4 actually being implemented. The FLUX Klein-2 4B runtime YAML can be drafted today, but it will sit dormant until the chain lights up.

### 8.3 Priority order, given the user's "CPU + Metal only" direction

1.  ~~**8.1 step 1** (compiler resolves Includes via HF loader)~~ — **DONE 2026-05-25**.
2.  ~~**8.1 step 2** (topology → dag.Graph lowering)~~ — **DONE 2026-05-25**.
3.  ~~**8.1 step 3** (execution.Backend.CallGraph dispatch)~~ — **DONE 2026-05-25**.
4.  ~~Diffusion runtime YAML for FLUX Klein-2 4B~~ (`asset/template/runtime/diffusion.yml`) — **DONE 2026-05-25** (a `caramba diffusion` subcommand also landed in `caramba/cmd/diffusion.go`). Runtime setup atoms for `random.normal`, `math.linspace`, and `math.scalar_broadcast` have now landed in `manifesto/runtime`; execution still depends on graph-side scheduler delta semantics and the remaining FLUX/VAE op binds. `convolution.conv2d` bind wiring for the VAE path has landed.
5.  **WeightStore implementation** — concrete `puter/execution.WeightStore` backed by `hf/safetensors` parsing. The dispatcher's hook exists (`execution.Backend.WithWeights`) but caramba currently wires the nil fallback. End-to-end token emission needs real weight loading.
6.  Once chat.yml emits a token on CPU, validate the same chain on Metal, then verify FLUX Klein-2 4B end-to-end.

XLA gating (§6.6.1 metal test link errors, §6.6.3 metal ULP drift) is unchanged P3 and does not block the chat path.

### 8.4 Specifics of step 3 — dispatcher design

The dispatcher in `puter/execution/backend.go` walks `request.Plan.Layers` and for each node ID resolves the matching `*ast.GraphNode` from `request.Graph.Nodes`. From there:

*   **Op routing.** A switch on `astNode.Op` maps to one `device.Backend` method. Indicative table:

    | `ast.GraphNode.Op`  | `device.Backend` method               |
    |---------------------|---------------------------------------|
    | `embedding.token`   | `Embedding.Lookup`                    |
    | `math.rmsnorm`      | `LayerNorm.RMSNorm`                   |
    | `math.layernorm`    | `LayerNorm.LayerNorm`                 |
    | `projection.linear` | `Matmul.Matmul`                       |
    | `positional.rope`   | `RoPE.RoPE`                           |
    | `attention.sdpa`    | `Attention.ScaledDotProductAttention` |
    | `attention.flash`   | `Attention.FlashAttention`            |
    | `activation.swiglu` | `Activation.SwiGLU`                   |
    | `math.add`          | `Elementwise.Add`                     |
    | `math.mul`          | `Elementwise.Mul`                     |

*   **Weight binding.** Each `ast.GraphNode.Weights.TensorName` is resolved through the HuggingFace safetensors index (`hf/safetensors`) into an `unsafe.Pointer` + shape. Weights are cached per session — load-once, dispatch-many. The dispatcher does not allocate weight buffers per call.

*   **Activation buffers.** Output tensors for each node are materialized from the `runtime.tensor.Backend` (`stateMemory` injected via the orchestrator). For the §3.1.b static memory planner once it lands, these become workspace offsets; until then, per-node `tensor.Tensor` handles are fine.

*   **Device selection.** `puter/pool.Pool.MemoryBackend()` already picks CPU vs Metal based on host availability. The dispatcher reads the active backend and calls its method directly — no per-call device routing inside the loop.

*   **Output forwarding.** When a `step.Out` ref starts with `state.`, the result is committed to the `StateStore`; otherwise it lands in the executor's `values` map. Both paths already work in `runtime/executor.go::runGraphCall`; the dispatcher only needs to populate `GraphCallResult.Outputs`.

---

*This document is a snapshot. Once any P0–P1 item lands, re-run the audit — the matrix shifts.*
