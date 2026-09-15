# TLE DSA Optimization Pattern Index

> **Hardware Compatibility:** These patterns are validated for **Ascend 910B/C** only. Other Ascend variants may have different memory hierarchies, core configurations, or instruction sets.

Use this file to choose optimization directions before reading any detailed pattern reference.

Read this index first. Then read only the one or two most relevant detailed pattern files for the current bottleneck.

**Bring-up order matters in TLE DSA.** The recommended ladder is: lockstep minimal CV kernel → skewed workspace pipeline → double buffering → tile tuning. Do not jump to a deeper pattern while an earlier stage has not passed its numeric gate on device.

## High Priority Patterns

### `cv-sync`

- Summary: Split a kernel body into `with tle.scope(core_mode="cube")` / `with tle.scope(core_mode="vector")` regions and coordinate data handoffs with explicit `sync_block_set`/`sync_block_wait` cross-core semaphores. Fundamental TLE DSA pattern — every mixed Cube/Vector kernel starts here.
- Source: [cv-sync.md](patterns/cv-sync.md)

### `explicit-memory`

- Summary: Allocate on-chip buffers explicitly with `tle.dsa.alloc(shape, dtype, mem_addr_space)` into UB / L1 / L0A / L0B / L0C instead of letting the compiler place values; convert with `to_tensor(writable=False)` for dot operands and slice with `subview`. Required before any other Expert memory pattern — and the only way to control the UB budget that killed earlier FA attempts.
- Source: [explicit-memory.md](patterns/explicit-memory.md)

### `double-buffer`

- Summary: Overlap MTE data prefetch with Cube compute via ping-pong buffering. In TLE DSA there are three mechanisms: `tle.dsa.pipeline(num_stages=2)` auto software pipelining, one 2×-sized buffer + `subview` ping-pong, and explicit prefetch ordered with barriers. **Never** use two same-shape `tile_alloc`s — the compiler merges them into one physical cbuf.
- Source: [double-buffer.md](patterns/double-buffer.md)

### `workspace-pipeline`

- Summary: Use GM workspace tensors as a ring buffer between Cube and Vector scopes, with READY/FREE `sync_block` semaphore pairs and a skewed task loop (MM1(g) ∥ MM2(g−1), Vec1(g) ∥ Vec2(g−1)) so three tasks stay in flight. This is the AscendC FAInfer overlapping schedule structure.
- Source: [workspace-pipeline.md](patterns/workspace-pipeline.md)

### `3task-pipeline`

- Summary: Complete 3-task CV pipeline data flow: MM1(Q·Kᵀ) → Vec1(softmax) → MM2(P·V) → Vec2(accumulate), including GM workspace layout, semaphore bank design, prologue/epilogue prime/drain, sub-tile UB optimization, online softmax, and TLE usage patterns for code generation.
- Source: [3task-pipeline.md](patterns/3task-pipeline.md)

## Generated Pattern Summaries

### `cv-sync`

- Priority: high
- Source: [cv-sync.md](patterns/cv-sync.md)
- Use When:
  - A kernel mixes `tl.dot` (Cube) with element-wise math, reductions, or softmax (Vector) in one compute flow.
  - You need explicit control over when GM workspace writes are visible across cores (the only legal Cube↔Vector exchange path is L0C →FIX→ GM →MTE2→ UB).
  - The kernel benefits from overlapping Cube MMA with Vector softmax, requiring precise cross-core semaphore placement.
  - You are porting an AscendC kernel that uses `SetFlag/WaitFlag` or cross-core notify — `sync_block_set/wait` is the direct analog.
- Avoid When:
  - The kernel is purely element-wise (no `tl.dot`) — scope separation adds complexity without benefit.
  - A plain Triton kernel with compiler-inserted sync already produces correct and performant results.

### `explicit-memory`

- Priority: high
- Source: [explicit-memory.md](patterns/explicit-memory.md)
- Use When:
  - The compiler's automatic placement is suboptimal or blows a budget (UB ~192 KB/core practical limit; explicit UB allocs interact multiplicatively with the `multibuffer` launch option).
  - You need exact buffer sizes/lifetimes to fit hardware limits (L0C 64 KB total for L0A+L0B+L0C).
  - Double-buffering requires two buffer sets at specific hardware levels.
  - You need GM workspace tensors as kernel parameters for cross-core exchange.
- Avoid When:
  - You are still prototyping kernel structure — explicit memory slows iteration.
  - The plain-Triton kernel already fits and performs.

### `double-buffer`

- Priority: high
- Source: [double-buffer.md](patterns/double-buffer.md)
- Use When:
  - The K/V load loop is memory-bound — MTE2 copy time dominates Cube compute.
  - L1 capacity holds two buffer sets (or one 2×-sized buffer).
  - You are already past the cv-sync bring-up stage with a passing numeric gate.
- Avoid When:
  - The kernel is compute-bound — double buffering won't help.
  - You were about to allocate two same-shape `tile_alloc`s as ping-pong — use one 2× buffer + `subview` instead (compiler merges identical allocs).
  - You were about to chain `subview → to_tensor → trans → tl.dot` — this lowering path is a known on-device failure; keep dot operands as whole-tile `to_tensor(writable=False)` or plain register tensors.

### `workspace-pipeline`

- Priority: high
- Source: [workspace-pipeline.md](patterns/workspace-pipeline.md)
- Use When:
  - Each task produces intermediate results (S = Q·Kᵀ, P = softmax(S), P·V) that flow Cube → Vector → Cube.
  - The Cube pipe must never drain — tasks should stay continuously in flight (skew ≥ 1, RING ≥ 3).
  - You are replicating an AscendC FAInfer-style schedule.
- Avoid When:
  - The kernel has no cross-core dependency, or one `sync_block_all` between phases is sufficient.
  - The task count per Q-tile is 1 — ring overhead exceeds benefit.
  - The lockstep (skew=0) single-slot variant has not passed its numeric gate yet — fix numerics first, then add skew.

### `3task-pipeline`

- Priority: high
- Source: [3task-pipeline.md](patterns/3task-pipeline.md)
- Use When:
  - Implementing the full Flash Attention CV pipeline with the MM1→Vec1→MM2→Vec2 four-stage schedule.
  - You need the concrete implementation details: semaphore bank design, workspace layout, sub-tile UB optimization, online softmax data flow.
  - Building or modifying a multi-stage Cube→Vector→Cube pipeline kernel.
- Avoid When:
  - Not an FA scenario — for general CV-mix kernels, `cv-sync` is sufficient.
  - Not yet familiar with the basic workspace-pipeline concepts — read `workspace-pipeline.md` first.
