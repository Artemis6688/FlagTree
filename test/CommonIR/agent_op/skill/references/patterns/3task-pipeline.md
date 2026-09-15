---
priority: high
---

# 3-Task CV Pipeline (TLE DSA)

> **Hardware:** Ascend 910B/C. Both use the A2_A3 compile path.

## Summary

Split a Cube→Vector→Cube data flow (e.g. Flash Attention's S=Q·Kᵀ → P=softmax(S) → O=P·V) into 4 sub-tasks pipelined via a GM workspace ring buffer and cross-core semaphores. Cube and Vector engines overlap: MM1 of task `g` runs in parallel with MM2 of task `g−1`, and Vec1(g) overlaps Vec2(g−1). This is the core scheduling pattern for multi-stage matmul+elementwise kernels on TLE DSA.

## Use When

- Kernel has a multi-stage Cube→Vector→Cube data flow with intermediates too large for on-chip buffers.
- The Cube pipe must stay continuously busy (no stalls between stages).
- Replicating an AscendC FAInfer-style overlapping schedule.

## Avoid When

- No cross-stage dependency; a single `sync_block_all` between phases suffices.
- Only 1 KV block per Q-tile — ring overhead exceeds benefit.
- The basic lockstep (skew=0, single-slot) variant has not passed the numeric gate. Fix correctness first; add skew second.

---

## 1. Pipeline Structure

### 1.1 Four sub-tasks

| Sub-task | Scope  | Reads from         | Writes to                  | Description                     |
|----------|--------|--------------------|----------------------------|---------------------------------|
| MM1(g)   | Cube   | Q (GM), K (GM)     | workspace_s[ring_slot]     | S = Q · Kᵀ per CB KV-blocks    |
| Vec1(g)  | Vector | workspace_s        | workspace_p, rescale, expsum | Online softmax → P, rescale, expsum |
| MM2(g−1) | Cube   | workspace_p, V (GM) | workspace_pv[prev_slot]   | O_part = P · V per CB blocks    |
| Vec2(g−1)| Vector | workspace_pv, rescale, expsum | acc_o (register) / Out (GM) | Rescale-accumulate; final output on last block |

### 1.2 Skewed overlap

MM2/Vec2 processes task `g−1` while MM1/Vec1 processes task `g`. The outer loop runs `num_global_tasks + 1` iterations. Guards:

- `g < GT`: enable MM1 and Vec1 (produces new data).
- `g >= 1`: enable MM2 and Vec2 (consumes previous task's data).

### 1.3 Timing diagram (RING=3)

```
Task   g=0     g=1     g=2     g=3     g=4
C MM1: [=S0=]  [=S1=]  [=S2=]  [=S3=]
C MM2:         [=P0V=] [=P1V=] [=P2V=] [=P3V=]
V V1:  [soft0] [soft1] [soft2] [soft3]
V V2:          [acc0]  [acc1]  [acc2]  [acc3]
```

### 1.4 Kernel structure (TLE scope pattern)

```python
for g in range(num_global_tasks + 1):

    with tle.scope(core_mode="cube"):
        if g < num_global_tasks:
            # MM1(g): S = Q·Kᵀ → workspace_s[ring_slot]
            ...
        if g >= 1:
            # MM2(g-1): O_part = P·V → workspace_pv[prev_ring_slot]
            ...

    with tle.scope(core_mode="vector"):
        if g < num_global_tasks:
            # Vec1(g): softmax(S) → workspace_p, rescale, expsum
            ...
        if g >= 1:
            # Vec2(g-1): acc_o = acc_o * rescale + O_part; finalize on last block
            ...
```

Key: `tle.scope(core_mode=...)` opens/closes each scope within every loop iteration. This compiles and runs correctly; if issues arise, hoist scope outside the loop as a fallback.

---

## 2. GM Workspace Data Layout

### 2.1 Tensor shapes and dtypes

| Workspace         | Shape                                      | dtype  | Reason                                                        |
|-------------------|--------------------------------------------|--------|---------------------------------------------------------------|
| workspace_s       | [NUM_CORES, RING, CB, BLOCK_M, BLOCK_N]   | fp16   | Score matrix; halves bandwidth vs fp32                        |
| workspace_p       | [NUM_CORES, RING, CB, BLOCK_M, BLOCK_N]   | fp16   | Softmax probabilities; MM2's `tl.dot` requires fp16 operands |
| workspace_pv      | [NUM_CORES, RING, CB, BLOCK_M, DIM]       | fp16   | P·V partial; Vec2 casts to fp32 after load for accumulation  |
| workspace_rescale | [NUM_CORES, RING, CB, BLOCK_M]            | fp32   | `exp(neg_max_new − neg_max_prv)` is precision-sensitive       |
| workspace_expsum  | [NUM_CORES, RING, CB, BLOCK_M]            | fp32   | Row-sum accumulation; fp16 upper bound (65504) is insufficient|

Host-side allocation:

```python
workspace_s       = torch.empty((NUM_CORES, RING, CB, BLOCK_M, BLOCK_N), dtype=torch.float16, device=device)
workspace_p       = torch.empty((NUM_CORES, RING, CB, BLOCK_M, BLOCK_N), dtype=torch.float16, device=device)
workspace_pv      = torch.empty((NUM_CORES, RING, CB, BLOCK_M, DIM),     dtype=torch.float16, device=device)
workspace_rescale = torch.empty((NUM_CORES, RING, CB, BLOCK_M),          dtype=torch.float32, device=device)
workspace_expsum  = torch.empty((NUM_CORES, RING, CB, BLOCK_M),          dtype=torch.float32, device=device)
```

### 2.2 Dimension ordering

Dimensions are ordered `[cid, ring_slot, cb_idx, tile_rows, tile_cols]` (outermost → innermost). This ordering ensures:

| Dimension  | Purpose                                           |
|------------|---------------------------------------------------|
| cid        | Per-core isolation; no cross-core contention       |
| ring_slot  | `g % RING`; next task's slot is physically adjacent |
| cb_idx     | KV-block index within one task                     |
| tile (M×N) | Innermost; contiguous in memory for MTE burst      |

### 2.3 Linear address computation

The flat element offset for a tile at `[cid, ring_slot, cb_idx]`:

```
offset = cid * (RING * CB * BLOCK_M * BLOCK_N)
       + ring_slot * (CB * BLOCK_M * BLOCK_N)
       + cb_idx * (BLOCK_M * BLOCK_N)
```

For sub-tile access (see §4), add `offset_m * BLOCK_N` where `offset_m = sub_idx * SUB_M`.

### 2.4 Row-major tile addressing with `tl.make_block_ptr`

All 2D workspace tiles use row-major layout: `strides=(inner_dim, 1), order=(1, 0)`.

```python
# 2D tile (e.g. workspace_s, workspace_p, workspace_pv)
bp = tl.make_block_ptr(
    workspace + offset,                   # base pointer with flat offset
    (BLOCK_M, BLOCK_N),                   # shape
    (BLOCK_N, 1),                         # strides: row-major
    (0, 0),                               # offsets
    (BLOCK_M, BLOCK_N),                   # block_shape
    (1, 0),                               # order: row-major
)
```

For sub-tile access within a BLOCK_M tile:

```python
offset_m = sub_idx * SUB_M
sub_bp = tl.make_block_ptr(
    workspace + offset + offset_m * BLOCK_N,  # shift by sub-tile rows
    (SUB_M, BLOCK_N),                          # sub-tile shape
    (BLOCK_N, 1),                              # strides unchanged
    (0, 0),
    (SUB_M, BLOCK_N),
    (1, 0),
)
```

### 2.5 1D workspace addressing (rescale, expsum)

rescale and expsum are 1D per row but stored as 2D `(SUB_M, 1)` block_ptrs to satisfy `tl.make_block_ptr` requirements, then reshaped via `[:, None]` / load:

```python
rescale_offset = cid * RING * CB * BLOCK_M + ring_slot * CB * BLOCK_M + cb_idx * BLOCK_M + offset_m
rescale_bp = tl.make_block_ptr(
    workspace_rescale + rescale_offset,
    (SUB_M, 1), (1, 1), (0, 0), (SUB_M, 1), (1, 0)
)
tl.store(rescale_bp, rescale[:, None])      # store
rescale = tl.load(rescale_bp).to(tl.float32) # load: shape [SUB_M, 1] → use [:, 0] or reshape
```

### 2.6 HBM budget check

```
total_bytes = NUM_CORES × RING × CB × (
    BLOCK_M × BLOCK_N × 2  (workspace_s, fp16)
  + BLOCK_M × BLOCK_N × 2  (workspace_p, fp16)
  + BLOCK_M × DIM × 2      (workspace_pv, fp16)
  + BLOCK_M × 4             (workspace_rescale, fp32)
  + BLOCK_M × 4             (workspace_expsum, fp32)
)
```

### 2.7 Alignment requirements

MTE2 (GM→UB) and MTE3 (UB→GM) burst transfers require 512-byte start-address alignment:

- fp16 with BLOCK_N ≥ 256: tile row = 256 × 2B = 512B → aligned.
- fp16 with BLOCK_N = 128: row = 256B; two adjacent rows combine into 512B burst. Row-major layout makes adjacent rows contiguous, so hardware handles this automatically.
- fp32 with BLOCK_M ≥ 128: 128 × 4B = 512B → aligned.

---

## 3. Semaphore Protocol

### 3.1 Event ID banks

```python
assert 2 * RING <= 16, "two banks × RING must fit in event_id [0,15]"
SEM_BANK_SP: tl.constexpr = tl.constexpr(0)    # bank for S/P semaphores
SEM_BANK_PV: tl.constexpr = RING               # bank for PV semaphores
```

Hardware keys a channel by `(event_id, sender_pipe, receiver_pipe)`. Same event_id can be reused when the pipe pair differs.

### 3.2 Six semaphore channels (per ring slot)

| Name      | Direction       | sender_pipe | receiver_pipe | event_id         | Meaning                              |
|-----------|-----------------|-------------|---------------|------------------|--------------------------------------|
| S_READY   | Cube → Vector   | PIPE_FIX    | PIPE_MTE2     | SEM_BANK_SP+slot | MM1 finished writing workspace_s     |
| S_FREE    | Vector → Cube   | PIPE_MTE2   | PIPE_FIX      | SEM_BANK_SP+slot | Vec1 finished reading workspace_s    |
| P_READY   | Vector → Cube   | PIPE_MTE3   | PIPE_MTE2     | SEM_BANK_SP+slot | Vec1 finished writing workspace_p    |
| P_FREE    | Cube → Vector   | PIPE_MTE2   | PIPE_MTE3     | SEM_BANK_SP+slot | MM2 finished reading workspace_p     |
| PV_READY  | Cube → Vector   | PIPE_FIX    | PIPE_MTE2     | SEM_BANK_PV+slot | MM2 finished writing workspace_pv    |
| PV_FREE   | Vector → Cube   | PIPE_MTE2   | PIPE_FIX      | SEM_BANK_PV+slot | Vec2 finished reading workspace_pv   |

S_READY and PV_READY share the same pipe pair (FIX→MTE2) but use different event_id banks (SEM_BANK_SP vs SEM_BANK_PV), so they are distinct channels.

### 3.3 Sync API calls

```python
from triton.experimental.tle.language.dsa.ascend import PIPE, sync_block_set, sync_block_wait

# Signal: "I finished writing/reading workspace_X[ring_slot]"
sync_block_set("cube", "vector", SEM_S_READY + ring_slot, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)

# Wait: "wait until the other scope signals it's done"
sync_block_wait("cube", "vector", SEM_S_READY + ring_slot, PIPE.PIPE_FIX, PIPE.PIPE_MTE2)
```

Arguments: `(sender_scope, receiver_scope, event_id, sender_pipe, receiver_pipe)`.

### 3.4 Prologue: prime semaphores

Before the main loop, pre-arm FREE signals for every ring slot so the first RING tasks can proceed without waiting:

```python
with tle.scope(core_mode="cube"):
    for s in tl.static_range(RING):
        sync_block_set("cube", "vector", SEM_P_FREE + s, PIPE.PIPE_MTE2, PIPE.PIPE_MTE3)

with tle.scope(core_mode="vector"):
    for s in tl.static_range(RING):
        sync_block_set("vector", "cube", SEM_S_FREE + s, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)
        sync_block_set("vector", "cube", SEM_PV_FREE + s, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)
```

### 3.5 Epilogue: drain outstanding tokens

After the main loop, consume all outstanding tokens to avoid hardware deadlock:

```python
with tle.scope(core_mode="cube"):
    for s in tl.static_range(RING):
        sync_block_wait("cube", "vector", SEM_P_FREE + s, PIPE.PIPE_MTE2, PIPE.PIPE_MTE3)

with tle.scope(core_mode="vector"):
    for s in tl.static_range(RING):
        sync_block_wait("vector", "cube", SEM_S_FREE + s, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)
        sync_block_wait("vector", "cube", SEM_PV_FREE + s, PIPE.PIPE_MTE2, PIPE.PIPE_FIX)
```

### 3.6 Per-sub-task sync flow

```
MM1(g):
  wait  S_FREE[slot]          # workspace_s[slot] is safe to overwrite
  ... compute and store S ...
  set   S_READY[slot]         # notify Vec1

Vec1(g):
  wait  S_READY[slot]         # workspace_s[slot] has valid scores
  wait  P_FREE[slot]          # workspace_p[slot] is safe to overwrite
  ... compute softmax, store P, rescale, expsum ...
  set   S_FREE[slot]          # release workspace_s[slot] for next MM1
  set   P_READY[slot]         # notify MM2

MM2(g-1):
  wait  P_READY[prev_slot]    # workspace_p[prev_slot] has valid probs
  ... compute P·V, store to workspace_pv ...
  set   P_FREE[prev_slot]     # release workspace_p for next Vec1
  set   PV_READY[prev_slot]   # notify Vec2

Vec2(g-1):
  wait  PV_READY[prev_slot]   # workspace_pv[prev_slot] has valid P·V
  ... rescale-accumulate, optionally write final output ...
  set   PV_FREE[prev_slot]    # release workspace_pv for next MM2
```

---

## 4. Optimization Techniques

### 4.1 SUB_VEC_NUM sub-tiling (halve UB pressure)

The Vector scope processes each BLOCK_M×BLOCK_N tile in `SUB_VEC_NUM` sequential sub-tiles of `SUB_M = BLOCK_M // SUB_VEC_NUM` rows. This halves peak UB occupancy because only SUB_M-sized buffers are live at once, allowing larger BLOCK_M (e.g. 256) to fit within the 192KB UB limit after PlanMemory optimization.

The Cube scope is unaffected: MM1/MM2 still produce/consume full BLOCK_M tiles. Only Vector sub-tiles them during read/compute.

```python
SUB_VEC_NUM: tl.constexpr = tl.constexpr(2)
SUB_M: tl.constexpr = BLOCK_M // SUB_VEC_NUM

# Inside Vec1/Vec2:
for sub_idx in tl.static_range(SUB_VEC_NUM):
    offset_m = sub_idx * SUB_M
    # load sub-tile from workspace: [SUB_M, BLOCK_N]
    sub_bp = tl.make_block_ptr(
        workspace + base_offset + offset_m * BLOCK_N,
        (SUB_M, BLOCK_N), (BLOCK_N, 1), (0, 0), (SUB_M, BLOCK_N), (1, 0))
    data = tl.load(sub_bp)
    ...
```

Each sub-tile maintains independent state (running max, accumulator, denominator) indexed by `sub_idx`. State arrays are saved/restored per sub-tile iteration because Triton's SSA form does not allow cross-iteration mutation of the same variable:

```python
# Maintain per-sub-tile state
if sub_idx == 0:
    neg_max_even = neg_max_even_0
    neg_max_odd = neg_max_odd_0
else:
    neg_max_even = neg_max_even_1
    neg_max_odd = neg_max_odd_1

# ... compute ...

# Save back
if sub_idx == 0:
    neg_max_even_0 = neg_max_even
    neg_max_odd_0 = neg_max_odd
else:
    neg_max_even_1 = neg_max_even
    neg_max_odd_1 = neg_max_odd
```

### 4.2 CB (combine_batch): group KV blocks per task

Each pipeline task processes `CB` KV blocks before signalling READY. This reduces semaphore overhead (one signal per CB blocks rather than per block) and amortizes pipeline startup cost.

```python
for cb_idx in range(CB):
    kv_idx = idx_in_combine * CB + cb_idx
    # process one KV block ...

# signal READY only after all CB blocks are done
sync_block_set(...)
```

Trade-off: larger CB reduces semaphore frequency but increases pipeline bubble (other scope waits longer). Search CB ∈ {1, 2, 4, 8, 16} via autotune.

### 4.3 RING depth

`RING` controls how many tasks can be in-flight simultaneously across the skewed pipeline. RING=3 is the minimum for the 3-task (4 sub-task) overlap pattern. Larger RING increases HBM usage linearly but allows more pipelining slack.

Constraint: `2 × RING ≤ 16` (event_id budget: two banks × RING slots).

### 4.4 Online softmax with ping-pong max

Vec1 uses standard online softmax with a running maximum. To avoid data dependencies between even/odd KV iterations, maintain two running-max registers per sub-tile (ping-pong):

```python
neg_max_new = tl.minimum(-block_row_max * sm_scale,
                          tl.where(cur_parity == 0, neg_max_even, neg_max_odd))
neg_max_prv = tl.where(cur_parity == 0, neg_max_odd, neg_max_even)

softmax_p = tl.exp(sm_scale * score + neg_max_new[:, None])
rescale = tl.exp(neg_max_new - neg_max_prv)
block_expsum = tl.sum(softmax_p, axis=-1, keep_dims=False)
```

Vec2 applies the rescale factor to accumulate across KV blocks:

```python
if kv_idx == 0:
    acc_o = pv_acc
    softmax_denom = block_expsum
else:
    acc_o = acc_o * rescale[:, None] + pv_acc
    softmax_denom = softmax_denom * rescale + block_expsum

if kv_idx == NUM_KV_BLOCKS - 1:
    output = (acc_o / softmax_denom[:, None]).to(out_dtype)
    tl.store(o_bp, output)
```

### 4.5 dtype selection

| Data           | dtype  | Reason                                                     |
|----------------|--------|------------------------------------------------------------|
| workspace_s    | fp16   | Score matrix; halves GM bandwidth; cast to fp32 in Vec1    |
| workspace_p    | fp16   | Softmax probs; `tl.dot` operands require fp16              |
| workspace_pv   | fp16   | P·V partial; cast to fp32 in Vec2 for accumulation         |
| rescale        | fp32   | `exp(diff)` is precision-sensitive; fp16 overflows easily  |
| expsum         | fp32   | Row-sum accumulation; fp16 max (65504) is insufficient     |
| acc_o          | fp32   | Register-only accumulator; precision through final divide  |
| softmax_denom  | fp32   | Accumulated denominator; same precision as expsum          |

### 4.6 Matmul via plain register tensors

Use `tl.dot` with plain register tensors loaded via `tl.load`. Do not use `tile_copy` + `tile_to_tensor` as dot operands — the 4D cbuf memref produced by `tile_to_tensor` causes a compile failure (the matmul intrinsic only accepts 2D).

```python
# Correct: plain register tensors
q = tl.load(q_ptr)                               # [BLOCK_M, DIM]
k = tl.load(k_bp)                                # [BLOCK_N, DIM]
s = tl.dot(q, tl.trans(k), tl.zeros((BLOCK_M, BLOCK_N), tl.float32))

# Correct: result stored to GM workspace directly
tl.store(workspace_bp, s.to(tl.float16))
```

### 4.7 Causal mask

Applied per sub-tile in Vec1 before softmax:

```python
if IS_CAUSAL:
    q_row_idx = global_head_idx * BLOCK_M + offset_m + tl.arange(0, SUB_M)
    kv_col_idx = kv_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    causal_mask = q_row_idx[:, None] >= kv_col_idx[None, :]
    score = tl.where(causal_mask, score, float("-inf"))
```

---

## 5. Task Distribution

Each AI core processes a contiguous block of output tiles. Static distribution (no load balancing):

```python
cid = tl.program_id(0)
block_start = cid * block_num_per_core + tl.where(cid < rem_block_num, cid, rem_block_num)
block_num = block_num_per_core + tl.where(cid < rem_block_num, 1, 0)
num_global_tasks = block_num * combined_block_num
```

Within each task `g`:
- `idx_in_combine = g % combined_block_num` — position within one output tile's KV iteration.
- `combined_block_idx = g // combined_block_num` — which output tile.
- `ring_slot = g % RING` — which ring buffer slot to use.

GQA (Grouped Query Attention) maps Q-heads to KV-heads: `kv_head_idx = head_idx // gqa_group`.

---

## 6. Compiler Options

These options must be set for manual-sync kernels:

```python
triton.Config(
    kwargs={...},
    num_stages=1,                                    # single-stage (manual pipeline)
    disable_auto_inject_block_sync=True,             # we manage semaphores manually
    multibuffer=False,                               # no compiler auto-multibuffering
    enable_hivm_auto_cv_balance=False,               # manual CV balance
    disable_auto_cv_work_space_manage=True,           # manual workspace management
    enable_tuning_mode=True,                         # enable autotune
    unit_flag=False,
    enable_auto_bind_sub_block=True,
    limit_auto_multi_buffer_only_for_local_buffer=False,
)
```

Launch configuration: `grid = (NUM_CORES,)` — one program per AI core, each driving one Cube + one Vector stream (MIX_1_1 mode via `enable_mixed_cv`).

---

## Hazards

1. **4D cbuf in dot**: `tile_copy` + `tile_to_tensor` produces a 4D cbuf memref; the matmul intrinsic only accepts 2D. Use plain `tl.load` register tensors for `tl.dot` operands.
2. **Ping-pong max correctness**: A negated-max even/odd rescale that applies the wrong parity is **not equivalent** to a single global running max and causes NaN. Use standard online softmax with a single running max per sub-tile, ping-ponged across even/odd KV indices.
3. **CB pipeline bubble**: each ring slot holds CB blocks; READY is signalled only after all CB blocks are done. Larger CB increases bubble. Search via autotune.
4. **scope per loop iteration**: `tle.scope` within each `for`-loop iteration compiles and runs correctly. If issues arise, fall back to hoisting scope outside the loop.
5. **Semaphore token balance**: every `sync_block_set` must have exactly one matching `sync_block_wait`. Prologue primes RING tokens per channel; epilogue drains them. Imbalance causes deadlock.

## What To Verify After Applying

- Semaphore pairing: every `sync_block_set` has exactly one matching `sync_block_wait` with identical `(sender, receiver, event_id, sender_pipe, receiver_pipe)` five-tuple.
- `2 × RING ≤ 16` (event_id budget).
- Prologue prime count = RING; epilogue drains every primed credit.
- Guards `g < GT` / `g >= 1` match the skew; loop trip count = `GT + 1`.
- GM workspace HBM budget: `NUM_CORES × RING × CB × tile_size × 5 workspaces`.
- Workspace addressing: verify every `tl.make_block_ptr` base offset is consistent with the `[cid, ring_slot, cb_idx, ...]` dimension order.
- Numeric gate before any RING/CB tuning. If NaN appears: bisect → RING=1 → disjoint ids → skew=0.

## Related Patterns

- `cv-sync`: semaphore mechanics, prime/drain protocol, launch options.
- `double-buffer`: per-core L1 ping-pong, complementary to cross-core pipeline.
- `explicit-memory`: workspace tensors and capacity budgets.
