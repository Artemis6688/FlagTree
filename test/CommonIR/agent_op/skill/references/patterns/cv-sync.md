---
priority: high
---

# CV Scope Separation with Manual Synchronization (TLE DSA)

## Summary

Split a kernel body into `with tle.scope(core_mode="cube")` and `with tle.scope(core_mode="vector")` regions, and coordinate Cube↔Vector data handoffs with explicit `sync_block_set` / `sync_block_wait` cross-core semaphores. 

## Use When

- A kernel mixes `tl.dot` (Cube) with element-wise math, reductions, or softmax (Vector) in one compute flow.
- Intermediate results must cross cores: the **only** legal Cube→Vector path is L0C →(FIX pipe)→ GM workspace →(MTE2)→ UB. There is no L0C→UB path.
- You are porting an AscendC kernel with explicit flag discipline, and the compiler's auto-injected sync is either wrong or too conservative.
- The kernel needs Cube and Vector work overlapped (see `workspace-pipeline` for the multi-task version).

## Avoid When

- The kernel is purely element-wise (no `tl.dot`).
- A plain Triton kernel with compiler-managed sync is already correct and fast enough.

## Pattern

### Step 0: Launch options (kernel compile options)

> **Hardware:** Ascend 910B/C only. Both 910B and 910C use the `linalg_to_bin_enable_npu_compile_A2_A3` compile path (the `910_95` path is for 950PR/950DT). See `third_party/ascend/backend/compiler.py` → `NPUOptions` for the full list.

There is no Developer/Expert mode switch — TLE DSA is always explicit. The following options are passed as keyword arguments to the kernel launch call and map to `NPUOptions` fields in `compiler.py`.

**Required for hand-placed sync:**

```python
kernel[grid](...,
             disable_auto_inject_block_sync=True,  # REQUIRED — prevents compiler from adding auto sync on top of yours
                                                    # → --disable-auto-inject-block-sync=True
             num_stages=1,                          # disable automatic software pipelining (default: 2 on 910B/C)
                                                    # → --enable-auto-multi-buffer=False (when combined with multibuffer)
             multibuffer=False,                     # disable auto multi-buffer (default: True on 910B/C)
                                                    # → --enable-auto-multi-buffer=False
)
```

**Recommended for CV-mix kernels with manual sync** (from the CommonIR `_USE_CUSTOM_COMPILE_OPT` reference in `compiler.py`):

```python
kernel[grid](...,
             disable_auto_inject_block_sync=True,
             num_stages=1,
             multibuffer=False,
             enable_hivm_auto_cv_balance=False,     # disable auto CV balance → --enable-hivm-auto-cv-balance=False
             disable_auto_cv_work_space_manage=True, # disable auto CV workspace management
                                                     # → --disable-auto-cv-work-space-manage=True
             enable_tuning_mode=True,                # enable tuning mode → --enable-tuning-mode=True
             unit_flag=False,                        # disable unit flag sync → --enable-hivm-unit-flag-sync=False
             enable_auto_bind_sub_block=True,        # → --enable-auto-bind-sub-block=True
             limit_auto_multi_buffer_only_for_local_buffer=False,
                                                     # → --limit-auto-multi-buffer-only-for-local-buffer=False
)
```

**Optional / situational (A2_A3 path):**

| Option | Compiler flag | Notes |
|---|---|---|
| `sync_solver=True` | `--enable-hivm-graph-sync-solver` + `--enable-hivm-cross-core-gss` | Graph-based sync solver (910B/C sets both flags) |
| `set_workspace_multibuffer=N` | `--set-workspace-multibuffer=N` | Compiler-managed workspace multi-buffering |
| `limit_auto_multi_buffer_of_local_buffer="no-limit"` | `--limit-auto-multi-buffer-of-local-buffer` | Per-buffer multi-buffer limit |
| `enable_ubuf_saving=True` | `--enable-ubuf-saving=True` | UB saving optimization (A2_A3 only; not available on the 910_95 path) |
| `enable_preload=True` | `--enable-preload=True` | Preload optimization (A2_A3 only) |
| `tile_mix_vector_loop=N` | `--tile-mix-vector-loop=N` | Vector loop tiling (A2_A3 only) |
| `tile_mix_cube_loop=N` | `--tile-mix-cube-loop=N` | Cube loop tiling (A2_A3 only) |

> **Note:** `enable_mixed_cv` only takes effect on the `910_95` (950PR/950DT) compile path; the A2_A3 path used by 910B/C does not pass this flag.

`disable_auto_inject_block_sync=True` is the analog of tilelang `TL_ASCEND_AUTO_CV_SYNC: False` — without it the compiler adds its own handshakes on top of yours.

### Step 1: Scope regions

```python
import triton.experimental.tle as tle
from triton.experimental.tle.language.dsa.ascend import PIPE, sync_block_set, sync_block_wait

@triton.jit
def kernel(...):
    with tle.scope(core_mode="cube"):
        ...   # tl.dot, GM->L1 copies, L0C->GM fixout
    with tle.scope(core_mode="vector"):
        ...   # UB element-wise, reduce, GM<->UB copies
```

Known-unverified subtlety: opening/closing a `tle.scope` pair **per iteration inside a runtime `for` loop** compiles and runs but has never passed a numeric gate. If you hit trouble, hoist scopes outside the loop as a structural fallback.

### Step 2: Cross-core semaphores

```python
sync_block_set(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None)
sync_block_wait(sender, receiver, event_id, sender_pipe=None, receiver_pipe=None)
```

- `sender`/`receiver`: strings `"cube"` / `"vector"`, must differ.
- `event_id`: int, budget **0–15** total. Ring designs need `2 * RING <= 16` (READY/FREE pairs).
- Pipe pairs: the channel is keyed by `(event_id, sender_pipe, receiver_pipe)`; **set and wait must use the identical tuple**. Defaults when omitted:
  - cube→vector: `(PIPE.PIPE_FIX, PIPE.PIPE_MTE2)` — FIX retires L0C→GM writes
  - vector→cube: `(PIPE.PIPE_MTE3, PIPE.PIPE_MTE2)` — MTE3 retires UB→GM writes
  - Tutorials also use vector→cube = `(PIPE.PIPE_MTE2, PIPE.PIPE_FIX)`; both appear in working code. Pick one convention and keep it identical on both sides.
- Producer sets, consumer waits. Direction is part of the contract.

### Step 3: 3-Task CV Pipeline (separate document)

The complete 3-task skewed pipeline (MM1 → Vec1 → MM2 → Vec2) data flow, semaphore design, and sub-tile optimization are covered in a dedicated document:

→ **[3task-pipeline.md](3task-pipeline.md)** — full pipeline structure, workspace layout, and online softmax patterns.

Step 4 / Step 5 below still apply to all CV-mix kernels (including non-FA scenarios).

### Step 4: Prime / drain protocol

- Every "buffer empty" credit whose first producer is the consumer side must be **primed before the loop**, one `sync_block_set` per ring/buffer slot. Prime count = ring depth.
- Every primed credit must be **drained after the loop** with a matching `sync_block_wait`, or compilation fails with an extra-set-event error.
- First-iteration underflow: a `sync_block_wait` reachable on the first executed iteration whose credit is produced only under a guard is a structural hang (linter Tier-3). For every wait, trace its credit to either a prologue prime or an unguarded earlier set in program order.

### Step 5: Intra-core pipe events (finer control inside one scope)

```python
from triton.experimental.tle.language.dsa import tile_set_flag, tile_wait_flag, tile_pipe_barrier
tile_set_flag(producer_pipe, consumer_pipe, event_id)   # cross-ENGINE flag within one core
tile_wait_flag(producer_pipe, consumer_pipe, event_id)
tile_pipe_barrier(pipe)                                 # intra-engine barrier
```

`PIPE` enum: `PIPE_S` (scalar), `PIPE_V`, `PIPE_M`, `PIPE_MTE1` (L1→L0), `PIPE_MTE2` (GM→L1/UB), `PIPE_MTE3` (UB→GM), `PIPE_FIX` (L0C→UB/GM), `PIPE_ALL`. There is no `cc_sync` in the codebase — cross-core is `sync_block_*`, debug-only ordering is `tl.debug_barrier()`.

### Step 6: Vector sub-core split

910C has 2 Vector sub-cores per AI Core. Split rows explicitly per lane — never let both lanes write the same GM region (the tilelang `T.serial` race class applies verbatim):

```python
from triton.experimental.tle.language.dsa.ascend import sub_vec_id, sub_vec_num
vid = sub_vec_id()          # tl.tensor in {0, 1}
half = BLOCK_M // 2
rows = vid * half + tl.arange(0, half)
```

Cross-lane exchange must go through a GM tensor (MTE3 out / MTE2 in); an on-chip buffer shared across lanes is device error `507015` (linter Tier-4).

## What To Verify After Applying

- Launch options are correct: `disable_auto_inject_block_sync=True`, `num_stages=1`, `multibuffer=False`; see Step 0 for the full recommended set.
- Every `sync_block_set` has exactly one matching `sync_block_wait` with an **identical five-tuple `(sender, receiver, event_id, sender_pipe, receiver_pipe)`**.
- Direction is correct: the producer scope calls set, the consumer scope calls wait. `sender`/`receiver` must name different cores (`"cube"` vs `"vector"`).
- `event_id` budget: range [0, 15]. The same event_id can be reused by different pipe pairs (hardware distinguishes channels by the `(event_id, sender_pipe, receiver_pipe)` three-tuple). Ring designs must ensure bank size stays within the limit (e.g. `2 * RING <= 16`).
- If using prologue prime / epilogue drain (see Step 4 or `3task-pipeline.md`): every primed set must have a corresponding drain wait, otherwise compilation fails with an extra-set-event error.
- Numeric gate on device: manual-sync errors typically manifest as **NaN or silent data corruption**, not compile errors. Run numeric verification after every sync change.

## Related Patterns

- `explicit-memory`: allocate the GM workspace and UB/L1 buffers this pattern synchronizes.
- `double-buffer`: extends with L1 ping-pong for MTE/Cube overlap.
- `workspace-pipeline`: full RING-deep multi-task pipeline — read after mastering this pattern.
