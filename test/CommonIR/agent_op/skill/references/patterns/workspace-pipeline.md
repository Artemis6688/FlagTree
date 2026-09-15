---
priority: high
---

# Cross-Core Workspace Pipeline / Task Ring (TLE DSA)

## Summary

Use GM workspace tensors as a ring buffer between Cube and Vector scopes, with READY/FREE `sync_block` semaphore pairs controlling a multi-task pipeline, and a **skewed** task loop so that MM1 of task `g` overlaps MM2 of task `g−1` (and Vec1(g) overlaps Vec2(g−1)). This is the AscendC FAInfer overlapping schedule structure.

> **Provenance warning.** The TLE DSA 3-task ring compiles and runs on device but may output **NaN** (sync issue under debugging). Every element below is *compile-proven*, not numerically proven. The minimal lockstep variant (skew=0, single slot) exists precisely to validate numerics before this structure is trusted. Bring up in that order.

## Use When

- Each Q-tile task produces intermediates (S = Q·Kᵀ, P = softmax(S), P·V) flowing Cube → Vector → Cube.
- The Cube pipe must never drain between tasks.
- You are replicating an AscendC FAInfer-style overlapping schedule.

## Avoid When

- No cross-core dependency, or one `sync_block_all` between phases suffices.
- One KV tile per Q-tile — ring overhead exceeds benefit.
- The skew=0 single-slot variant has not passed numerics yet. Fix numerics first; add skew second.

## Pattern

### Step 1: Auto-tuning configuration

Pipeline parameters should be searched automatically via `triton.autotune`; do not hardcode tile sizes.

**Autotune config template:**

```python
import triton

def _generate_fa_configs():
    """Generate the autotune search space for the 3-task pipeline."""
    configs = []
    BLOCK_SIZES = [32, 64, 128, 256]        # BLOCK_M and BLOCK_N share this space; they may be equal
    for BLOCK_M in BLOCK_SIZES:
        for BLOCK_N in BLOCK_SIZES:
            for CB in [1, 2, 4, 8, 16]:
                for SUB_VEC_NUM in [1, 2, 4]:
                    SUB_M = BLOCK_M // SUB_VEC_NUM
                    if SUB_M < 32:          # sub-tile too small to be useful
                        continue
                    configs.append(triton.Config(
                        kwargs={
                            'BLOCK_M': BLOCK_M,
                            'BLOCK_N': BLOCK_N,
                            'CB': CB,
                            'SUB_M': SUB_M,
                        },
                        # compiler options (must be fixed for manual-sync kernels)
                        num_stages=1,
                        disable_auto_inject_block_sync=True,
                        multibuffer=False,
                        enable_hivm_auto_cv_balance=False,
                        disable_auto_cv_work_space_manage=True,
                        enable_tuning_mode=True,
                        unit_flag=False,
                        enable_auto_bind_sub_block=True,
                        limit_auto_multi_buffer_only_for_local_buffer=False,
                    ))
    return configs


def _prune_fa_configs(configs, named_args, **kwargs):
    """Prune configs that are invalid for the given input shape."""
    N_CTX = kwargs.get("S", 4096)
    HEAD_DIM = kwargs.get("DIM", 64)
    valid = []
    for conf in configs:
        BM = conf.kwargs["BLOCK_M"]
        BN = conf.kwargs["BLOCK_N"]
        CB = conf.kwargs["CB"]
        if N_CTX % BM != 0 or N_CTX % BN != 0:
            continue                                # seq_len not divisible
        num_kv_blocks = N_CTX // BN
        if num_kv_blocks < CB or num_kv_blocks % CB != 0:
            continue                                # CB does not divide kv_blocks
        # short sequences prefer small tiles; long sequences prefer large tiles for MMA efficiency
        if N_CTX <= 512 and BM > 128:
            continue
        if N_CTX <= 256 and BM > 64:
            continue
        valid.append(conf)
    return valid


@triton.autotune(
    configs=_generate_fa_configs(),
    key=["S", "DIM"],                               # bucket by seq_len and head_dim
    prune_configs_by={"early_config_prune": _prune_fa_configs},
)
@triton.jit
def pipeline_3task_kernel(
    Q, K, V, Out, workspace_s, workspace_p, workspace_pv,
    workspace_rescale, workspace_expsum, sm_scale,
    ...,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    DIM: tl.constexpr, CB: tl.constexpr, SUB_M: tl.constexpr,
):
    ...
```

**Search dimensions and constraints:**

| Parameter | Search space | Constraints |
|---|---|---|
| `BLOCK_M` | 32, 64, 128, 256 | `S % BLOCK_M == 0`; `BLOCK_M % SUB_VEC_NUM == 0`; prefer smaller values for short sequences |
| `BLOCK_N` | 32, 64, 128, 256 | `S % BLOCK_N == 0`; may equal BLOCK_M |
| `CB` | 1, 2, 4, 8, 16 | `num_kv_blocks % CB == 0`; large CB increases pipeline bubble |
| `SUB_VEC_NUM` | 1, 2, 4 | `BLOCK_M % SUB_VEC_NUM == 0`; `SUB_M >= 32` |
| `RING` | 3 (fixed) | `2 * RING <= 16`; fix first, search only after numeric gate passes |

**Fixed parameters (not part of autotune search):**

| Parameter | Value | Reason |
|---|---|---|
| `RING` | 3 | semaphore count depends on RING, affects kernel structure; fix and verify correctness first |
| `DIM` | = head_dim | determined by the model (64, 128, etc.) |
| `NUM_CORES` | 20 | 910B/C hardware constant |
| compiler options | see `triton.Config` | must be fixed for manual-sync kernels |

**Host-side derived values (outside autotune):**

```python
# These are determined by input shape, not searched
num_seq_blocks = S // BLOCK_M
num_kv_blocks = S // BLOCK_N
conbined_block_num = num_kv_blocks // CB         # tasks per output tile
block_num = num_seq_blocks * Hq * B              # total output tiles
block_num_per_core = block_num // NUM_CORES
rem_block_num = block_num % NUM_CORES
```

**Workspace layout (host-side allocation; shape varies with autotune config):**

```python
workspace_s       [NUM_CORES, RING, CB, BLOCK_M, BLOCK_N]  fp16  # S = Q·Kᵀ
workspace_p       [NUM_CORES, RING, CB, BLOCK_M, BLOCK_N]  fp16  # P = softmax(S)
workspace_pv      [NUM_CORES, RING, CB, BLOCK_M, DIM]      fp16  # P·V partial
workspace_rescale [NUM_CORES, RING, CB, BLOCK_M]            fp32  # rescale factor
workspace_expsum  [NUM_CORES, RING, CB, BLOCK_M]            fp32  # partial row-sum
```

HBM budget: `NUM_CORES × RING × CB × (BLOCK_M×BLOCK_N×2×2 + BLOCK_M×DIM×2 + BLOCK_M×4×2)` bytes. The `prune_configs_by` callback can further filter configs that exceed the HBM limit.

### Step 2: Semaphore banks

One READY/FREE pair per workspace; keep the two directions in disjoint id banks:

```python
# bank 0..RING-1: slot-ready flags; bank RING..2*RING-1: slot-free flags
SEM_S_READY  = 0            # cube->vector : ws_s  slot has data
SEM_P_READY  = ...          # vector->cube : ws_p  slot has data
SEM_PV_READY = ...          # cube->vector : ws_pv slot has data
SEM_S_FREE   = RING + 0     # vector->cube : ws_s  slot consumed
...
```

Or the flattened two-bank scheme: `SEM_BANK_SP = 0`, `SEM_BANK_PV = RING`, slot flag = `bank + g % RING`, constraint `2 * RING <= 16`.

### Step 3: Prime FREE credits, then skewed loop

Consumer scope primes RING FREE credits per producer; producer waits FREE before writing a slot, sets READY after; consumer waits READY before reading, sets FREE after its **last** read of the slot. The skewed loop runs `GT + 1` iterations:

```python
with tle.scope(core_mode="cube"):
    for g in range(GT + 1):
        if g < GT:                       # MM1(g): S -> ws_s[g % RING]
            r = g % RING
            sync_block_wait('vector', 'cube', SEM_S_FREE + r, pipe.PIPE_MTE2, pipe.PIPE_FIX)
            ...  # tl.dot Q·Kᵀ, store S tile to ws_s slot r
            sync_block_set('cube', 'vector', SEM_S_READY + r, pipe.PIPE_FIX, pipe.PIPE_MTE2)
        if g >= 1:                       # MM2(g-1): P·V -> ws_pv[(g-1) % RING]
            r2 = (g - 1) % RING
            sync_block_wait('vector', 'cube', SEM_P_READY + r2, ...)   # P ready?
            ...  # tl.dot P·V, store partial to ws_pv slot r2
            sync_block_set('cube', 'vector', SEM_PV_READY + r2, ...)
            sync_block_set('cube', 'vector', SEM_P_FREE + r2, ...)     # release ws_p slot

with tle.scope(core_mode="vector"):
    # prime: all ws_s / ws_pv slots start FREE
    for r in range(RING):
        sync_block_set('vector', 'cube', SEM_S_FREE + r, ...)
        sync_block_set('vector', 'cube', SEM_PV_FREE + r, ...)
    for g in range(GT + 1):
        if g < GT:                       # Vec1(g): softmax ws_s -> ws_p
            ...
        if g >= 1:                       # Vec2(g-1): rescale + accumulate ws_pv into O
            ...
# epilogue: drain remaining primed credits on both sides
```

### Step 4: Task-skew timing (NR=1 simplified)

```
Task   g=0     g=1     g=2     g=3
C MM1: [=S0=]  [=S1=]  [=S2=]  [=S3=]
C MM2:         [=P0V=] [=P1V=] [=P2V=]
V V1:          [soft0] [soft1] [soft2]
V V2:                  [acc0]  [acc1]
```

The `g < GT` / `g >= 1` guards create the prologue/epilogue; three tasks stay in flight.

## Workspace Data Layout Optimization

> For complete sub-task implementations and semaphore protocol, see `3task-pipeline.md`. This section focuses on **how GM workspace physical layout affects MTE transfer efficiency**.

### Dimension ordering: `[cid, ring_slot, cb_idx, BLOCK_M, ...]`

The outer-to-inner dimension order determines address contiguity and MTE burst efficiency:

| Level | Dimension | Design intent |
|---|---|---|
| Outermost | `cid` (core_id) | Each core owns an exclusive region; no contention |
| 2nd | `ring_slot` (g % RING) | Different pipeline slots within the same core are isolated |
| 3rd | `cb_idx` | CB KV blocks are physically adjacent; the inner `for cb_idx in range(CB)` loop accesses them sequentially |
| Tile-internal | `[BLOCK_M, BLOCK_N]` or `[BLOCK_M, DIM]` | Row-major; innermost dimension is contiguous |

Linear address expansion:

```
workspace_s address = base
  + cid       * (RING * CB * BLOCK_M * BLOCK_N)
  + ring_slot * (CB * BLOCK_M * BLOCK_N)
  + cb_idx    * (BLOCK_M * BLOCK_N)
  + row * BLOCK_N + col
```

### Row-major tile addressing (actual code pattern)

All workspace `tl.make_block_ptr` calls use `strides=(inner_dim, 1), order=(1, 0)` to guarantee row-major layout:

```python
# workspace_s: S = Q·Kᵀ, shape [BLOCK_M, BLOCK_N]
score_bp = tl.make_block_ptr(
    workspace_s + cid * RING * CB * BLOCK_M * BLOCK_N
               + ring_slot * CB * BLOCK_M * BLOCK_N
               + cb_idx * BLOCK_M * BLOCK_N,
    (BLOCK_M, BLOCK_N), (BLOCK_N, 1),       # strides: row-major
    (0, 0), (BLOCK_M, BLOCK_N), (1, 0))     # order: row-major

# workspace_pv: P·V partial, shape [BLOCK_M, DIM]
pv_bp = tl.make_block_ptr(
    workspace_pv + cid * RING * CB * BLOCK_M * DIM
                 + ring_slot * CB * BLOCK_M * DIM
                 + cb_idx * BLOCK_M * DIM,
    (BLOCK_M, DIM), (DIM, 1),               # strides: row-major
    (0, 0), (BLOCK_M, DIM), (1, 0))
```

Row-major guarantees that each row of `BLOCK_N` (or `DIM`) elements is physically contiguous, allowing MTE2/MTE3 to issue contiguous bursts instead of strided scatter.

### Sub-tile access contiguity

Vec1/Vec2 split `BLOCK_M` rows into `SUB_VEC_NUM` sub-tiles of `SUB_M` rows each. Because of the row-major layout, each sub-tile still occupies a contiguous address range:

```python
# Vec1 reading a sub-tile from workspace_s: offset_m = sub_idx * SUB_M
score_load_bp = tl.make_block_ptr(
    workspace_s + ... + offset_m * BLOCK_N,    # offset to sub-tile start row
    (SUB_M, BLOCK_N), (BLOCK_N, 1),            # sub-tile is still row-major
    (0, 0), (SUB_M, BLOCK_N), (1, 0))

# Vec2 reading a sub-tile from workspace_pv
pv_load_bp = tl.make_block_ptr(
    workspace_pv + ... + offset_m * DIM,
    (SUB_M, DIM), (DIM, 1),
    (0, 0), (SUB_M, DIM), (1, 0))
```

Sub-tile `[SUB_M, BLOCK_N]` spans `SUB_M × BLOCK_N` contiguous elements; MTE2 can transfer the entire sub-tile in a single burst.

### rescale / expsum 1D layout

rescale and expsum store one scalar per row; logical shape is `[BLOCK_M]`, but accessed as `[SUB_M, 1]` block_ptr per sub-tile:

```python
rescale_offset = cid * RING * CB * BLOCK_M
               + ring_slot * CB * BLOCK_M
               + cb_idx * BLOCK_M + offset_m

rescale_bp = tl.make_block_ptr(
    workspace_rescale + rescale_offset,
    (SUB_M, 1), (1, 1), (0, 0), (SUB_M, 1), (1, 0))

# reshape to 1D after load:
rescale = tl.reshape(tl.load(rescale_bp).to(tl.float32), (SUB_M, ))
```

`(SUB_M, 1)` shape + `(1, 1)` strides = `SUB_M` contiguous fp32 elements.

### dtype selection

| Workspace | dtype | Reason |
|---|---|---|
| `workspace_s` | fp16 | Intermediate S = Q·Kᵀ; MMA output is cast via `.to(fp16)` before store, **halving GM write bandwidth** |
| `workspace_p` | fp16 | Softmax probabilities; MM2's `tl.dot` operands require fp16 |
| `workspace_pv` | fp16 | P·V partial; Vec2 casts to `.to(tl.float32)` after load for accumulation |
| `workspace_rescale` | fp32 | `exp(neg_max_new - neg_max_prv)` is precision-sensitive; fp16 overflows under extreme max differences |
| `workspace_expsum` | fp32 | Row-sum accumulation is precision-sensitive; fp16's 65504 upper bound is insufficient |

Host-side allocation matches kernel-side dtypes:

```python
workspace_s       = torch.empty((NUM_CORES, RING, CB, BLOCK_M, BLOCK_N), dtype=torch.float16, device=device)
workspace_p       = torch.empty((NUM_CORES, RING, CB, BLOCK_M, BLOCK_N), dtype=torch.float16, device=device)
workspace_pv      = torch.empty((NUM_CORES, RING, CB, BLOCK_M, DIM),     dtype=torch.float16, device=device)
workspace_rescale = torch.empty((NUM_CORES, RING, CB, BLOCK_M),          dtype=torch.float32, device=device)
workspace_expsum  = torch.empty((NUM_CORES, RING, CB, BLOCK_M),          dtype=torch.float32, device=device)
```

### Alignment requirements

MTE2 (GM→UB) and MTE3 (UB→GM) burst transfers require start-address alignment:

- **512-byte alignment**: ensures bursts begin at an HBM bank boundary, avoiding split bursts.
- Tile start address = `base + (cid × ... + ring_slot × ... + cb_idx × tile_size) × elem_bytes`.
- fp16 with `BLOCK_N >= 256` or `DIM >= 256` automatically satisfies 512B (256 × 2B = 512B).
- fp16 with BLOCK_N=128: tile row width = 256B; two adjacent rows combine into one 512B burst. Row-major layout makes adjacent rows physically contiguous, so hardware handles this automatically.
- fp32 with `BLOCK_M >= 128` automatically satisfies 512B (128 × 4B = 512B).

## What To Verify After Applying

- Semaphore pairing: every `sync_block_set` has exactly one matching `sync_block_wait` with identical `(sender, receiver, event_id, sender_pipe, receiver_pipe)` five-tuple.
- `2 * RING <= 16` (event_id budget).
- Prologue prime count = RING; epilogue drains every primed credit.
- Guards `g < GT` / `g >= 1` match the skew; loop trip count = `GT + 1`.
- GM workspace HBM budget: `NUM_CORES × RING × CB × tile_size × 5`.
- **Numeric gate before any RING/CB tuning.** If NaN appears: bisect → RING=1 → disjoint ids → skew=0.
- Workspace addressing: verify that every `tl.make_block_ptr` base offset computation is consistent with the `[cid, ring_slot, cb_idx, ...]` dimension order.

## Related Patterns

- `cv-sync`: semaphore mechanics, prime/drain, launch options — read first.
- `double-buffer`: per-core L1 ping-pong inside each task.
- `explicit-memory`: workspace tensors and capacity budgets.
