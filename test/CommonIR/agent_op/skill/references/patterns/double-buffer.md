---
priority: high
---

# Double Buffering (TLE DSA)

## Summary

Overlap MTE data prefetch with Cube compute using ping-pong buffering, hiding GM→L1 latency behind MMA. TLE DSA offers three mechanisms; this file ranks them by how much evidence stands behind each on Ascend.

## Use When

- The K/V load loop is memory-bound — MTE2 copy time dominates Cube compute time.
- L1 capacity holds two buffer sets (or one 2×-sized buffer).
- The lockstep single-buffer version already passes its numeric gate on device.

## Avoid When

- The kernel is compute-bound.
- You cannot fit two buffer sets.
- The previous pipeline stage (cv-sync / workspace-pipeline) has not passed numerics yet — double buffering multiplies the debugging surface.

## The cardinal rule

**Never ping-pong between two same-shape `tile_alloc`s.** `commonir_to_hivm` merges identical shape/dtype allocs into one physical cbuf; both "buffers" alias and the prefetch overwrites the data dot is reading. This compiles and runs but produces WRONG results. Use one 2×-sized buffer plus `subview` slots instead.

## Mechanism A — `dsa.pipeline(num_stages=2)` (preferred first step)

Automatic software pipelining: the compiler double-buffers the `dsa.copy` DMA against compute. Idiomatic AscendC multi-buffer replacement:

```python
# Launch with multibuffer=True (default) so eligible buffers are doubled.
x_ub = tle.dsa.alloc([BLOCK_D], dtype=x_dt, mem_addr_space=UB)   # allocated ONCE

for d_chunk in tle.dsa.pipeline(0, NUM_D_BLOCKS, 1, num_stages=2):
    tle.dsa.copy(x_ptr + offs, x_ub, [tail_d])        # DMA stage (MTE2)
    t = tle.dsa.to_tensor(x_ub).to(tl.float32)        # compute stage (Vector)
    ...

with tle.dsa.hint(inter_no_alias=True):               # copies across iterations don't alias
    tle.dsa.copy(y_ub, out_ptr + offs, [tail_d])
```

Per-tensor multi-buffer annotation also exists: `extension.multibuffer(tensor, 2)` (size must be 2). Related launch knobs: `limit_auto_multi_buffer_only_for_local_buffer=False`, `limit_auto_multi_buffer_of_local_buffer="no-limit"`.

Caution (from the tilelang `auto_pipeline` postmortem, mechanism-independent): pipelining multiplies the footprint of buffers in the loop body. Budget UB explicitly before raising `num_stages`.

## Mechanism B — explicit prefetch + `tl.debug_barrier()`

Explicit prefetch pattern: prefetch next K-tile into the idle slot, `tl.debug_barrier()`, dot from the current slot, swap. Keep this only as a stepping stone; on its own it carries the alloc-merge hazard above.

## Timing diagram (2-deep)

| Time | MTE Copy | Cube Compute |
|------|----------|-------------|
| t₀ | copy tile 0 → slot 0 | |
| t₁ | copy tile 1 → slot 1 | dot tile 0 |
| t₂ | copy tile 2 → slot 0 | dot tile 1 |
| t₃ | | dot tile 2 |

## What To Verify After Applying

- UB/L1 budget re-checked with the doubling in effect (and ×2 again if `multibuffer=True` applies to these buffers).
- dot operands avoid the subview→trans chain.
- Numeric gate first, then measure: prefetch without correctness is not progress.
- Run the sync linter when combined with manual `sync_block` flags.

## Related Patterns

- `explicit-memory`: the alloc/subview machinery and the merge hazard.
- `cv-sync`: flag discipline if you hand-synchronize the prefetch.
- `workspace-pipeline`: cross-core ring that this per-core overlap complements.
