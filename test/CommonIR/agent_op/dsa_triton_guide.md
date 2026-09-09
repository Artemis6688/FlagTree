# 昇腾 DSA Triton 算子修改点总结

参考算子
FlagGems/src/flag_gems/runtime/backend/_ascend/fused/fused_add_rms_norm.py
FlagGems/src/flag_gems/fused/swiglu.py
vllm-ascend/vllm_ascend/ops/causal_conv1d.py

---

## 一、总体共性修改模式

### 1. 导入与依赖变化

| 项目 | 原始版本 | DSA 版本 |
|------|---------|----------|
| sigmoid/exp 等数学函数 | `tl.sigmoid` 或 `tl_extra_shim.exp` | `triton.language.math.exp`（直接用 `math.exp`） |
| DSA 扩展 | 无 | `import triton.experimental.tle as tle`，使用 `tle.dsa.*` |
| 装饰器 | `@libentry()` 等 FlagGems GPU 工具 | 去掉 `@libentry()`，仅保留 `@triton.jit` |
| 设备 API | `torch.cuda` 或 `torch_device_fn` | `torch.npu`（昇腾 NPU 设备接口） |

### 2. 核心数（Core）感知的网格调度

DSA 版本普遍新增了 **NPU 向量核心数感知** 的调度逻辑：

```python
_CACHED_CORE_NUM = None

def _get_core_num():
    global _CACHED_CORE_NUM
    if _CACHED_CORE_NUM is None:
        try:
            current_device = torch.npu.current_device()
            torch.npu.set_device(current_device)
            cores_dict = torch.npu.get_device_limit(current_device)
            _CACHED_CORE_NUM = cores_dict["vector_core_num"] or 24
        except (AttributeError, KeyError, TypeError):
            _CACHED_CORE_NUM = 24
    return _CACHED_CORE_NUM
```

- 原始版本：grid 按数据分块大小计算，如 `grid = (cdiv(M, BLOCK_M), cdiv(H, BLOCK_H))`，不关心硬件核心数。
- DSA 版本：grid 大小 = `num_cores`（向量核心数），每个 program 在核心内循环处理多行/多块数据（**核心级分配 + 核内循环**）。

### 3. Tiling 策略从 2D grid 变为 1D grid + 核内循环

**原始版本**（GPU 风格）：
- 2D grid：`(cdiv(M, BLOCK_M), cdiv(H, BLOCK_H))`
- 每个 program 处理一个固定的 (M_tile, H_tile)

**DSA 版本**（昇腾风格）：
- 1D grid：`(num_cores,)` 或 `(num_programs_m, num_programs_n)`
- 每个 program 负责一个连续的行范围（BLOCK_SIZE_M），在 kernel 内部用 **双层循环** 遍历行和列：
  ```python
  for tile_m_idx in range(0, BLOCK_SIZE_M, TILE_SIZE_M):
      for tile_h_idx in range(0, H, TILE_SIZE_H):
          # 处理一个 (TILE_SIZE_M, TILE_SIZE_H) 的小块
  ```

### 4. Tile 大小动态计算

DSA 版本的 tile 大小根据实际数据量动态调整，而非使用固定常量：

```python
TILE_SIZE_M = min(triton.next_power_of_2(M), 32)
TILE_SIZE_H = min(triton.next_power_of_2(H), 256)

# 小数据量时只用 1 个核心
if M * H < 256 * 64:
    num_cores = 1

# BLOCK_SIZE_M 按 TILE_SIZE_M 的整数倍向上取整
num_tiles_m = triton.cdiv(M, TILE_SIZE_M)
num_cores = min(num_cores, num_tiles_m)
BLOCK_SIZE_M = triton.cdiv(num_tiles_m, num_cores) * TILE_SIZE_M
```

### 5. DSA 专用写回优化：`tle.dsa.copy` + `tle.dsa.to_buffer`

DSA 版本在输出时区分 **完整 tile** 和 **边界 tile**：

```python
if (m_idx + TILE_SIZE_M <= M) and (tile_h_idx + TILE_SIZE_H <= H):
    # 完整 tile：使用 DSA copy 写回（无 mask，更高效）
    out_buf = tle.dsa.to_buffer(out, space=tle.dsa.ascend.UB)
    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(out_buf, output_ptr + output_offset, [TILE_SIZE_M, TILE_SIZE_H])
else:
    # 边界 tile：退回 tl.store + mask
    tl.store(output_ptr + output_offset, out, mask=mask)
```

- `tle.dsa.to_buffer(tensor, space=tle.dsa.ascend.UB)`：将数据放入昇腾 Unified Buffer
- `tle.dsa.copy(src_buf, dst_ptr, shape)`：整块搬运，不需要 mask，吞吐更高
- `tle.dsa.hint(inter_no_alias=True)`：提示编译器输入输出无别名，允许更激进优化

### 6. DSA 专用计算优化：`tle.dsa.extract_slice`

causal_conv1d 系列算子中，权重向量的分片取用使用 `tle.dsa.extract_slice`：

```python
w_tile_2d = tl.load(w_ptrs_2d, mask=mask_w_2d, other=0.0).to(tl.float32)
w_tile = tl.reshape(w_tile_2d, (KERNEL_WIDTH * BLOCK_N,))

# 从 1D 权重向量按需取出对应 kernel 位置的切片
matrix_w = tle.dsa.extract_slice(
    w_tile, offsets=(j * BLOCK_N,), sizes=(BLOCK_N,), strides=(1,)
)
```

这是因为昇腾硬件上 **预加载权重到寄存器/UB** + **extract_slice 切片** 比逐次 load 更高效。

### 7. Kernel 启动参数新增昇腾专用选项

DSA 版本 kernel 调用时新增：

```python
kernel[(grid,)](
    ...,
    multibuffer=True,                                    # 启用多缓冲流水
    limit_auto_multi_buffer_of_local_buffer="no-limit",  # 不限制本地缓冲的多缓冲数
    limit_auto_multi_buffer_only_for_local_buffer=False,  # 多缓冲不仅限于本地缓冲
)
```

这些参数控制昇腾编译器的 **数据搬运流水线**，让 load 和 compute 可以并行执行。

### 8. 循环结构偏好

| 原始版本 | DSA 版本 |
|---------|----------|
| `for off in range(...)` | `for off in tl.range(...)` 或 `tl.static_range(...)` |
| Python range，Triton 编译器自动展开 | `tl.range` 让编译器对流水线优化更好控制；`tl.static_range` 强制编译时展开 |

### 9. 去掉反向传播 kernel

DSA 版本通常只保留前向 kernel，去掉 `dswiglu_kernel` 等反向传播实现（推理场景不需要）。

### 10. 输入预处理差异

- 原始版本：有时将 input_a 和 input_b 从同一个 tensor 中通过指针偏移读取
- DSA 版本：倾向于先用 `torch.split` 在 Python 侧拆分为两个独立 tensor 再传入 kernel，简化 kernel 内的地址计算

---

## 二、各算子具体修改对照

### 算子 1：swiglu

| 维度 | 原始版本 | DSA 版本 |
|------|---------|----------|
| 文件 | `flag_gems/fused/swiglu.py` | `flaggems_vllm/.../ops/swiglu.py` |
| Grid | 2D: `(cdiv(M, 64), cdiv(H, 64))` | 1D: `(num_cores,)` |
| 分块 | 固定 `BLOCK_SIZE_M=64, BLOCK_SIZE_H=64` | 动态 `TILE_SIZE_M`, `TILE_SIZE_H`，根据数据量调整 |
| 输入方式 | kernel 接收完整 input_ptr，内部通过偏移读取 a 和 b | Python 侧 `torch.split` 后传入 `input_a_ptr`, `input_b_ptr` |
| sigmoid 实现 | `tl.sigmoid(x_a)` | 手动 `1.0 / (1.0 + math.exp(-x_a_f))` |
| 写回 | 统一 `tl.store` | 完整 tile 用 `tle.dsa.copy`，边界用 `tl.store` |
| 反向 kernel | 有 `dswiglu_kernel` | 无（仅推理） |
| CUDA 检查 | `if not input_tensor.is_cuda: raise` | 无（NPU 设备） |
| 核心数 | 不涉及 | `_get_core_num()` 动态获取 |
| 启动参数 | 无额外参数 | `multibuffer=True`, `limit_auto_multi_buffer_of_local_buffer="no-limit"` |

### 算子 2：add_rms_norm

| 维度 | 原始版本 | DSA 版本 |
|------|---------|----------|
| 文件 | `flag_gems/.../fused_add_rms_norm.py` | `flaggems_vllm/.../ops/add_rms_norm.py` |
| 语义 | **in-place**：写回 X 和 residual | **out-of-place**：输出到新 tensor `y` |
| Grid | `M` 个 programs（每行一个） | `USE_CORES` 个 programs（多行一核） |
| 多行处理 | 每个 program 处理 1 行 | 每个 program 循环处理 `N_ROWS` 行 |
| 大 N 处理 | 仅一个 kernel，`BLOCK_SIZE = min(next_power_of_2(N), 8192)` | 分两个 kernel：N ≤ 4096 单块 kernel；N > 4096 列循环 kernel |
| UB 溢出保护 | 无（GPU 不受此限） | `_BLOCK_N = 4096` 阈值，防止昇腾 Unified Buffer 溢出 |
| 权重预加载 | 每个循环迭代中 load weight | 单块 kernel 中 weight 在循环外一次性 preload |
| variance 计算 | 分块累加 `_var_base += x * x / N`，最后 `tl.sum` | 单块 kernel：先 `(x * x) / N` 再 `tl.sum`；列循环 kernel：每块 `var += tl.sum(x * x)`，最后 `var / N` |
| 装饰器 | `@libentry()` + `@triton.jit(do_not_specialize=["eps"])` | `@triton.jit(do_not_specialize=["eps"])` |
| 启动参数 | 无额外 | `multibuffer=True`, `limit_auto_multi_buffer_only_for_local_buffer=False`, `limit_auto_multi_buffer_of_local_buffer="no-limit"` |
| stride 参数 | 传入 `x_stride_r`, `x_stride_c`, `r_stride_r`, `r_stride_c` | 不传 stride，假设 contiguous（用 `N` 做偏移） |
| 核心数 | 不涉及 | `_get_core_num()` + `_get_rows_and_cores(M, CORES)` |

### 算子 3：causal_conv1d_fn

| 维度 | 原始版本 | DSA 版本 |
|------|---------|----------|
| 文件 | `vllm_ascend/ops/causal_conv1d.py` (纯 PyTorch) | `flaggems_vllm/.../ops/causal_conv1d_fn.py` (Triton kernel) |
| 实现方式 | **纯 PyTorch**：用 `F.conv1d` + 循环逐 sequence 处理 | **Triton kernel**：DSA 向量化实现 |
| 核心变化 | 完全重写，从 Python 循环变为硬件并行 kernel | — |
| 权重处理 | PyTorch 自动处理 | 预加载为 2D tile → reshape 为 1D → `tle.dsa.extract_slice` 按需取切片 |
| 卷积实现 | `F.conv1d(..., groups=dim)` | 手动滑窗：维护 `col0, col1, ...` 状态寄存器，逐 token 卷积 |
| conv_state 更新 | Python 侧 `F.pad` + tensor 操作 | kernel 内直接管理 conv_state 的读写和滚动 |
| APC 支持 | 循环中逐个处理，简单 if 分支 | kernel 编译期 `IS_APC_ENABLED` constexpr 分支，零开销 |
| Grid | 不涉及（纯 Python） | 2D: `(num_programs_seq * num_chunks, cdiv(dim, BLOCK_N))` |
| num_stages | 不涉及 | `num_stages=2`（流水线级数） |

### 算子 4：causal_conv1d_update

| 维度 | 原始版本 | DSA 版本 |
|------|---------|----------|
| 文件 | `vllm_ascend/ops/causal_conv1d.py` (纯 PyTorch) | `flaggems_vllm/.../ops/causal_conv1d_update.py` (Triton kernel) |
| 实现方式 | **纯 PyTorch**：`_run_one` 内用 `torch.roll` + 切片操作逐 sequence 更新 | **Triton kernel**：DSA 向量化 tiled 实现 |
| Tiling | 不涉及 | 3 维 tiling：`B_TILE`（batch）× `BLOCK_N`（channel）× `T_CHUNK`（token chunk） |
| 状态滚动 | `torch.roll(state, -seqlen, -1)` + 切片赋值 | kernel 内 `tle.dsa.copy` 批量搬运状态，分块滚动 |
| Spec decoding | Python `if num_accepted_tokens is not None` | `IS_SPEC_DECODING: tl.constexpr` 编译期分支 |
| 权重处理 | PyTorch 自动 | 预加载 2D → 1D → `tle.dsa.extract_slice` |
| batch 分组 | 外层 for 循环逐 sequence | `B_TILE` 个 sequence 在同一 program 内处理（减少 program 启动开销） |
| Grid | 不涉及 | 2D: `(cdiv(batch, B_TILE), cdiv(dim, BLOCK_N))` |

---

## 三、迁移 Checklist（供 AI 生成算子参考）

将一个标准 Triton（GPU）算子迁移为昇腾 DSA Triton 算子时，按以下步骤操作：

### 步骤 1：调整导入
- [ ] 添加 `import triton.experimental.tle as tle`
- [ ] 数学函数改用 `triton.language.math`（如 `math.exp`）
- [ ] 去掉 GPU 专用工具（`@libentry()`, `tl_extra_shim` 等）
- [ ] 设备接口 `torch.cuda` → `torch.npu`

### 步骤 2：添加核心数感知
- [ ] 实现 `_get_core_num()` 获取 NPU 向量核心数
- [ ] Grid 大小基于 `num_cores` 而非纯数据分块数

### 步骤 3：重构 Tiling 策略
- [ ] 2D grid → 1D grid（或降维 grid） + 核内循环
- [ ] Tile 大小动态计算（`triton.next_power_of_2`，设置上限避免 UB 溢出）
- [ ] 确保 `BLOCK_SIZE` 是 `TILE_SIZE` 的整数倍
- [ ] 小数据量退化为单核执行

### 步骤 4：使用 DSA 写回优化
- [ ] 完整 tile 使用 `tle.dsa.to_buffer` + `tle.dsa.copy`（无 mask，高效）
- [ ] 边界 tile 保留 `tl.store` + mask
- [ ] 如需要，添加 `tle.dsa.hint(inter_no_alias=True)`

### 步骤 5：使用 DSA 计算优化
- [ ] 权重等需要复用的数据：预加载到寄存器/UB
- [ ] 使用 `tle.dsa.extract_slice` 从预加载的 1D 向量中取切片
- [ ] 状态搬运使用 `tle.dsa.copy` 替代逐元素操作

### 步骤 6：Kernel 启动参数
- [ ] 添加 `multibuffer=True`
- [ ] 添加 `limit_auto_multi_buffer_of_local_buffer="no-limit"`
- [ ] 视情况添加 `limit_auto_multi_buffer_only_for_local_buffer=False`
- [ ] 复杂 kernel 可设置 `num_stages=2`

### 步骤 7：循环与分支
- [ ] Python `range` → `tl.range` 或 `tl.static_range`（编译器更好优化流水线）
- [ ] 运行时条件分支 → `tl.constexpr` 编译期分支（零开销）

### 步骤 8：简化与清理
- [ ] 去掉反向传播 kernel（如仅推理场景）
- [ ] 输入尽量在 Python 侧预处理（split、contiguous），简化 kernel 地址计算
- [ ] contiguous tensor 可省略 stride 参数，直接用维度大小做偏移
- [ ] 去掉 CUDA 专用检查

### 步骤 9：注意昇腾硬件限制
- [ ] Unified Buffer 容量有限，单块不宜过大（参考阈值 `_BLOCK_N = 4096`）
- [ ] M 维度上限 65535（程序数限制）
- [ ] `tle.dsa.copy` 不支持 mask，只能用于完整 tile
