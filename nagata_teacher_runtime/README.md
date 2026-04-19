# Enhanced Nagata Teacher Runtime

这套脚本直接基于你现有的四个文件：

- `nagata_patch.py`
- `visualize_nagata.py`
- `nagata_storage.py`
- `nsm_reader.py`

但为了避免教师场生成依赖 `PyVista`，我把 `visualize_nagata.py` 里的增强预处理关键逻辑抽到了 `enhanced_nagata_backend.py` 中。

## 文件说明

### 1. `enhanced_nagata_backend.py`
负责：
- 读取 `.nsm`
- 加载/重算 `.eng`
- 检测裂隙边
- 计算增强 `c_sharps`
- 构造 `NagataModelQuery`
- 提供 patch 求值与最近点查询接口
- 估计活跃块

### 2. `generate_teacher_field.py`
负责：
- 调用增强 Nagata 后端
- 构造窄带活跃块
- 在每个块内规则采样
- 对采样点做最近点查询
- 生成带符号距离并导出 `.npz`

## 最小使用方法

```bash
python generate_teacher_field.py input_model.nsm output_teacher.npz \
  --tau 0.02 \
  --block-size 0.02 \
  --samples-per-axis 4 \
  --k-nearest 16 \
  --bake-cache
```

## 输出字段

- `points`: 采样点坐标 `(N,3)`
- `sdf`: 带符号距离 `(N,)`
- `unsigned_distance`: 无符号距离 `(N,)`
- `nearest_points`: 最近点 `(N,3)`
- `normals`: 最近点法向 `(N,3)`
- `triangle_index`: 对应 patch id `(N,)`
- `uv`: 最近点参数 `(N,2)`
- `feature_code`: 最近特征类型编码 `(N,)`
- `block_coord`: 所属活跃块坐标 `(N,3)`
- `active_blocks`: 所有活跃块坐标 `(M,3)`
- `metadata_json`: 元数据 JSON

## 说明

当前脚本的符号采用：

```text
sign = sign((x - x_nearest) · n_nearest)
```

这适合先生成第一版窄带教师。若你后续需要更稳的全局 inside/outside，可以再接入 winding number 或 flood fill。
