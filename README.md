# Enhanced Nagata SDF

基于增强 Nagata 曲面三角形，直接构建**可查询的稀疏窄带符号距离场（Sparse Narrow-Band SDF）**。

这个清理版项目只保留三条主线：

1. **增强 Nagata 几何后端**
   - 裂隙边检测
   - 共享边界系数 `c_sharp`
   - 增强曲面求值与最近点查询
2. **稀疏窄带 SDF 构建**
   - 活跃块枚举
   - 局部规则节点格
   - 截断窄带值写出
3. **可直接查询的运行时**
   - 读取 `.npz`
   - 三线性插值查询

项目已经彻底收敛为“增强几何真值生成 + 稀疏窄带 SDF 构建 + 可查询运行时”这一条主线。

## 目录结构

```text
enhanced-nagata-sdf/
├── enhanced_nagata_sdf/
│   ├── enhanced_nagata_backend.py
│   ├── enhanced_nagata_backend_torch.py
│   ├── nagata_patch.py
│   ├── nagata_storage.py
│   ├── nsm_reader.py
│   ├── sparse_narrowband_sdf.py
│   └── visualize_nagata.py
├── scripts/
│   ├── build_sparse_narrowband_sdf.py
│   └── query_sparse_narrowband_sdf.py
├── examples/
│   ├── build_and_query.py
│   └── verify_sphere_sdf.py
├── tests/
├── models/
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
```

可选：

```bash
pip install pyvista matplotlib   # 可视化
pip install torch                # GPU 加速后端
```

## 快速开始

### 1) 构建稀疏窄带 SDF

```bash
python scripts/build_sparse_narrowband_sdf.py   models/sphere.nsm   outputs/sphere_sdf.npz   --tau 0.02   --block-size 0.02   --block-resolution 8   --bake-cache
```

### 2) 查询 SDF

```bash
python scripts/query_sparse_narrowband_sdf.py   outputs/sphere_sdf.npz   --point 0.1 0.2 0.3
```

### 3) Python API

```python
from enhanced_nagata_sdf import EnhancedNagataBackend, SparseNarrowbandBuildConfig, build_sparse_narrowband_sdf

backend = EnhancedNagataBackend("models/sphere.nsm", bake_cache=True)
config = SparseNarrowbandBuildConfig(tau=0.02, block_size=0.02, block_resolution=8)
sdf, metadata = build_sparse_narrowband_sdf(backend, config)
phi = sdf.query([0.1, 0.2, 0.3])
print(phi)
```

## 示例与验证

- `examples/build_and_query.py`：最小构建与查询示例
- `examples/verify_sphere_sdf.py`：与解析球 SDF 做误差对比的基础验证算例
- `enhanced_nagata_sdf/visualize_nagata.py`：增强 Nagata 曲面可视化辅助工具

## 最终产物

最终产物是：

- **`SparseNarrowbandSDF`**：可直接查询的窄带场
- 输出格式：`*_sdf.npz`

它不是样本表，也不依赖额外拟合器，而是可直接查询的最终场表示。
