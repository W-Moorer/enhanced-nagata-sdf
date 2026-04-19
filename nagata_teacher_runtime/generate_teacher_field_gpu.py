"""
GPU 版教师场生成入口

与 CPU 版 generate_teacher_field.py 接口一致，使用 GPU 后端进行查询。

设计:
- CPU 负责: 读取 NSM、加载 ENG 缓存、枚举活跃块、生成采样点
- GPU 负责: 批量查询最近点和 SDF
- 输出格式: 与 CPU 版完全兼容
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import torch

FEATURE_CODE = {
    "FACE": 0,
    "EDGE": 1,
    "SHARPEDGE": 2,
    "VERTEX": 3,
}


def sample_points_in_block(block: Tuple[int, int, int], block_size: float, samples_per_axis: int) -> np.ndarray:
    """
    在指定 block 内生成采样点。

    参数:
        block: 块的整数坐标 (ix, iy, iz)
        block_size: 块的边长
        samples_per_axis: 每个轴上的采样点数

    返回:
        采样点数组，shape 为 (samples_per_axis^3, 3)
    """
    b = np.asarray(block, dtype=float)
    origin = b * float(block_size)
    axis = (np.arange(samples_per_axis, dtype=float) + 0.5) / float(samples_per_axis)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    pts_local = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    return origin[None, :] + pts_local * float(block_size)


def _empty_result_dict() -> Dict[str, np.ndarray]:
    """返回和原输出兼容的空数组字典。"""
    return {
        "points": np.zeros((0, 3), dtype=np.float64),
        "sdf": np.zeros((0,), dtype=np.float64),
        "unsigned_distance": np.zeros((0,), dtype=np.float64),
        "nearest_points": np.zeros((0, 3), dtype=np.float64),
        "normals": np.zeros((0, 3), dtype=np.float64),
        "triangle_index": np.zeros((0,), dtype=np.int32),
        "uv": np.zeros((0, 2), dtype=np.float64),
        "feature_code": np.zeros((0,), dtype=np.int8),
        "block_coord": np.zeros((0, 3), dtype=np.int32),
    }


def generate_teacher_field_gpu(
    backend,
    *,
    tau: float,
    block_size: float,
    samples_per_axis: int,
    k_nearest: int,
    max_blocks: int = -1,
    gpu_batch_size: int = 4096,
) -> Dict[str, np.ndarray]:
    """
    GPU 版教师场生成函数。

    CPU 负责:
    - 枚举活跃块
    - 生成采样点
    - tau 过滤

    GPU 负责:
    - 批量查询最近点和 SDF

    参数:
        backend: EnhancedNagataBackendTorch 实例
        tau: 窄带半宽
        block_size: 背景块边长
        samples_per_axis: 每个块每轴采样数
        k_nearest: 查询时候选 patch 数
        max_blocks: 调试时限制处理块数，-1 表示不限制
        gpu_batch_size: GPU 批次大小

    返回:
        包含教师场数据的字典，与 CPU 版格式一致
    """
    active_blocks = backend.enumerate_active_blocks(tau=float(tau), block_size=float(block_size))
    if max_blocks > 0:
        active_blocks = active_blocks[: int(max_blocks)]

    # CPU: 生成所有采样点
    all_points: List[np.ndarray] = []
    all_block_coords: List[Tuple[int, int, int]] = []

    samples_per_block = samples_per_axis ** 3
    print(f"活跃块数: {len(active_blocks)}")
    print(f"每块采样: {samples_per_block} 点")
    print(f"总采样点预估: {len(active_blocks) * samples_per_block}")

    for block in active_blocks:
        pts = sample_points_in_block(block, block_size=block_size, samples_per_axis=samples_per_axis)
        all_points.append(pts)
        all_block_coords.extend([block] * pts.shape[0])

    if not all_points:
        result = _empty_result_dict()
        metadata = {
            "nsm_path": backend.build_info.nsm_path,
            "eng_path": backend.build_info.eng_path,
            "num_vertices": backend.build_info.num_vertices,
            "num_triangles": backend.build_info.num_triangles,
            "num_crease_edges": backend.build_info.num_crease_edges,
            "used_cache": backend.build_info.used_cache,
            "tau": float(tau),
            "block_size": float(block_size),
            "samples_per_axis": int(samples_per_axis),
            "k_nearest": int(k_nearest),
            "num_active_blocks": int(len(active_blocks)),
            "num_samples": 0,
            "feature_code_map": FEATURE_CODE,
        }
        result["active_blocks"] = np.asarray(active_blocks, dtype=np.int32) if active_blocks else np.zeros((0, 3), dtype=np.int32)
        result["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
        return result

    all_points_arr = np.concatenate(all_points, axis=0)
    all_block_coords_arr = np.array(all_block_coords, dtype=np.int32)
    print(f"总采样点数: {all_points_arr.shape[0]}")

    # GPU: 批量查询
    t_gpu_start = time.perf_counter()
    gpu_result = backend.query_points_gpu(
        all_points_arr,
        k_nearest=int(k_nearest),
        batch_size=int(gpu_batch_size),
    )
    gpu_time = time.perf_counter() - t_gpu_start
    print(f"GPU 查询耗时: {gpu_time:.2f} 秒")

    # CPU: tau 过滤
    tau_val = float(tau)
    mask = np.abs(gpu_result["sdf"]) <= tau_val

    points = gpu_result["points"][mask]
    sdf = gpu_result["sdf"][mask]
    unsigned_distance = gpu_result["unsigned_distance"][mask]
    nearest_points = gpu_result["nearest_points"][mask]
    normals = gpu_result["normals"][mask]
    triangle_index = gpu_result["triangle_index"][mask]
    uv = gpu_result["uv"][mask]
    feature_code = gpu_result["feature_code"][mask]
    block_coord = all_block_coords_arr[mask]

    n_retained = points.shape[0]
    print(f"tau 过滤后保留样本: {n_retained}")

    metadata = {
        "nsm_path": backend.build_info.nsm_path,
        "eng_path": backend.build_info.eng_path,
        "num_vertices": backend.build_info.num_vertices,
        "num_triangles": backend.build_info.num_triangles,
        "num_crease_edges": backend.build_info.num_crease_edges,
        "used_cache": backend.build_info.used_cache,
        "tau": float(tau),
        "block_size": float(block_size),
        "samples_per_axis": int(samples_per_axis),
        "k_nearest": int(k_nearest),
        "num_active_blocks": int(len(active_blocks)),
        "num_samples": int(n_retained),
        "feature_code_map": FEATURE_CODE,
        "gpu_time": gpu_time,
        "gpu_device": str(backend.device),
    }

    return {
        "points": points,
        "sdf": sdf,
        "unsigned_distance": unsigned_distance,
        "nearest_points": nearest_points,
        "normals": normals,
        "triangle_index": triangle_index,
        "uv": uv,
        "feature_code": feature_code,
        "block_coord": block_coord,
        "active_blocks": np.asarray(active_blocks, dtype=np.int32) if active_blocks else np.zeros((0, 3), dtype=np.int32),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }


def main() -> None:
    """
    主函数，GPU 版教师场生成入口。
    """
    parser = argparse.ArgumentParser(description="基于 GPU 增强 Nagata 曲面生成稀疏窄带教师场")
    parser.add_argument("nsm", type=str, help="输入 .nsm 文件路径")
    parser.add_argument("output", type=str, help="输出 .npz 文件路径")
    parser.add_argument("--tau", type=float, default=0.02, help="窄带半宽")
    parser.add_argument("--block-size", type=float, default=0.02, help="背景块边长")
    parser.add_argument("--samples-per-axis", type=int, default=4, help="每个块每轴采样数")
    parser.add_argument("--k-nearest", type=int, default=16, help="查询时候选 patch 数")
    parser.add_argument("--gap-threshold", type=float, default=1e-4, help="裂隙边检测阈值")
    parser.add_argument("--k-factor", type=float, default=0.0, help="增强曲面高斯衰减参数")
    parser.add_argument("--no-cache", action="store_true", help="不读取 .eng 缓存")
    parser.add_argument("--force-recompute", action="store_true", help="强制重算 c_sharps")
    parser.add_argument("--bake-cache", action="store_true", help="将重算后的 c_sharps 写回 .eng")
    parser.add_argument("--max-blocks", type=int, default=-1, help="调试时限制处理块数")
    parser.add_argument("--gpu-batch-size", type=int, default=4096, help="GPU 批次大小")
    parser.add_argument("--device", type=str, default=None, help="GPU 设备 (e.g. cuda:0)")
    args = parser.parse_args()

    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("错误: CUDA 不可用，无法运行 GPU 版教师场生成。")
        print(f"PyTorch 版本: {torch.__version__}")
        sys.exit(1)

    print(f"CUDA 可用: {torch.cuda.is_available()}")
    print(f"CUDA 设备数: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.get_device_name(0)}")

    # 导入 GPU 后端
    from enhanced_nagata_backend_torch import EnhancedNagataBackendTorch

    backend = EnhancedNagataBackendTorch(
        args.nsm,
        use_cache=not args.no_cache,
        force_recompute=args.force_recompute,
        bake_cache=args.bake_cache,
        gap_threshold=args.gap_threshold,
        k_factor=args.k_factor,
        device=args.device,
    )

    print("GPU 后端构建完成:")
    print(f"  顶点数: {backend.build_info.num_vertices}")
    print(f"  三角形数: {backend.build_info.num_triangles}")
    print(f"  裂隙边数: {backend.build_info.num_crease_edges}")
    print(f"  使用缓存: {backend.build_info.used_cache}")
    print(f"  ENG 路径: {backend.build_info.eng_path}")

    data = generate_teacher_field_gpu(
        backend,
        tau=args.tau,
        block_size=args.block_size,
        samples_per_axis=args.samples_per_axis,
        k_nearest=args.k_nearest,
        max_blocks=args.max_blocks,
        gpu_batch_size=args.gpu_batch_size,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"教师场已写出: {out}")
    print(f"样本数: {data['points'].shape[0]}")


if __name__ == "__main__":
    main()
