from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from enhanced_nagata_backend import EnhancedNagataBackend


FEATURE_CODE = {
    "FACE": 0,
    "EDGE": 1,
    "SHARPEDGE": 2,
    "VERTEX": 3,
}


def sample_points_in_block(block: Tuple[int, int, int], block_size: float, samples_per_axis: int) -> np.ndarray:
    b = np.asarray(block, dtype=float)
    origin = b * float(block_size)
    axis = (np.arange(samples_per_axis, dtype=float) + 0.5) / float(samples_per_axis)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    pts_local = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    return origin[None, :] + pts_local * float(block_size)


def generate_teacher_field(
    backend: EnhancedNagataBackend,
    *,
    tau: float,
    block_size: float,
    samples_per_axis: int,
    k_nearest: int,
    max_blocks: int = -1,
) -> Dict[str, np.ndarray]:
    active_blocks = backend.enumerate_active_blocks(tau=float(tau), block_size=float(block_size))
    if max_blocks > 0:
        active_blocks = active_blocks[: int(max_blocks)]

    points: List[np.ndarray] = []
    sdfs: List[float] = []
    unsigned_distances: List[float] = []
    nearest_points: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    tri_indices: List[int] = []
    uvs: List[Tuple[float, float]] = []
    feature_codes: List[int] = []
    block_coords: List[Tuple[int, int, int]] = []

    print(f"活跃块数: {len(active_blocks)}")
    print(f"每块采样: {samples_per_axis**3} 点")

    for bi, block in enumerate(active_blocks):
        pts = sample_points_in_block(block, block_size=block_size, samples_per_axis=samples_per_axis)
        for p in pts:
            q = backend.query_point(p, k_nearest=k_nearest)
            if abs(q.signed_distance) > float(tau):
                continue
            points.append(q.point)
            sdfs.append(q.signed_distance)
            unsigned_distances.append(q.distance)
            nearest_points.append(q.nearest_point)
            normals.append(q.normal)
            tri_indices.append(q.triangle_index)
            uvs.append(q.uv)
            feature_codes.append(FEATURE_CODE.get(q.feature_type, -1))
            block_coords.append(block)

        if (bi + 1) % max(1, min(100, len(active_blocks) // 10 if active_blocks else 1)) == 0:
            print(f"已处理块 {bi + 1}/{len(active_blocks)}，当前保留样本 {len(points)}")

    if points:
        arr_points = np.asarray(points, dtype=np.float64)
        arr_sdfs = np.asarray(sdfs, dtype=np.float64)
        arr_unsigned = np.asarray(unsigned_distances, dtype=np.float64)
        arr_nearest = np.asarray(nearest_points, dtype=np.float64)
        arr_normals = np.asarray(normals, dtype=np.float64)
        arr_tri = np.asarray(tri_indices, dtype=np.int32)
        arr_uv = np.asarray(uvs, dtype=np.float64)
        arr_feature = np.asarray(feature_codes, dtype=np.int8)
        arr_blocks = np.asarray(block_coords, dtype=np.int32)
    else:
        arr_points = np.zeros((0, 3), dtype=np.float64)
        arr_sdfs = np.zeros((0,), dtype=np.float64)
        arr_unsigned = np.zeros((0,), dtype=np.float64)
        arr_nearest = np.zeros((0, 3), dtype=np.float64)
        arr_normals = np.zeros((0, 3), dtype=np.float64)
        arr_tri = np.zeros((0,), dtype=np.int32)
        arr_uv = np.zeros((0, 2), dtype=np.float64)
        arr_feature = np.zeros((0,), dtype=np.int8)
        arr_blocks = np.zeros((0, 3), dtype=np.int32)

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
        "num_samples": int(arr_points.shape[0]),
        "feature_code_map": FEATURE_CODE,
    }

    return {
        "points": arr_points,
        "sdf": arr_sdfs,
        "unsigned_distance": arr_unsigned,
        "nearest_points": arr_nearest,
        "normals": arr_normals,
        "triangle_index": arr_tri,
        "uv": arr_uv,
        "feature_code": arr_feature,
        "block_coord": arr_blocks,
        "active_blocks": np.asarray(active_blocks, dtype=np.int32) if active_blocks else np.zeros((0, 3), dtype=np.int32),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="基于增强 Nagata 曲面生成稀疏窄带教师场")
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
    args = parser.parse_args()

    backend = EnhancedNagataBackend(
        args.nsm,
        use_cache=not args.no_cache,
        force_recompute=args.force_recompute,
        bake_cache=args.bake_cache,
        gap_threshold=args.gap_threshold,
        k_factor=args.k_factor,
    )

    print("后端构建完成:")
    print(f"  顶点数: {backend.build_info.num_vertices}")
    print(f"  三角形数: {backend.build_info.num_triangles}")
    print(f"  裂隙边数: {backend.build_info.num_crease_edges}")
    print(f"  使用缓存: {backend.build_info.used_cache}")
    print(f"  ENG 路径: {backend.build_info.eng_path}")

    data = generate_teacher_field(
        backend,
        tau=args.tau,
        block_size=args.block_size,
        samples_per_axis=args.samples_per_axis,
        k_nearest=args.k_nearest,
        max_blocks=args.max_blocks,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"教师场已写出: {out}")
    print(f"样本数: {data['points'].shape[0]}")


if __name__ == "__main__":
    main()
