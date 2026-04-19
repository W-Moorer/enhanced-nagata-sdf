from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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

_WORKER_BACKEND = None


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
    """
    返回和原输出兼容的空数组字典，不含 metadata_json 和 active_blocks。
    
    返回:
        包含所有必需字段的空结果字典
    """
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


def _concat_result_dicts(parts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    把多个 worker 返回的结果拼起来。
    
    参数:
        parts: worker 返回的结果列表
    
    返回:
        拼接后的完整结果字典
    """
    if not parts:
        return _empty_result_dict()
    out = {}
    for key in parts[0].keys():
        out[key] = np.concatenate([p[key] for p in parts], axis=0) if parts else _empty_result_dict()[key]
    return out


def _split_blocks(active_blocks: List[Tuple[int, int, int]], chunk_blocks: int) -> List[List[Tuple[int, int, int]]]:
    """
    把 active_blocks 切成多个 list chunk。
    
    参数:
        active_blocks: 活跃块列表
        chunk_blocks: 每个 chunk 包含的块数
    
    返回:
        chunk 列表
    """
    chunk_blocks = max(1, int(chunk_blocks))
    return [active_blocks[i:i + chunk_blocks] for i in range(0, len(active_blocks), chunk_blocks)]


def _init_worker_backend(
    nsm_path: str,
    use_cache: bool,
    force_recompute: bool,
    bake_cache: bool,
    gap_threshold: float,
    k_factor: float,
) -> None:
    """
    在 worker 中构造全局 _WORKER_BACKEND。
    
    参数:
        nsm_path: NSM 文件路径
        use_cache: 是否使用缓存
        force_recompute: 是否强制重算
        bake_cache: 是否写回缓存
        gap_threshold: 裂隙边检测阈值
        k_factor: 增强曲面高斯衰减参数
    """
    global _WORKER_BACKEND
    from enhanced_nagata_backend import EnhancedNagataBackend
    _WORKER_BACKEND = EnhancedNagataBackend(
        nsm_path,
        use_cache=use_cache,
        force_recompute=force_recompute,
        bake_cache=bake_cache,
        gap_threshold=gap_threshold,
        k_factor=k_factor,
    )


def _process_block_chunk(args: tuple) -> Dict[str, np.ndarray]:
    """
    处理一个 block chunk，返回局部结果字典。
    
    参数:
        args: 元组 (block_chunk, tau, block_size, samples_per_axis, k_nearest)
    
    返回:
        局部结果字典，包含所有必需字段
    """
    global _WORKER_BACKEND
    block_chunk, tau, block_size, samples_per_axis, k_nearest = args

    points = []
    sdfs = []
    unsigned_distances = []
    nearest_points = []
    normals = []
    tri_indices = []
    uvs = []
    feature_codes = []
    block_coords = []

    for block in block_chunk:
        pts = sample_points_in_block(block, block_size=block_size, samples_per_axis=samples_per_axis)
        for p in pts:
            q = _WORKER_BACKEND.query_point(p, k_nearest=k_nearest)
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

    if not points:
        return _empty_result_dict()

    return {
        "points": np.asarray(points, dtype=np.float64),
        "sdf": np.asarray(sdfs, dtype=np.float64),
        "unsigned_distance": np.asarray(unsigned_distances, dtype=np.float64),
        "nearest_points": np.asarray(nearest_points, dtype=np.float64),
        "normals": np.asarray(normals, dtype=np.float64),
        "triangle_index": np.asarray(tri_indices, dtype=np.int32),
        "uv": np.asarray(uvs, dtype=np.float64),
        "feature_code": np.asarray(feature_codes, dtype=np.int8),
        "block_coord": np.asarray(block_coords, dtype=np.int32),
    }


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


def generate_teacher_field_parallel(
    backend: EnhancedNagataBackend,
    *,
    tau: float,
    block_size: float,
    samples_per_axis: int,
    k_nearest: int,
    max_blocks: int = -1,
    workers: int = 4,
    chunk_blocks: int = 32,
) -> Dict[str, np.ndarray]:
    """
    并行版本的教师场生成函数。
    
    参数:
        backend: 增强 Nagata 后端实例（仅用于获取信息和枚举活跃块）
        tau: 窄带半宽
        block_size: 背景块边长
        samples_per_axis: 每个块每轴采样数
        k_nearest: 查询时候选 patch 数
        max_blocks: 调试时限制处理块数，-1 表示不限制
        workers: CPU 并行进程数
        chunk_blocks: 每个任务包含的 block 数
    
    返回:
        包含教师场数据的字典，与原 generate_teacher_field 格式一致
    """
    active_blocks = backend.enumerate_active_blocks(tau=float(tau), block_size=float(block_size))
    if max_blocks > 0:
        active_blocks = active_blocks[: int(max_blocks)]

    chunks = _split_blocks(active_blocks, chunk_blocks)
    print(f"活跃块数: {len(active_blocks)}")
    print(f"每块采样: {samples_per_axis**3} 点")
    print(f"并行进程数: {workers}")
    print(f"chunk 数量: {len(chunks)}")

    parts: List[Dict[str, np.ndarray]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker_backend,
        initargs=(
            backend.nsm_path,
            True,   # use_cache
            False,  # force_recompute
            False,  # bake_cache
            backend.gap_threshold,
            backend.k_factor,
        ),
    ) as ex:
        futures = [
            ex.submit(_process_block_chunk, (chunk, tau, block_size, samples_per_axis, k_nearest))
            for chunk in chunks
        ]
        for i, fut in enumerate(as_completed(futures), start=1):
            part = fut.result()
            parts.append(part)
            current = sum(p["points"].shape[0] for p in parts)
            print(f"已完成 chunk {i}/{len(futures)}，累计样本 {current}")

    merged = _concat_result_dicts(parts)

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
        "num_samples": int(merged["points"].shape[0]),
        "feature_code_map": FEATURE_CODE,
        "workers": int(workers),
        "chunk_blocks": int(chunk_blocks),
    }

    merged["active_blocks"] = np.asarray(active_blocks, dtype=np.int32) if active_blocks else np.zeros((0, 3), dtype=np.int32)
    merged["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    return merged


def main() -> None:
    """
    主函数，支持单进程和并行两种模式。
    
    当 workers <= 1 时使用单进程版本，否则使用并行版本。
    """
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
    parser.add_argument("--workers", type=int, default=1, help="CPU 并行进程数，1 表示单进程")
    parser.add_argument("--chunk-blocks", type=int, default=32, help="每个任务包含的 block 数")
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

    workers = min(args.workers, os.cpu_count() or 1)
    chunk_blocks = max(1, args.chunk_blocks)

    if workers <= 1:
        print("使用单进程模式")
        data = generate_teacher_field(
            backend,
            tau=args.tau,
            block_size=args.block_size,
            samples_per_axis=args.samples_per_axis,
            k_nearest=args.k_nearest,
            max_blocks=args.max_blocks,
        )
    else:
        print(f"使用并行模式，workers={workers}, chunk_blocks={chunk_blocks}")
        data = generate_teacher_field_parallel(
            backend,
            tau=args.tau,
            block_size=args.block_size,
            samples_per_axis=args.samples_per_axis,
            k_nearest=args.k_nearest,
            max_blocks=args.max_blocks,
            workers=workers,
            chunk_blocks=chunk_blocks,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(f"教师场已写出: {out}")
    print(f"样本数: {data['points'].shape[0]}")


if __name__ == "__main__":
    main()
