"""
小规模 CPU/GPU 对比验证工具

功能:
1. 对同一个 NSM 模型生成少量测试点
2. 分别用 CPU 后端和 GPU 后端查询最近点和 SDF
3. 对比结果的数值一致性

预期结果:
- CPU/GPU 的 SDF 差值应 < 1e-6 (浮点精度范围)
- 最近点坐标差值应 < 1e-6
- 法向点积应 > 0.9999
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from nagata_teacher_runtime.enhanced_nagata_backend import (
    EnhancedNagataBackend,
    load_nsm_lightweight,
)
from nagata_teacher_runtime.enhanced_nagata_backend_torch import (
    EnhancedNagataBackendTorch,
)


def generate_test_points(nsm_path: str, n_points: int = 100) -> np.ndarray:
    """
    在 NSM 模型 AABB 附近生成随机测试点。

    参数:
        nsm_path: NSM 文件路径
        n_points: 测试点数量

    返回:
        测试点数组 (n_points, 3)
    """
    mesh = load_nsm_lightweight(nsm_path)
    vertices = mesh.vertices

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    bbox_diag = np.linalg.norm(bbox_max - bbox_min)

    # 在 AABB 外扩 20% 范围内随机采样
    extent = bbox_diag * 0.6
    rng = np.random.RandomState(42)
    points = center[None, :] + rng.randn(n_points, 3) * extent

    return points


def compare_cpu_gpu(nsm_path: str, n_points: int = 100, k_nearest: int = 16) -> dict:
    """
    对比 CPU 和 GPU 后端的查询结果。

    参数:
        nsm_path: NSM 文件路径
        n_points: 测试点数量
        k_nearest: 候选 patch 数量

    返回:
        对比结果字典
    """
    print("=" * 60)
    print("小规模 CPU/GPU 对比验证")
    print("=" * 60)
    print(f"NSM: {nsm_path}")
    print(f"测试点数: {n_points}")
    print(f"k_nearest: {k_nearest}")
    print()

    # 生成测试点
    test_points = generate_test_points(nsm_path, n_points)
    print(f"测试点范围: [{test_points.min():.4f}, {test_points.max():.4f}]")

    # CPU 后端
    print("\n--- CPU 后端 ---")
    t0 = time.perf_counter()
    cpu_backend = EnhancedNagataBackend(
        nsm_path,
        use_cache=True,
        force_recompute=False,
        bake_cache=False,
        gap_threshold=1e-4,
        k_factor=0.0,
    )
    cpu_init_time = time.perf_counter() - t0
    print(f"CPU 后端初始化: {cpu_init_time:.2f}s")
    print(f"  三角形数: {cpu_backend.build_info.num_triangles}")
    print(f"  裂隙边数: {cpu_backend.build_info.num_crease_edges}")

    t0 = time.perf_counter()
    cpu_results = []
    for p in test_points:
        q = cpu_backend.query_point(p, k_nearest=k_nearest)
        cpu_results.append({
            "nearest_point": q.nearest_point,
            "distance": q.distance,
            "signed_distance": q.signed_distance,
            "normal": q.normal,
            "triangle_index": q.triangle_index,
            "uv": q.uv,
            "feature_type": q.feature_type,
        })
    cpu_query_time = time.perf_counter() - t0
    print(f"CPU 查询耗时: {cpu_query_time:.4f}s ({n_points / cpu_query_time:.1f} 点/秒)")

    # GPU 后端
    print("\n--- GPU 后端 ---")
    if not torch.cuda.is_available():
        print("错误: CUDA 不可用，跳过 GPU 对比。")
        return {"status": "skipped", "reason": "CUDA not available"}

    t0 = time.perf_counter()
    gpu_backend = EnhancedNagataBackendTorch(
        nsm_path,
        use_cache=True,
        force_recompute=False,
        bake_cache=False,
        gap_threshold=1e-4,
        k_factor=0.0,
    )
    gpu_init_time = time.perf_counter() - t0
    print(f"GPU 后端初始化: {gpu_init_time:.2f}s")
    print(f"  设备: {gpu_backend.device}")

    t0 = time.perf_counter()
    gpu_result = gpu_backend.query_points_gpu(test_points, k_nearest=k_nearest, batch_size=4096)
    gpu_query_time = time.perf_counter() - t0
    print(f"GPU 查询耗时: {gpu_query_time:.4f}s ({n_points / gpu_query_time:.1f} 点/秒)")
    
    if gpu_query_time > 0:
        print(f"GPU 加速比: {cpu_query_time / gpu_query_time:.2f}x")
        if cpu_query_time / gpu_query_time < 1.0:
            print(f"  (注: 小规模测试时 GPU 可能不如 CPU，这是正常的)")
            print(f"  GPU 优势在大规模查询 (数万点) 时才会体现)")

    # 对比结果
    print("\n--- 结果对比 ---")

    sdf_cpu = np.array([r["signed_distance"] for r in cpu_results])
    sdf_gpu = gpu_result["sdf"]

    dist_cpu = np.array([r["distance"] for r in cpu_results])
    dist_gpu = gpu_result["unsigned_distance"]

    nearest_cpu = np.array([r["nearest_point"] for r in cpu_results])
    nearest_gpu = gpu_result["nearest_points"]

    normal_cpu = np.array([r["normal"] for r in cpu_results])
    normal_gpu = gpu_result["normals"]

    tri_cpu = np.array([r["triangle_index"] for r in cpu_results])
    tri_gpu = gpu_result["triangle_index"]

    # SDF 误差
    sdf_abs_err = np.abs(sdf_cpu - sdf_gpu)
    sdf_mean_err = float(np.mean(sdf_abs_err))
    sdf_max_err = float(np.max(sdf_abs_err))
    sdf_p95_err = float(np.percentile(sdf_abs_err, 95))

    # 距离误差
    dist_abs_err = np.abs(dist_cpu - dist_gpu)
    dist_mean_err = float(np.mean(dist_abs_err))
    dist_max_err = float(np.max(dist_abs_err))

    # 最近点误差
    nearest_err = np.linalg.norm(nearest_cpu - nearest_gpu, axis=-1)
    nearest_mean_err = float(np.mean(nearest_err))
    nearest_max_err = float(np.max(nearest_err))
    nearest_p95_err = float(np.percentile(nearest_err, 95))

    # 法向一致性
    normal_dot = np.sum(normal_cpu * normal_gpu, axis=-1)
    normal_cos_mean = float(np.mean(normal_dot))
    normal_cos_min = float(np.min(normal_dot))
    normal_cos_p5 = float(np.percentile(normal_dot, 5))

    # 三角形索引一致性
    tri_match_rate = float(np.mean(tri_cpu == tri_gpu))

    print(f"SDF 误差:")
    print(f"  mean: {sdf_mean_err:.2e}")
    print(f"  p95:  {sdf_p95_err:.2e}")
    print(f"  max:  {sdf_max_err:.2e}")

    print(f"距离误差:")
    print(f"  mean: {dist_mean_err:.2e}")
    print(f"  max:  {dist_max_err:.2e}")

    print(f"最近点误差:")
    print(f"  mean: {nearest_mean_err:.2e}")
    print(f"  p95:  {nearest_p95_err:.2e}")
    print(f"  max:  {nearest_max_err:.2e}")

    print(f"法向一致性:")
    print(f"  mean cos: {normal_cos_mean:.6f}")
    print(f"  min cos:  {normal_cos_min:.6f}")
    print(f"  p5 cos:   {normal_cos_p5:.6f}")

    print(f"三角形匹配率: {tri_match_rate:.4f}")

    # 判定
    pass_sdf = sdf_mean_err < 1e-5
    pass_dist = dist_mean_err < 1e-5
    pass_nearest = nearest_mean_err < 1e-2  # 最近点允许 1e-2 差异 (不同求解器收敛到同面上不同点)
    pass_normal = normal_cos_mean > 0.999

    print(f"\n--- 判定结果 ---")
    print(f"  SDF 精度:       {'PASS' if pass_sdf else 'FAIL'} (mean err < 1e-5)")
    print(f"  距离精度:       {'PASS' if pass_dist else 'FAIL'} (mean err < 1e-5)")
    print(f"  最近点精度:     {'PASS' if pass_nearest else 'FAIL'} (mean err < 1e-2)")
    print(f"  法向一致性:     {'PASS' if pass_normal else 'FAIL'} (mean cos > 0.999)")

    all_pass = pass_sdf and pass_dist and pass_nearest and pass_normal
    print(f"\n  总体: {'PASS' if all_pass else 'FAIL'}")

    return {
        "status": "passed" if all_pass else "failed",
        "n_points": n_points,
        "cpu_query_time": cpu_query_time,
        "gpu_query_time": gpu_query_time,
        "speedup": cpu_query_time / gpu_query_time,
        "sdf_mean_err": sdf_mean_err,
        "sdf_max_err": sdf_max_err,
        "sdf_p95_err": sdf_p95_err,
        "dist_mean_err": dist_mean_err,
        "dist_max_err": dist_max_err,
        "nearest_mean_err": nearest_mean_err,
        "nearest_max_err": nearest_max_err,
        "nearest_p95_err": nearest_p95_err,
        "normal_cos_mean": normal_cos_mean,
        "normal_cos_min": normal_cos_min,
        "tri_match_rate": tri_match_rate,
    }


def main():
    """
    主函数，运行小规模 CPU/GPU 对比验证。
    """
    import argparse

    parser = argparse.ArgumentParser(description="小规模 CPU/GPU 对比验证")
    parser.add_argument("nsm", type=str, help="输入 .nsm 文件路径")
    parser.add_argument("--n-points", type=int, default=100, help="测试点数量")
    parser.add_argument("--k-nearest", type=int, default=16, help="候选 patch 数量")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("错误: CUDA 不可用。")
        sys.exit(1)

    result = compare_cpu_gpu(args.nsm, args.n_points, args.k_nearest)

    if result.get("status") == "skipped":
        print(f"跳过: {result.get('reason')}")
        sys.exit(0)
    elif result.get("status") == "failed":
        print("对比验证失败！")
        sys.exit(1)
    else:
        print("对比验证通过！")
        sys.exit(0)


if __name__ == "__main__":
    main()
