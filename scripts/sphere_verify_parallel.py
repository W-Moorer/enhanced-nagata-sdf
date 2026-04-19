"""
并行版本 sphere.nsm 教师场分阶段验证脚本。

按照 prompt2.md 步骤 7 第三档策略：
A. max_blocks = 500
B. max_blocks = 2000
C. max_blocks = -1 (全量)

每一档都打印完整误差指标。
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from nagata_teacher_runtime.enhanced_nagata_backend import (
    load_nsm_lightweight,
    EnhancedNagataBackend,
)
from nagata_teacher_runtime.generate_teacher_field import generate_teacher_field_parallel


def estimate_sphere_params():
    """读取 sphere.nsm 并估计球参数"""
    print("=" * 60)
    print("步骤 1：读取 sphere.nsm 并估计球参数")
    print("=" * 60)

    mesh = load_nsm_lightweight("models/sphere.nsm")
    vertices = mesh.vertices

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    center_fit = vertices.mean(axis=0)
    radii = np.linalg.norm(vertices - center_fit[None, :], axis=1)
    R_fit = float(radii.mean())
    R_std = float(radii.std())

    print(f"bbox_min: {bbox_min}")
    print(f"bbox_max: {bbox_max}")
    print(f"bbox_diag: {bbox_diag}")
    print(f"center_fit: {center_fit}")
    print(f"R_fit: {R_fit}")
    print(f"R_std: {R_std}")
    print(f"R_std / R_fit: {R_std / R_fit}")

    return {
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_diag": bbox_diag,
        "center_fit": center_fit.tolist(),
        "R_fit": R_fit,
        "R_std": R_std,
        "R_std_over_R_fit": R_std / R_fit,
    }


def set_parameters(R_fit):
    """固定参数 - 使用降级第二档"""
    print("\n" + "=" * 60)
    print("步骤 2：固定参数（降级第二档）")
    print("=" * 60)

    tau = 0.05 * R_fit
    block_size = tau
    samples_per_axis = 3
    k_nearest = 16
    workers = os.cpu_count() or 1  # 使用全部 CPU 核心
    chunk_blocks = 64  # 增大 chunk 减少调度开销

    print(f"tau: {tau}")
    print(f"block_size: {block_size}")
    print(f"samples_per_axis: {samples_per_axis}")
    print(f"k_nearest: {k_nearest}")
    print(f"workers: {workers}")
    print(f"chunk_blocks: {chunk_blocks}")

    return {
        "tau": tau,
        "block_size": block_size,
        "samples_per_axis": samples_per_axis,
        "k_nearest": k_nearest,
        "workers": workers,
        "chunk_blocks": chunk_blocks,
    }


def construct_backend():
    """构造 backend - 使用缓存加速"""
    print("\n" + "=" * 60)
    print("步骤 3：构造 backend")
    print("=" * 60)

    backend = EnhancedNagataBackend(
        "models/sphere.nsm",
        use_cache=True,
        force_recompute=False,
        bake_cache=False,
        gap_threshold=1e-4,
        k_factor=0.0,
    )

    print(f"nsm_path: {backend.build_info.nsm_path}")
    print(f"num_vertices: {backend.build_info.num_vertices}")
    print(f"num_triangles: {backend.build_info.num_triangles}")
    print(f"num_crease_edges: {backend.build_info.num_crease_edges}")
    print(f"used_cache: {backend.build_info.used_cache}")
    print(f"eng_path: {backend.build_info.eng_path}")

    return backend


def generate_and_verify(backend, params, sphere_params, max_blocks, stage_name):
    """生成教师场并验证准确性"""
    print("\n" + "=" * 60)
    print(f"阶段: {stage_name}")
    print(f"max_blocks: {max_blocks}")
    print("=" * 60)

    t0 = time.perf_counter()

    data = generate_teacher_field_parallel(
        backend,
        tau=params["tau"],
        block_size=params["block_size"],
        samples_per_axis=params["samples_per_axis"],
        k_nearest=params["k_nearest"],
        max_blocks=max_blocks,
        workers=params["workers"],
        chunk_blocks=params["chunk_blocks"],
    )

    parallel_time = time.perf_counter() - t0
    num_samples = int(data["points"].shape[0])

    print(f"\n{stage_name} 完成:")
    print(f"  parallel_time: {parallel_time:.2f} 秒")
    print(f"  num_samples: {num_samples}")

    if num_samples == 0:
        print("WARNING: 样本数为 0")
        return None, parallel_time

    # 验证 SDF 准确性
    points = data["points"]
    phi_teacher = data["sdf"]
    center_fit = np.array(sphere_params["center_fit"])
    R_fit = sphere_params["R_fit"]

    phi_ref = np.linalg.norm(points - center_fit[None, :], axis=1) - R_fit
    err_same = float(np.mean(np.abs(phi_teacher - phi_ref)))
    err_flip = float(np.mean(np.abs(phi_teacher + phi_ref)))

    if err_flip < err_same:
        phi_ref_used = -phi_ref
        chosen_sign = "flipped"
    else:
        phi_ref_used = phi_ref
        chosen_sign = "same"

    abs_err = np.abs(phi_teacher - phi_ref_used)
    mean_abs_err = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((phi_teacher - phi_ref_used) ** 2)))
    p95_abs_err = float(np.percentile(abs_err, 95))
    max_abs_err = float(np.max(abs_err))

    teacher_sign = np.where(phi_teacher >= 0.0, 1, -1)
    ref_sign = np.where(phi_ref_used >= 0.0, 1, -1)
    sign_mismatch_rate = float(np.mean(teacher_sign != ref_sign))

    print(f"  chosen_sign_convention: {chosen_sign}")
    print(f"  mean_abs_err: {mean_abs_err}")
    print(f"  rmse: {rmse}")
    print(f"  p95_abs_err: {p95_abs_err}")
    print(f"  max_abs_err: {max_abs_err}")
    print(f"  sign_mismatch_rate: {sign_mismatch_rate}")

    # 验证最近点与法向
    nearest_points = data["nearest_points"]
    normals = data["normals"]

    nearest_r = np.linalg.norm(nearest_points - center_fit[None, :], axis=1)
    radial_err = np.abs(nearest_r - R_fit)

    n_true = nearest_points - center_fit[None, :]
    n_true_norm = np.linalg.norm(n_true, axis=1, keepdims=True)
    n_true = n_true / np.clip(n_true_norm, 1e-12, None)
    normal_cos = np.sum(normals * n_true, axis=1)

    radial_err_mean = float(np.mean(radial_err))
    radial_err_p95 = float(np.percentile(radial_err, 95))
    radial_err_max = float(np.max(radial_err))
    normal_cos_mean = float(np.mean(normal_cos))
    normal_cos_p5 = float(np.percentile(normal_cos, 5))
    normal_cos_min = float(np.min(normal_cos))

    print(f"  radial_err.mean: {radial_err_mean}")
    print(f"  radial_err.p95: {radial_err_p95}")
    print(f"  radial_err.max: {radial_err_max}")
    print(f"  normal_cos.mean: {normal_cos_mean}")
    print(f"  normal_cos.p5: {normal_cos_p5}")
    print(f"  normal_cos.min: {normal_cos_min}")

    result = {
        "stage": stage_name,
        "max_blocks": max_blocks,
        "parallel_time": parallel_time,
        "num_samples": num_samples,
        "chosen_sign_convention": chosen_sign,
        "mean_abs_err": mean_abs_err,
        "rmse": rmse,
        "p95_abs_err": p95_abs_err,
        "max_abs_err": max_abs_err,
        "sign_mismatch_rate": sign_mismatch_rate,
        "radial_err_mean": radial_err_mean,
        "radial_err_p95": radial_err_p95,
        "radial_err_max": radial_err_max,
        "normal_cos_mean": normal_cos_mean,
        "normal_cos_p5": normal_cos_p5,
        "normal_cos_min": normal_cos_min,
    }

    # 保存最后一阶段的 npz
    if max_blocks == -1:
        os.makedirs("outputs", exist_ok=True)
        np.savez_compressed("outputs/sphere_teacher_parallel.npz", **data)
        print(f"\n全量教师场已保存: outputs/sphere_teacher_parallel.npz")

    return result, parallel_time


def main():
    sphere_params = estimate_sphere_params()
    params = set_parameters(sphere_params["R_fit"])
    backend = construct_backend()

    # 分阶段验证
    stages = [
        ("A. max_blocks=500", 500),
        ("B. max_blocks=2000", 2000),
        ("C. max_blocks=-1 (全量)", -1),
    ]

    all_results = []
    total_time = 0.0

    for stage_name, max_blocks in stages:
        result, stage_time = generate_and_verify(backend, params, sphere_params, max_blocks, stage_name)
        total_time += stage_time
        if result is not None:
            all_results.append(result)

    # 输出总结
    print("\n" + "=" * 60)
    print("分阶段验证总结")
    print("=" * 60)

    for r in all_results:
        print(f"\n{r['stage']}:")
        print(f"  parallel_time: {r['parallel_time']:.2f}s")
        print(f"  num_samples: {r['num_samples']}")
        print(f"  mean_abs_err: {r['mean_abs_err']}")
        print(f"  rmse: {r['rmse']}")
        print(f"  p95_abs_err: {r['p95_abs_err']}")
        print(f"  max_abs_err: {r['max_abs_err']}")
        print(f"  sign_mismatch_rate: {r['sign_mismatch_rate']}")
        print(f"  radial_err.mean: {r['radial_err_mean']}")
        print(f"  normal_cos.mean: {r['normal_cos_mean']}")
        print(f"  normal_cos.min: {r['normal_cos_min']}")

    print(f"\n总耗时: {total_time:.2f} 秒")

    # 写入 JSON 报告（使用最后一阶段结果）
    if all_results:
        final = all_results[-1]
        report = {
            **sphere_params,
            "tau": params["tau"],
            "block_size": params["block_size"],
            "samples_per_axis": params["samples_per_axis"],
            "k_nearest": params["k_nearest"],
            "workers": params["workers"],
            "chunk_blocks": params["chunk_blocks"],
            "num_vertices": backend.build_info.num_vertices,
            "num_triangles": backend.build_info.num_triangles,
            "num_crease_edges": backend.build_info.num_crease_edges,
            "used_cache": backend.build_info.used_cache,
            "stages": all_results,
            "total_time": total_time,
            **{k: v for k, v in final.items() if k != "stage"},
        }

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/sphere_teacher_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\nJSON 报告已写出: outputs/sphere_teacher_report.json")


if __name__ == "__main__":
    main()
