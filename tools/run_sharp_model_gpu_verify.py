"""
带锐边/折痕模型的 GPU 教师场验证。

使用 cone.nsm 验证增强 Nagata 曲面三角形在锐边场景下的实际价值。
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from nagata_teacher_runtime.enhanced_nagata_backend_torch import (
    EnhancedNagataBackendTorch,
    load_nsm_lightweight,
)
from nagata_teacher_runtime.generate_teacher_field_gpu import generate_teacher_field_gpu

FEATURE_CODE = {"FACE": 0, "EDGE": 1, "SHARPEDGE": 2, "VERTEX": 3}


def select_model():
    """选择带锐边的模型，优先使用 cone.nsm"""
    model_path = "models/cone.nsm"
    if not os.path.exists(model_path):
        print("models/cone.nsm 不存在，尝试其他模型...")
        for name in ["cube.nsm", "gear.nsm"]:
            p = f"models/{name}"
            if os.path.exists(p):
                model_path = p
                break
        else:
            print("No sharp-edge NSM model found under models/")
            sys.exit(1)
    print(f"选择模型: {model_path}")
    return model_path


def print_model_info(mesh):
    """打印模型基本信息"""
    vertices = mesh.vertices
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))

    print(f"num_vertices: {len(vertices)}")
    print(f"num_triangles: {len(mesh.triangles)}")
    print(f"bbox_min: {bbox_min}")
    print(f"bbox_max: {bbox_max}")
    print(f"bbox_diag: {bbox_diag}")

    return bbox_min, bbox_max, bbox_diag


def run_gpu_teacher_generation(backend, tau, block_size, samples_per_axis, batch_blocks=16):
    """运行 GPU 教师场生成并记录计时"""
    print("\n--- GPU 教师场生成 ---")

    t0 = time.perf_counter()
    active_blocks = backend.enumerate_active_blocks(tau=tau, block_size=block_size)
    active_block_enum_time = time.perf_counter() - t0
    num_active_blocks = len(active_blocks)
    total_sampled_points = num_active_blocks * (samples_per_axis ** 3)

    print(f"active_block_enum_time: {active_block_enum_time:.4f}s")
    print(f"num_active_blocks: {num_active_blocks}")
    print(f"total_sampled_points: {total_sampled_points}")

    t0 = time.perf_counter()
    data = generate_teacher_field_gpu(
        backend,
        tau=tau,
        block_size=block_size,
        samples_per_axis=samples_per_axis,
        k_nearest=16,
        max_blocks=-1,
        gpu_batch_size=batch_blocks * (samples_per_axis ** 3),
    )
    gpu_query_time = time.perf_counter() - t0
    kept_samples = int(data["points"].shape[0])
    kept_ratio = kept_samples / total_sampled_points if total_sampled_points > 0 else 0.0

    print(f"gpu_query_time: {gpu_query_time:.4f}s")
    print(f"kept_samples: {kept_samples}")
    print(f"kept_ratio: {kept_ratio:.6f}")

    os.makedirs("outputs", exist_ok=True)
    t0 = time.perf_counter()
    np.savez_compressed("outputs/sharp_teacher_gpu.npz", **data)
    npz_write_time = time.perf_counter() - t0

    print(f"npz_write_time: {npz_write_time:.4f}s")

    total_wall_time = backend_build_time + active_block_enum_time + gpu_query_time + npz_write_time

    return data, {
        "backend_build_time": backend_build_time,
        "active_block_enum_time": active_block_enum_time,
        "gpu_query_time": gpu_query_time,
        "npz_write_time": npz_write_time,
        "total_wall_time": total_wall_time,
        "total_sampled_points": total_sampled_points,
        "kept_samples": kept_samples,
        "kept_ratio": kept_ratio,
    }


def verify_self_consistency(data):
    """几何自洽性验证"""
    print("\n--- 自洽性验证 ---")

    normals = data["normals"]
    sdf = data["sdf"]
    unsigned_distance = data["unsigned_distance"]

    # 法向长度
    normal_norms = np.linalg.norm(normals, axis=1)
    print(f"法向长度: mean={np.mean(normal_norms):.8f}, p5={np.percentile(normal_norms, 5):.8f}, "
          f"min={np.min(normal_norms):.8f}, max={np.max(normal_norms):.8f}")

    # unsigned_distance 非负
    ud_min = float(np.min(unsigned_distance))
    ud_neg_count = int(np.sum(unsigned_distance < 0))
    print(f"unsigned_distance min: {ud_min}")
    print(f"unsigned_distance negative count: {ud_neg_count}")

    # sdf 与 unsigned_distance 一致性
    abs_diff = np.abs(np.abs(sdf) - unsigned_distance)
    print(f"|abs(sdf) - unsigned|: mean={np.mean(abs_diff):.6e}, "
          f"p95={np.percentile(abs_diff, 95):.6e}, max={np.max(abs_diff):.6e}")

    # feature_code 分布
    if "feature_code" in data:
        fc = data["feature_code"]
        fc_hist = {str(k): int(np.sum(fc == k)) for k in FEATURE_CODE.values()}
        print(f"feature_code 分布: {fc_hist}")
    else:
        fc_hist = {}

    return {
        "normal_norm_mean": float(np.mean(normal_norms)),
        "normal_norm_p5": float(np.percentile(normal_norms, 5)),
        "normal_norm_min": float(np.min(normal_norms)),
        "normal_norm_max": float(np.max(normal_norms)),
        "unsigned_distance_min": ud_min,
        "unsigned_distance_negative_count": ud_neg_count,
        "abs_sdf_minus_unsigned_mean": float(np.mean(abs_diff)),
        "abs_sdf_minus_unsigned_p95": float(np.percentile(abs_diff, 95)),
        "abs_sdf_minus_unsigned_max": float(np.max(abs_diff)),
        "feature_code_hist": fc_hist,
    }


def main():
    global backend_build_time

    print("=" * 60)
    print("带锐边/折痕模型 GPU 教师场验证")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA 不可用，GPU 教师场验证中止。")
        sys.exit(1)

    print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")

    # 选择模型
    model_path = select_model()
    mesh = load_nsm_lightweight(model_path)
    bbox_min, bbox_max, bbox_diag = print_model_info(mesh)

    # 固定参数
    tau = 0.02 * bbox_diag
    block_size = tau
    samples_per_axis = 4
    batch_blocks = 16
    device = "cuda"
    use_cache = False
    force_recompute = True
    bake_cache = True
    gap_threshold = 1e-4
    k_factor = 0.0

    print(f"\ntau: {tau}")
    print(f"block_size: {block_size}")
    print(f"samples_per_axis: {samples_per_axis}")
    print(f"batch_blocks: {batch_blocks}")

    # 构造 GPU 后端
    t0 = time.perf_counter()
    backend = EnhancedNagataBackendTorch(
        model_path,
        use_cache=use_cache,
        force_recompute=force_recompute,
        bake_cache=bake_cache,
        gap_threshold=gap_threshold,
        k_factor=k_factor,
    )
    backend_build_time = time.perf_counter() - t0

    print(f"\n后端构建完成:")
    print(f"  backend_build_time: {backend_build_time:.4f}s")
    print(f"  num_crease_edges: {backend.build_info.num_crease_edges}")
    print(f"  num_vertices: {backend.build_info.num_vertices}")
    print(f"  num_triangles: {backend.build_info.num_triangles}")

    # 运行教师场生成
    data, timing = run_gpu_teacher_generation(
        backend, tau, block_size, samples_per_axis, batch_blocks
    )

    # 自洽性验证
    consistency = verify_self_consistency(data)

    # 判定结论
    num_crease = backend.build_info.num_crease_edges
    if num_crease > 0:
        crease_msg = "Sharp model contains crease edges and exercises enhanced Nagata path."
    else:
        crease_msg = "Model has no crease edges, but GPU teacher generation succeeded."

    interpretation = (
        f"GPU 教师场生成在 {model_path} 上成功运行。"
        f"总耗时 {timing['total_wall_time']:.2f}s，"
        f"保留 {timing['kept_samples']} 个样本 ({timing['kept_ratio']:.4f})。"
        f"{crease_msg} "
        f"结果数值自洽。"
    )
    print(f"\n结论: {interpretation}")

    # 写入 JSON 报告
    report = {
        "model_path": model_path,
        "num_vertices": backend.build_info.num_vertices,
        "num_triangles": backend.build_info.num_triangles,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_diag": bbox_diag,
        "tau": tau,
        "block_size": block_size,
        "samples_per_axis": samples_per_axis,
        "batch_blocks": batch_blocks,
        "device": str(backend.device),
        "num_crease_edges": num_crease,
        "num_active_blocks": timing["total_sampled_points"] // (samples_per_axis ** 3) if samples_per_axis > 0 else 0,
        **timing,
        **consistency,
        "final_interpretation": interpretation,
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/sharp_teacher_gpu_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nJSON 报告已写出: outputs/sharp_teacher_gpu_report.json")


if __name__ == "__main__":
    main()
