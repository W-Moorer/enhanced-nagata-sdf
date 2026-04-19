"""
验证 sphere.nsm 教师场与解析球 SDF 的一致性。
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# 确保能导入 nagata_teacher_runtime 模块
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from nagata_teacher_runtime.enhanced_nagata_backend import (
    load_nsm_lightweight,
    EnhancedNagataBackend,
)
from nagata_teacher_runtime.generate_teacher_field import generate_teacher_field


def step1_estimate_sphere_params():
    """步骤 1：读取 sphere.nsm 并估计球参数"""
    print("=" * 60)
    print("步骤 1：读取 sphere.nsm 并估计球参数")
    print("=" * 60)

    mesh = load_nsm_lightweight("models/sphere.nsm")
    vertices = mesh.vertices

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_diag = np.linalg.norm(bbox_max - bbox_min)
    center_fit = vertices.mean(axis=0)
    radii = np.linalg.norm(vertices - center_fit[None, :], axis=1)
    R_fit = radii.mean()
    R_std = radii.std()

    print(f"bbox_min: {bbox_min}")
    print(f"bbox_max: {bbox_max}")
    print(f"bbox_diag: {bbox_diag}")
    print(f"center_fit: {center_fit}")
    print(f"R_fit: {R_fit}")
    print(f"R_std: {R_std}")
    print(f"R_std / R_fit: {R_std / R_fit}")

    if R_std / R_fit > 1e-2:
        print("WARNING: R_std / R_fit > 1e-2，网格点与理想球偏差较大！")

    return {
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_diag": bbox_diag,
        "center_fit": center_fit,
        "R_fit": R_fit,
        "R_std": R_std,
        "R_std_ratio": R_std / R_fit,
    }


def step2_set_parameters(R_fit):
    """步骤 2：固定参数"""
    print("\n" + "=" * 60)
    print("步骤 2：固定参数")
    print("=" * 60)

    tau = 0.05 * R_fit
    block_size = tau
    samples_per_axis = 4
    k_nearest = 16

    gap_threshold = 1e-4
    k_factor = 0.0
    use_cache = False
    force_recompute = True
    bake_cache = True

    print(f"tau: {tau}")
    print(f"block_size: {block_size}")
    print(f"samples_per_axis: {samples_per_axis}")
    print(f"k_nearest: {k_nearest}")
    print(f"gap_threshold: {gap_threshold}")
    print(f"k_factor: {k_factor}")
    print(f"use_cache: {use_cache}")
    print(f"force_recompute: {force_recompute}")
    print(f"bake_cache: {bake_cache}")

    return {
        "tau": tau,
        "block_size": block_size,
        "samples_per_axis": samples_per_axis,
        "k_nearest": k_nearest,
        "gap_threshold": gap_threshold,
        "k_factor": k_factor,
        "use_cache": use_cache,
        "force_recompute": force_recompute,
        "bake_cache": bake_cache,
    }


def step3_construct_backend(params):
    """步骤 3：构造后端"""
    print("\n" + "=" * 60)
    print("步骤 3：构造后端")
    print("=" * 60)

    backend = EnhancedNagataBackend(
        "models/sphere.nsm",
        use_cache=params["use_cache"],
        force_recompute=params["force_recompute"],
        bake_cache=params["bake_cache"],
        gap_threshold=params["gap_threshold"],
        k_factor=params["k_factor"],
    )

    print(f"backend.build_info.nsm_path: {backend.build_info.nsm_path}")
    print(f"backend.build_info.num_vertices: {backend.build_info.num_vertices}")
    print(f"backend.build_info.num_triangles: {backend.build_info.num_triangles}")
    print(f"backend.build_info.num_crease_edges: {backend.build_info.num_crease_edges}")
    print(f"backend.build_info.used_cache: {backend.build_info.used_cache}")
    print(f"backend.build_info.eng_path: {backend.build_info.eng_path}")

    if backend.build_info.num_crease_edges != 0:
        print(f"NOTE: num_crease_edges != 0，值为 {backend.build_info.num_crease_edges}，继续执行。")

    return backend


def step4_generate_teacher_field(backend, params):
    """步骤 4：生成教师场"""
    print("\n" + "=" * 60)
    print("步骤 4：生成教师场")
    print("=" * 60)

    data = generate_teacher_field(
        backend,
        tau=params["tau"],
        block_size=params["block_size"],
        samples_per_axis=params["samples_per_axis"],
        k_nearest=params["k_nearest"],
        max_blocks=-1,
    )

    os.makedirs("outputs", exist_ok=True)
    np.savez_compressed("outputs/sphere_teacher.npz", **data)

    print(f"data['points'].shape: {data['points'].shape}")
    print(f"data['sdf'].shape: {data['sdf'].shape}")
    print(f"data['nearest_points'].shape: {data['nearest_points'].shape}")
    print(f"data['normals'].shape: {data['normals'].shape}")
    print(f"data['triangle_index'].shape: {data['triangle_index'].shape}")
    metadata_json = data["metadata_json"].tobytes().decode("utf-8")
    print(f"metadata_json: {metadata_json}")

    if data["points"].shape[0] == 0:
        print("ERROR: points 数量为 0，停止执行！")
        sys.exit(1)

    return data, json.loads(metadata_json)


def step5_verify_sdf_consistency(data, sphere_params):
    """步骤 5：验证教师场与解析球 SDF 的一致性"""
    print("\n" + "=" * 60)
    print("步骤 5：验证教师场与解析球 SDF 的一致性")
    print("=" * 60)

    points = data["points"]
    sdf_teacher = data["sdf"]
    center_fit = sphere_params["center_fit"]
    R_fit = sphere_params["R_fit"]

    phi_ref = np.linalg.norm(points - center_fit[None, :], axis=1) - R_fit

    err_same = np.abs(sdf_teacher - phi_ref)
    err_flip = np.abs(sdf_teacher + phi_ref)

    if np.mean(err_flip) < np.mean(err_same):
        phi_ref_used = -phi_ref
        chosen_sign = "flipped"
    else:
        phi_ref_used = phi_ref
        chosen_sign = "same"

    abs_err = np.abs(sdf_teacher - phi_ref_used)
    rmse = np.sqrt(np.mean((sdf_teacher - phi_ref_used) ** 2))
    p95_abs_err = np.percentile(abs_err, 95)
    max_abs_err = np.max(abs_err)
    mean_abs_err = np.mean(abs_err)

    sign_teacher = np.where(sdf_teacher == 0, 1, np.sign(sdf_teacher))
    sign_ref = np.where(phi_ref_used == 0, 1, np.sign(phi_ref_used))
    sign_mismatch_rate = np.mean(sign_teacher != sign_ref)

    print(f"mean_abs_err: {mean_abs_err}")
    print(f"rmse: {rmse}")
    print(f"p95_abs_err: {p95_abs_err}")
    print(f"max_abs_err: {max_abs_err}")
    print(f"sign_mismatch_rate: {sign_mismatch_rate}")
    print(f"chosen_sign_convention: {chosen_sign}")

    return {
        "mean_abs_err": mean_abs_err,
        "rmse": rmse,
        "p95_abs_err": p95_abs_err,
        "max_abs_err": max_abs_err,
        "sign_mismatch_rate": sign_mismatch_rate,
        "chosen_sign_convention": chosen_sign,
    }


def step6_verify_nearest_points(data, sphere_params):
    """步骤 6：验证最近点正确性"""
    print("\n" + "=" * 60)
    print("步骤 6：验证最近点正确性")
    print("=" * 60)

    nearest_points = data["nearest_points"]
    normals = data["normals"]
    center_fit = sphere_params["center_fit"]
    R_fit = sphere_params["R_fit"]

    radial_err = np.abs(np.linalg.norm(nearest_points - center_fit[None, :], axis=1) - R_fit)

    denom = np.linalg.norm(nearest_points - center_fit[None, :], axis=1)
    n_true = (nearest_points - center_fit[None, :]) / denom[:, None]
    normal_cos = np.sum(normals * n_true, axis=1)

    radial_err_mean = np.mean(radial_err)
    radial_err_p95 = np.percentile(radial_err, 95)
    radial_err_max = np.max(radial_err)
    normal_cos_mean = np.mean(normal_cos)
    normal_cos_p5 = np.percentile(normal_cos, 5)
    normal_cos_min = np.min(normal_cos)

    print(f"radial_err.mean(): {radial_err_mean}")
    print(f"radial_err.p95: {radial_err_p95}")
    print(f"radial_err.max: {radial_err_max}")
    print(f"normal_cos.mean(): {normal_cos_mean}")
    print(f"normal_cos.p5: {normal_cos_p5}")
    print(f"normal_cos.min: {normal_cos_min}")

    return {
        "radial_err_mean": radial_err_mean,
        "radial_err_p95": radial_err_p95,
        "radial_err_max": radial_err_max,
        "normal_cos_mean": normal_cos_mean,
        "normal_cos_p5": normal_cos_p5,
        "normal_cos_min": normal_cos_min,
    }


def step7_output_summary(sphere_params, params, backend, metadata, sdf_results, nearest_results):
    """步骤 7：输出结论 summary"""
    print("\n" + "=" * 60)
    print("步骤 7：最终 Summary")
    print("=" * 60)

    print("\nA. 模型尺度")
    print(f"  bbox_diag: {sphere_params['bbox_diag']}")
    print(f"  R_fit: {sphere_params['R_fit']}")
    print(f"  R_std / R_fit: {sphere_params['R_std_ratio']}")

    print("\nB. 教师生成参数")
    print(f"  tau: {params['tau']}")
    print(f"  block_size: {params['block_size']}")
    print(f"  samples_per_axis: {params['samples_per_axis']}")
    print(f"  k_nearest: {params['k_nearest']}")

    print("\nC. 后端构建结果")
    print(f"  num_vertices: {backend.build_info.num_vertices}")
    print(f"  num_triangles: {backend.build_info.num_triangles}")
    print(f"  num_crease_edges: {backend.build_info.num_crease_edges}")
    print(f"  used_cache: {backend.build_info.used_cache}")
    print(f"  eng_path: {backend.build_info.eng_path}")

    print("\nD. 教师场结果")
    print(f"  num_active_blocks: {metadata['num_active_blocks']}")
    print(f"  num_samples: {metadata['num_samples']}")

    print("\nE. 球解析验证结果")
    print(f"  mean_abs_err: {sdf_results['mean_abs_err']}")
    print(f"  rmse: {sdf_results['rmse']}")
    print(f"  p95_abs_err: {sdf_results['p95_abs_err']}")
    print(f"  max_abs_err: {sdf_results['max_abs_err']}")
    print(f"  sign_mismatch_rate: {sdf_results['sign_mismatch_rate']}")
    print(f"  chosen_sign_convention: {sdf_results['chosen_sign_convention']}")

    print("\nF. 最近点与法向验证")
    print(f"  radial_err.mean: {nearest_results['radial_err_mean']}")
    print(f"  radial_err.p95: {nearest_results['radial_err_p95']}")
    print(f"  radial_err.max: {nearest_results['radial_err_max']}")
    print(f"  normal_cos.mean: {nearest_results['normal_cos_mean']}")
    print(f"  normal_cos.p5: {nearest_results['normal_cos_p5']}")
    print(f"  normal_cos.min: {nearest_results['normal_cos_min']}")


def main():
    sphere_params = step1_estimate_sphere_params()
    params = step2_set_parameters(sphere_params["R_fit"])
    backend = step3_construct_backend(params)
    data, metadata = step4_generate_teacher_field(backend, params)
    sdf_results = step5_verify_sdf_consistency(data, sphere_params)
    nearest_results = step6_verify_nearest_points(data, sphere_params)
    step7_output_summary(sphere_params, params, backend, metadata, sdf_results, nearest_results)


if __name__ == "__main__":
    main()
