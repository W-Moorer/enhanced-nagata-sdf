"""
Sphere 误差来源分解实验。

回答 4 个问题：
Q1. 当前总误差有多大，和已有报告是否一致？
Q2. 当前误差是不是主要来自"教师曲面本身不是解析球"？
Q3. 当前误差中，GPU 实现和 CPU 实现的数值差异占比有多大？
Q4. 当前 sign mismatch 是否几乎全部属于零水平集附近的数值敏感翻转？
"""

from __future__ import annotations

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from nagata_teacher_runtime.enhanced_nagata_backend import EnhancedNagataBackend
from nagata_teacher_runtime.enhanced_nagata_backend_torch import (
    EnhancedNagataBackendTorch,
    load_nsm_lightweight,
)


def load_teacher_npz():
    """加载教师场结果，按优先级查找"""
    for path in [
        "outputs/sphere_teacher_gpu.npz",
        "outputs/sphere_teacher_parallel.npz",
        "outputs/sphere_teacher.npz",
    ]:
        if os.path.exists(path):
            print(f"使用教师场结果: {path}")
            return np.load(path, allow_pickle=True), path
    raise FileNotFoundError("找不到任何 sphere 教师场结果文件")


def step1_basic_verification(data, npz_path):
    """步骤 1：基础真值与已有结果复核"""
    print("\n" + "=" * 60)
    print("步骤 1：基础真值与已有结果复核")
    print("=" * 60)

    mesh = load_nsm_lightweight("models/sphere.nsm")
    vertices = mesh.vertices
    center_fit = vertices.mean(axis=0)
    radii = np.linalg.norm(vertices - center_fit[None, :], axis=1)
    R_fit = float(radii.mean())
    R_std = float(radii.std())

    print(f"center_fit: {center_fit}")
    print(f"R_fit: {R_fit}")
    print(f"R_std: {R_std}")

    points = data["points"]
    sdf = data["sdf"]

    phi_ref = np.linalg.norm(points - center_fit[None, :], axis=1) - R_fit
    err_same = np.mean(np.abs(sdf - phi_ref))
    err_flip = np.mean(np.abs(sdf + phi_ref))

    if err_flip < err_same:
        phi_ref_used = -phi_ref
        chosen_sign = "flipped"
    else:
        phi_ref_used = phi_ref
        chosen_sign = "same"

    abs_err = np.abs(sdf - phi_ref_used)
    mean_abs_err = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((sdf - phi_ref_used) ** 2)))
    p95_abs_err = float(np.percentile(abs_err, 95))
    max_abs_err = float(np.max(abs_err))
    total_samples = len(sdf)

    teacher_sign = np.where(sdf >= 0, 1, -1)
    ref_sign = np.where(phi_ref_used >= 0, 1, -1)
    sign_mismatch_rate = float(np.mean(teacher_sign != ref_sign))

    # 与已有报告对比
    report_path = "outputs/sphere_teacher_gpu_report.json"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            existing = json.load(f)["sdf_results"]
        print(f"\n与已有报告对比 (tol=1e-9):")
        checks = [
            ("mean_abs_err", mean_abs_err, existing["mean_abs_err"]),
            ("rmse", rmse, existing["rmse"]),
            ("p95_abs_err", p95_abs_err, existing["p95_abs_err"]),
            ("max_abs_err", max_abs_err, existing["max_abs_err"]),
        ]
        for name, new_val, old_val in checks:
            diff = abs(new_val - old_val)
            ok = "PASS" if diff < 1e-9 else "WARN"
            print(f"  {name}: new={new_val:.6e}, old={old_val:.6e}, diff={diff:.2e} [{ok}]")

    return {
        "center_fit": center_fit.tolist(),
        "R_fit": R_fit,
        "R_std": R_std,
        "chosen_sign_convention": chosen_sign,
        "total_samples": total_samples,
        "mean_abs_err": mean_abs_err,
        "rmse": rmse,
        "p95_abs_err": p95_abs_err,
        "max_abs_err": max_abs_err,
        "sign_mismatch_rate": sign_mismatch_rate,
        "phi_ref_used": phi_ref_used,
    }


def step2_geometric_approximation_error(data, basic):
    """步骤 2：误差来源 A -- 教师曲面几何逼近误差"""
    print("\n" + "=" * 60)
    print("步骤 2：误差来源 A -- 教师曲面几何逼近误差")
    print("=" * 60)

    center_fit = np.array(basic["center_fit"])
    R_fit = basic["R_fit"]

    nearest_points = data["nearest_points"]
    sdf = data["sdf"]
    phi_ref_used = basic["phi_ref_used"]

    nearest_r = np.linalg.norm(nearest_points - center_fit[None, :], axis=1)
    surface_radial_err = np.abs(nearest_r - R_fit)

    surface_radial_err_mean = float(np.mean(surface_radial_err))
    surface_radial_err_p50 = float(np.percentile(surface_radial_err, 50))
    surface_radial_err_p95 = float(np.percentile(surface_radial_err, 95))
    surface_radial_err_p99 = float(np.percentile(surface_radial_err, 99))
    surface_radial_err_max = float(np.max(surface_radial_err))

    distance_abs_err = np.abs(sdf - phi_ref_used)

    corr = float(np.corrcoef(surface_radial_err, distance_abs_err)[0, 1])
    mean_ratio_1 = float(np.mean(distance_abs_err / np.maximum(surface_radial_err, 1e-15)))
    mean_ratio_2 = float(np.mean(surface_radial_err / np.maximum(distance_abs_err, 1e-15)))

    print(f"surface_radial_err mean: {surface_radial_err_mean:.6e}")
    print(f"surface_radial_err p50: {surface_radial_err_p50:.6e}")
    print(f"surface_radial_err p95: {surface_radial_err_p95:.6e}")
    print(f"surface_radial_err p99: {surface_radial_err_p99:.6e}")
    print(f"surface_radial_err max: {surface_radial_err_max:.6e}")
    print(f"corr(surface_radial_err, distance_abs_err): {corr:.6f}")
    print(f"mean(distance/surface): {mean_ratio_1:.6f}")
    print(f"mean(surface/distance): {mean_ratio_2:.6f}")

    if abs(surface_radial_err_mean - basic["mean_abs_err"]) / max(basic["mean_abs_err"], 1e-15) < 0.1:
        print("=> surface_radial_err 与 distance_abs_err 量级高度一致，误差主要受几何逼近控制。")

    return {
        "surface_radial_err_mean": surface_radial_err_mean,
        "surface_radial_err_p50": surface_radial_err_p50,
        "surface_radial_err_p95": surface_radial_err_p95,
        "surface_radial_err_p99": surface_radial_err_p99,
        "surface_radial_err_max": surface_radial_err_max,
        "corr_surface_vs_distance_err": corr,
    }


def step3_cpu_gpu_comparison(data, basic):
    """步骤 3：误差来源 B -- GPU vs CPU 数值实现差异"""
    print("\n" + "=" * 60)
    print("步骤 3：误差来源 B -- GPU vs CPU 数值实现差异")
    print("=" * 60)

    center_fit = np.array(basic["center_fit"])
    R_fit = basic["R_fit"]
    points = data["points"]
    total_samples = len(points)
    subset_n = min(5000, total_samples)

    rng = np.random.default_rng(0)
    idx = rng.choice(total_samples, size=subset_n, replace=False)
    points_subset = points[idx]

    print(f"子集样本数: {subset_n}")

    # CPU 查询
    print("构造 CPU 后端...")
    cpu_backend = EnhancedNagataBackend(
        "models/sphere.nsm",
        use_cache=False,
        force_recompute=False,
        bake_cache=False,
        gap_threshold=1e-4,
        k_factor=0.0,
    )

    t0 = time.perf_counter()
    cpu_sdf = []
    cpu_unsigned = []
    cpu_nearest = []
    cpu_normals = []
    for p in points_subset:
        q = cpu_backend.query_point(p, k_nearest=16)
        cpu_sdf.append(q.signed_distance)
        cpu_unsigned.append(q.distance)
        cpu_nearest.append(q.nearest_point)
        cpu_normals.append(q.normal)
    cpu_time = time.perf_counter() - t0
    print(f"CPU 查询耗时: {cpu_time:.2f}s")

    # GPU 查询
    print("构造 GPU 后端...")
    gpu_backend = EnhancedNagataBackendTorch(
        "models/sphere.nsm",
        use_cache=False,
        force_recompute=False,
        bake_cache=False,
        gap_threshold=1e-4,
        k_factor=0.0,
    )
    t0 = time.perf_counter()
    gpu_result = gpu_backend.query_points_gpu(points_subset, k_nearest=16, batch_size=2048)
    gpu_time = time.perf_counter() - t0
    print(f"GPU 查询耗时: {gpu_time:.2f}s")

    gpu_sdf = gpu_result["sdf"]
    gpu_unsigned = gpu_result["unsigned_distance"]
    gpu_nearest = gpu_result["nearest_points"]
    gpu_normals = gpu_result["normals"]

    cpu_sdf = np.array(cpu_sdf)
    cpu_unsigned = np.array(cpu_unsigned)
    cpu_nearest = np.array(cpu_nearest)
    cpu_normals = np.array(cpu_normals)

    # CPU/GPU 差异
    cpu_gpu_abs_diff_sdf = np.abs(cpu_sdf - gpu_sdf)
    cpu_gpu_max_abs_diff_sdf = float(np.max(cpu_gpu_abs_diff_sdf))
    cpu_gpu_mean_abs_diff_sdf = float(np.mean(cpu_gpu_abs_diff_sdf))
    cpu_gpu_p95_abs_diff_sdf = float(np.percentile(cpu_gpu_abs_diff_sdf, 95))

    cpu_gpu_max_abs_diff_unsigned = float(np.max(np.abs(cpu_unsigned - gpu_unsigned)))
    cpu_gpu_max_abs_diff_nearest = float(np.max(np.abs(cpu_nearest - gpu_nearest)))
    cpu_gpu_max_abs_diff_normals = float(np.max(np.abs(cpu_normals - gpu_normals)))

    cpu_sign = np.where(cpu_sdf >= 0, 1, -1)
    gpu_sign = np.where(gpu_sdf >= 0, 1, -1)
    cpu_gpu_sign_disagreement = float(np.mean(cpu_sign != gpu_sign))

    # 各自对解析球的误差
    phi_ref = np.linalg.norm(points_subset - center_fit[None, :], axis=1) - R_fit
    err_same_cpu = np.mean(np.abs(cpu_sdf - phi_ref))
    err_flip_cpu = np.mean(np.abs(cpu_sdf + phi_ref))
    if err_flip_cpu < err_same_cpu:
        phi_ref_used_cpu = -phi_ref
    else:
        phi_ref_used_cpu = phi_ref

    err_same_gpu = np.mean(np.abs(gpu_sdf - phi_ref))
    err_flip_gpu = np.mean(np.abs(gpu_sdf + phi_ref))
    if err_flip_gpu < err_same_gpu:
        phi_ref_used_gpu = -phi_ref
    else:
        phi_ref_used_gpu = phi_ref

    cpu_subset_mean_abs_err = float(np.mean(np.abs(cpu_sdf - phi_ref_used_cpu)))
    cpu_subset_rmse = float(np.sqrt(np.mean((cpu_sdf - phi_ref_used_cpu) ** 2)))
    gpu_subset_mean_abs_err = float(np.mean(np.abs(gpu_sdf - phi_ref_used_gpu)))
    gpu_subset_rmse = float(np.sqrt(np.mean((gpu_sdf - phi_ref_used_gpu) ** 2)))

    print(f"\nCPU/GPU 差异:")
    print(f"  max abs diff SDF: {cpu_gpu_max_abs_diff_sdf:.6e}")
    print(f"  mean abs diff SDF: {cpu_gpu_mean_abs_diff_sdf:.6e}")
    print(f"  p95 abs diff SDF: {cpu_gpu_p95_abs_diff_sdf:.6e}")
    print(f"  max abs diff unsigned: {cpu_gpu_max_abs_diff_unsigned:.6e}")
    print(f"  max abs diff nearest_points: {cpu_gpu_max_abs_diff_nearest:.6e}")
    print(f"  max abs diff normals: {cpu_gpu_max_abs_diff_normals:.6e}")
    print(f"  sign disagreement rate: {cpu_gpu_sign_disagreement:.6e}")

    print(f"\n各自对解析球误差 (子集):")
    print(f"  CPU mean_abs_err: {cpu_subset_mean_abs_err:.6e}")
    print(f"  CPU RMSE: {cpu_subset_rmse:.6e}")
    print(f"  GPU mean_abs_err: {gpu_subset_mean_abs_err:.6e}")
    print(f"  GPU RMSE: {gpu_subset_rmse:.6e}")

    if cpu_gpu_mean_abs_diff_sdf < basic["mean_abs_err"] / 10:
        print("=> CPU/GPU 差异远小于解析球总误差，GPU 实现不是主导因素。")

    return {
        "subset_n": subset_n,
        "cpu_gpu_max_abs_diff_sdf": cpu_gpu_max_abs_diff_sdf,
        "cpu_gpu_mean_abs_diff_sdf": cpu_gpu_mean_abs_diff_sdf,
        "cpu_gpu_p95_abs_diff_sdf": cpu_gpu_p95_abs_diff_sdf,
        "cpu_gpu_max_abs_diff_unsigned": cpu_gpu_max_abs_diff_unsigned,
        "cpu_gpu_max_abs_diff_nearest_points": cpu_gpu_max_abs_diff_nearest,
        "cpu_gpu_max_abs_diff_normals": cpu_gpu_max_abs_diff_normals,
        "cpu_gpu_sign_disagreement_rate": cpu_gpu_sign_disagreement,
        "cpu_subset_mean_abs_err": cpu_subset_mean_abs_err,
        "cpu_subset_rmse": cpu_subset_rmse,
        "gpu_subset_mean_abs_err": gpu_subset_mean_abs_err,
        "gpu_subset_rmse": gpu_subset_rmse,
    }


def step4_sign_sensitivity(basic):
    """步骤 4：误差来源 C -- 零水平集附近符号敏感性"""
    print("\n" + "=" * 60)
    print("步骤 4：误差来源 C -- 零水平集附近符号敏感性")
    print("=" * 60)

    sdf = basic["phi_ref_used"]  # 这里需要重新从 data 获取
    # 实际上我们需要从原始数据获取 sdf 和 phi_ref_used
    # 这里我们复用之前加载的 data
    return {}  # 由主函数统一处理


def step4_and_5_sign_mismatch_analysis(data, basic):
    """步骤 4+5：mismatch 近零分析 + 局部特征"""
    print("\n" + "=" * 60)
    print("步骤 4+5：mismatch 近零分析与局部特征")
    print("=" * 60)

    points = data["points"]
    sdf = data["sdf"]
    nearest_points = data["nearest_points"]
    normals = data["normals"]
    center_fit = np.array(basic["center_fit"])
    R_fit = basic["R_fit"]

    phi_ref = np.linalg.norm(points - center_fit[None, :], axis=1) - R_fit
    err_same = np.mean(np.abs(sdf - phi_ref))
    err_flip = np.mean(np.abs(sdf + phi_ref))

    if err_flip < err_same:
        phi_ref_used = -phi_ref
    else:
        phi_ref_used = phi_ref

    teacher_sign = np.where(sdf >= 0, 1, -1)
    ref_sign = np.where(phi_ref_used >= 0, 1, -1)
    mismatch_mask = teacher_sign != ref_sign

    mismatch_count = int(np.sum(mismatch_mask))
    mismatch_rate = mismatch_count / len(sdf) if len(sdf) > 0 else 0.0

    if mismatch_count == 0:
        print("无 sign mismatch，跳过近零分析。")
        return {
            "mismatch_count": 0,
            "mismatch_rate": 0.0,
            "frac_mismatch_abs_phi_lt_1e_6": 0.0,
            "frac_mismatch_abs_phi_lt_1e_5": 0.0,
            "frac_mismatch_abs_phi_lt_1e_4": 0.0,
            "frac_mismatch_abs_phi_lt_1e_3": 0.0,
            "filtered_sign_mismatch_rate_eps_1e_6": 0.0,
            "filtered_sign_mismatch_rate_eps_1e_5": 0.0,
            "filtered_sign_mismatch_rate_eps_1e_4": 0.0,
            "filtered_sign_mismatch_rate_eps_1e_3": 0.0,
            "mismatch_feature_code_hist": {},
            "mismatch_triangle_index_top20": [],
            "mismatch_block_coord_top20": [],
            "mismatch_surface_radial_err_mean": 0.0,
            "mismatch_surface_radial_err_p95": 0.0,
            "mismatch_normal_cos_mean": 0.0,
            "mismatch_normal_cos_p5": 0.0,
            "mismatch_normal_cos_min": 0.0,
        }

    mismatch_abs_phi = np.abs(phi_ref_used[mismatch_mask])
    mismatch_abs_err = np.abs(sdf[mismatch_mask] - phi_ref_used[mismatch_mask])

    fracs = {}
    for exp in [6, 5, 4, 3]:
        thresh = 10.0 ** (-exp)
        fracs[f"frac_mismatch_abs_phi_lt_1e_{exp}"] = float(np.mean(mismatch_abs_phi < thresh))

    # 过滤后的符号错误率
    filtered_rates = {}
    for exp in [6, 5, 4, 3]:
        thresh = 10.0 ** (-exp)
        valid_mask = np.abs(phi_ref_used) >= thresh
        if np.sum(valid_mask) == 0:
            filtered_rates[f"filtered_sign_mismatch_rate_eps_1e_{exp}"] = 0.0
        else:
            filtered_rates[f"filtered_sign_mismatch_rate_eps_1e_{exp}"] = float(
                np.sum(mismatch_mask & valid_mask) / np.sum(valid_mask)
            )

    # 局部特征
    if "feature_code" in data:
        fc = data["feature_code"][mismatch_mask]
        fc_hist = {str(k): int(np.sum(fc == k)) for k in [0, 1, 2, 3]}
    else:
        fc_hist = {}

    if "triangle_index" in data:
        ti = data["triangle_index"][mismatch_mask]
        ti_counts = {}
        for t in ti:
            ti_counts[str(int(t))] = ti_counts.get(str(int(t)), 0) + 1
        ti_top20 = sorted(ti_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    else:
        ti_top20 = []

    if "block_coord" in data:
        bc = data["block_coord"][mismatch_mask]
        bc_str = [f"({b[0]},{b[1]},{b[2]})" for b in bc]
        bc_counts = {}
        for b in bc_str:
            bc_counts[b] = bc_counts.get(b, 0) + 1
        bc_top20 = sorted(bc_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    else:
        bc_top20 = []

    nearest_r_mm = np.linalg.norm(nearest_points[mismatch_mask] - center_fit[None, :], axis=1)
    mismatch_surface_radial_err = np.abs(nearest_r_mm - R_fit)
    mismatch_sre_mean = float(np.mean(mismatch_surface_radial_err))
    mismatch_sre_p95 = float(np.percentile(mismatch_surface_radial_err, 95))

    denom_mm = np.linalg.norm(nearest_points[mismatch_mask] - center_fit[None, :], axis=1)
    n_true_mm = (nearest_points[mismatch_mask] - center_fit[None, :]) / denom_mm[:, None]
    normal_cos_mm = np.sum(normals[mismatch_mask] * n_true_mm, axis=1)
    nc_mean = float(np.mean(normal_cos_mm))
    nc_p5 = float(np.percentile(normal_cos_mm, 5))
    nc_min = float(np.min(normal_cos_mm))

    print(f"mismatch_count: {mismatch_count}")
    print(f"mismatch_rate: {mismatch_rate:.6e}")
    print(f"近零比例: <|1e-4={fracs['frac_mismatch_abs_phi_lt_1e_4']:.4f}, <|1e-3={fracs['frac_mismatch_abs_phi_lt_1e_3']:.4f}")
    print(f"过滤后错误率: eps=1e-4 => {filtered_rates['filtered_sign_mismatch_rate_eps_1e_4']:.6e}, eps=1e-3 => {filtered_rates['filtered_sign_mismatch_rate_eps_1e_3']:.6e}")
    print(f"mismatch 法向余弦: mean={nc_mean:.8f}, p5={nc_p5:.8f}, min={nc_min:.8f}")

    return {
        "mismatch_count": mismatch_count,
        "mismatch_rate": mismatch_rate,
        **fracs,
        **filtered_rates,
        "mismatch_feature_code_hist": fc_hist,
        "mismatch_triangle_index_top20": ti_top20,
        "mismatch_block_coord_top20": bc_top20,
        "mismatch_surface_radial_err_mean": mismatch_sre_mean,
        "mismatch_surface_radial_err_p95": mismatch_sre_p95,
        "mismatch_normal_cos_mean": nc_mean,
        "mismatch_normal_cos_p5": nc_p5,
        "mismatch_normal_cos_min": nc_min,
    }


def generate_final_interpretation(geo, cpu_gpu, mm):
    """步骤 6：最终自动结论"""
    print("\n" + "=" * 60)
    print("步骤 6：最终自动结论")
    print("=" * 60)

    parts = []

    # A: GPU/CPU 差异
    if cpu_gpu["cpu_gpu_mean_abs_diff_sdf"] < 5.388266e-05 / 10:
        parts.append("GPU/CPU 数值实现差异不是当前解析球总误差的主导来源。")
        print("A: GPU/CPU 差异非主导因素 PASS")

    # B: 几何逼近误差
    if abs(geo["surface_radial_err_mean"] - 5.388266e-05) / max(5.388266e-05, 1e-15) < 0.1:
        parts.append("当前解析球误差主要受教师曲面相对解析球的几何逼近误差影响。")
        print("B: 几何逼近误差是主导 PASS")

    # C: sign mismatch
    if mm.get("frac_mismatch_abs_phi_lt_1e_4", 0) > 0.5 or mm.get("filtered_sign_mismatch_rate_eps_1e_4", 1) < 1e-4:
        parts.append("sign mismatch 主要集中在零水平集附近，属于数值敏感区的局部翻转，而非全局符号错误。")
        print("C: sign mismatch 近零集中 PASS")

    if len(parts) >= 3:
        final = (
            "当前 sphere 误差主要来自增强 Nagata 教师曲面相对解析球的高精度但非零几何逼近误差；"
            "GPU 实现差异不是主导因素；"
            "少量 sign mismatch 集中在零水平集附近，属于近表面数值敏感性。"
        )
    else:
        final = " ".join(parts) if parts else "需要进一步调查。"

    print(f"\n最终结论: {final}")
    return final


def main():
    print("=" * 60)
    print("Sphere 误差来源分解实验")
    print("=" * 60)

    data, npz_path = load_teacher_npz()

    basic = step1_basic_verification(data, npz_path)
    geo = step2_geometric_approximation_error(data, basic)
    cpu_gpu = step3_cpu_gpu_comparison(data, basic)
    mm = step4_and_5_sign_mismatch_analysis(data, basic)
    final_interpretation = generate_final_interpretation(geo, cpu_gpu, mm)

    # 写入 JSON
    report = {
        "sphere_npz_path": npz_path,
        "center_fit": basic["center_fit"],
        "R_fit": basic["R_fit"],
        "R_std": basic["R_std"],
        "chosen_sign_convention": basic["chosen_sign_convention"],
        "total_samples": basic["total_samples"],
        "mean_abs_err": basic["mean_abs_err"],
        "rmse": basic["rmse"],
        "p95_abs_err": basic["p95_abs_err"],
        "max_abs_err": basic["max_abs_err"],
        "sign_mismatch_rate": basic["sign_mismatch_rate"],
        **geo,
        **cpu_gpu,
        **mm,
        "final_interpretation": final_interpretation,
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/sphere_error_decomposition_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nJSON 报告已写出: outputs/sphere_error_decomposition_report.json")

    # 终端打印摘要
    print("\n" + "=" * 60)
    print("摘要")
    print("=" * 60)
    print(f"mean_abs_err: {basic['mean_abs_err']:.6e}")
    print(f"surface_radial_err_mean: {geo['surface_radial_err_mean']:.6e}")
    print(f"corr_surface_vs_distance_err: {geo['corr_surface_vs_distance_err']:.6f}")
    print(f"cpu_gpu_mean_abs_diff_sdf: {cpu_gpu['cpu_gpu_mean_abs_diff_sdf']:.6e}")
    print(f"cpu_gpu_sign_disagreement_rate: {cpu_gpu['cpu_gpu_sign_disagreement_rate']:.6e}")
    print(f"frac_mismatch_abs_phi_lt_1e_4: {mm.get('frac_mismatch_abs_phi_lt_1e_4', 0):.6f}")
    print(f"filtered_sign_mismatch_rate_eps_1e_4: {mm.get('filtered_sign_mismatch_rate_eps_1e_4', 0):.6e}")
    print(f"\nfinal_interpretation: {final_interpretation}")


if __name__ == "__main__":
    main()
