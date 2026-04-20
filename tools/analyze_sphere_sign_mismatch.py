"""
分析 sphere 上的 sign mismatch。

目标：确认当前 sphere 的 sign mismatch 是否几乎全部集中在零水平集附近。
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from nagata_teacher_runtime.enhanced_nagata_backend_torch import load_nsm_lightweight

FEATURE_CODE = {"FACE": 0, "EDGE": 1, "SHARPEDGE": 2, "VERTEX": 3}


def load_sphere_npz():
    """加载已有的 sphere 教师场结果"""
    candidates = [
        "outputs/sphere_teacher_parallel.npz",
        "outputs/sphere_teacher_gpu.npz",
        "outputs/sphere_teacher.npz",
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"使用已有结果: {path}")
            return np.load(path, allow_pickle=True), path
    raise FileNotFoundError("找不到任何 sphere 教师场结果文件")


def estimate_sphere_params():
    """拟合球参数"""
    mesh = load_nsm_lightweight("models/sphere.nsm")
    vertices = mesh.vertices

    center_fit = vertices.mean(axis=0)
    radii = np.linalg.norm(vertices - center_fit[None, :], axis=1)
    R_fit = float(radii.mean())
    R_std = float(radii.std())

    print(f"center_fit: {center_fit}")
    print(f"R_fit: {R_fit}")
    print(f"R_std: {R_std}")
    print(f"R_std / R_fit: {R_std / R_fit}")

    return center_fit, R_fit, R_std


def analyze_sign_mismatch(data, center_fit, R_fit):
    """分析 sign mismatch 样本"""
    points = data["points"]
    phi_teacher = data["sdf"]
    nearest_points = data["nearest_points"]
    normals = data["normals"]

    phi_ref = np.linalg.norm(points - center_fit[None, :], axis=1) - R_fit

    err_same = np.mean(np.abs(phi_teacher - phi_ref))
    err_flip = np.mean(np.abs(phi_teacher + phi_ref))

    if err_flip < err_same:
        phi_ref_used = -phi_ref
        chosen_sign_convention = "flipped"
    else:
        phi_ref_used = phi_ref
        chosen_sign_convention = "same"

    teacher_sign = np.where(phi_teacher >= 0, 1, -1)
    ref_sign = np.where(phi_ref_used >= 0, 1, -1)
    mismatch_mask = teacher_sign != ref_sign

    total_samples = len(phi_teacher)
    mismatch_count = int(np.sum(mismatch_mask))
    mismatch_rate = mismatch_count / total_samples if total_samples > 0 else 0.0

    print(f"\n总样本数: {total_samples}")
    print(f"mismatch 数量: {mismatch_count}")
    print(f"mismatch 率: {mismatch_rate:.6f}")
    print(f"符号约定: {chosen_sign_convention}")

    if mismatch_count == 0:
        print("\n无 sign mismatch，无需进一步分析。")
        return None

    mismatch_abs_phi_ref = np.abs(phi_ref_used[mismatch_mask])
    mismatch_abs_err = np.abs(phi_teacher[mismatch_mask] - phi_ref_used[mismatch_mask])

    # 零水平集附近分布
    frac_lt_1e6 = float(np.mean(mismatch_abs_phi_ref < 1e-6))
    frac_lt_1e5 = float(np.mean(mismatch_abs_phi_ref < 1e-5))
    frac_lt_1e4 = float(np.mean(mismatch_abs_phi_ref < 1e-4))
    frac_lt_1e3 = float(np.mean(mismatch_abs_phi_ref < 1e-3))

    print(f"\nmismatch |phi_ref| 分布:")
    print(f"  mean: {float(np.mean(mismatch_abs_phi_ref)):.6e}")
    print(f"  p50: {float(np.percentile(mismatch_abs_phi_ref, 50)):.6e}")
    print(f"  p95: {float(np.percentile(mismatch_abs_phi_ref, 95)):.6e}")
    print(f"  max: {float(np.max(mismatch_abs_phi_ref)):.6e}")
    print(f"  <|1e-6: {frac_lt_1e6:.6f}")
    print(f"  <|1e-5: {frac_lt_1e5:.6f}")
    print(f"  <|1e-4: {frac_lt_1e4:.6f}")
    print(f"  <|1e-3: {frac_lt_1e3:.6f}")

    print(f"\nmismatch 绝对误差:")
    print(f"  mean: {float(np.mean(mismatch_abs_err)):.6e}")
    print(f"  p95: {float(np.percentile(mismatch_abs_err, 95)):.6e}")
    print(f"  max: {float(np.max(mismatch_abs_err)):.6e}")

    # 法向余弦
    denom = np.linalg.norm(nearest_points[mismatch_mask] - center_fit[None, :], axis=1)
    n_true = (nearest_points[mismatch_mask] - center_fit[None, :]) / denom[:, None]
    normal_cos = np.sum(normals[mismatch_mask] * n_true, axis=1)

    print(f"\nmismatch 法向余弦:")
    print(f"  mean: {float(np.mean(normal_cos)):.8f}")
    print(f"  p5: {float(np.percentile(normal_cos, 5)):.8f}")
    print(f"  min: {float(np.min(normal_cos)):.8f}")

    # feature_code 分布
    if "feature_code" in data:
        fc = data["feature_code"][mismatch_mask]
        fc_hist = {str(k): int(np.sum(fc == k)) for k in FEATURE_CODE.values()}
    else:
        fc_hist = {}

    # block_coord 分布
    if "block_coord" in data:
        bc = data["block_coord"][mismatch_mask]
        bc_str = [f"({b[0]},{b[1]},{b[2]})" for b in bc]
        bc_counts = {}
        for b in bc_str:
            bc_counts[b] = bc_counts.get(b, 0) + 1
        bc_top20 = sorted(bc_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    else:
        bc_top20 = []

    # triangle_index 分布
    if "triangle_index" in data:
        ti = data["triangle_index"][mismatch_mask]
        ti_counts = {}
        for t in ti:
            ti_counts[str(int(t))] = ti_counts.get(str(int(t)), 0) + 1
        ti_top20 = sorted(ti_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    else:
        ti_top20 = []

    # 判定结论
    if frac_lt_1e4 > 0.5 or frac_lt_1e3 > 0.9:
        interpretation = (
            "sphere 的 sign mismatch 主要集中在零水平集附近，"
            "属于数值敏感区的局部翻转，不构成全局符号错误。"
        )
    else:
        interpretation = (
            "sign mismatch 分布较广，需要进一步调查是否为系统性问题。"
        )

    print(f"\n结论: {interpretation}")

    return {
        "mismatch_count": mismatch_count,
        "mismatch_rate": mismatch_rate,
        "mismatch_abs_phi_mean": float(np.mean(mismatch_abs_phi_ref)),
        "mismatch_abs_phi_p50": float(np.percentile(mismatch_abs_phi_ref, 50)),
        "mismatch_abs_phi_p95": float(np.percentile(mismatch_abs_phi_ref, 95)),
        "mismatch_abs_phi_max": float(np.max(mismatch_abs_phi_ref)),
        "mismatch_abs_err_mean": float(np.mean(mismatch_abs_err)),
        "mismatch_abs_err_p95": float(np.percentile(mismatch_abs_err, 95)),
        "mismatch_abs_err_max": float(np.max(mismatch_abs_err)),
        "frac_mismatch_abs_phi_lt_1e_6": frac_lt_1e6,
        "frac_mismatch_abs_phi_lt_1e_5": frac_lt_1e5,
        "frac_mismatch_abs_phi_lt_1e_4": frac_lt_1e4,
        "frac_mismatch_abs_phi_lt_1e_3": frac_lt_1e3,
        "mismatch_feature_code_hist": fc_hist,
        "mismatch_block_coord_count_top20": bc_top20,
        "mismatch_triangle_index_count_top20": ti_top20,
        "mismatch_normal_cos_mean": float(np.mean(normal_cos)),
        "mismatch_normal_cos_p5": float(np.percentile(normal_cos, 5)),
        "mismatch_normal_cos_min": float(np.min(normal_cos)),
        "final_interpretation": interpretation,
    }


def main():
    print("=" * 60)
    print("Sphere Sign Mismatch 分析")
    print("=" * 60)

    data, npz_path = load_sphere_npz()
    center_fit, R_fit, R_std = estimate_sphere_params()
    result = analyze_sign_mismatch(data, center_fit, R_fit)

    if result is None:
        return

    # 写入 JSON 报告
    report = {
        "sphere_npz_path": npz_path,
        "center_fit": center_fit.tolist(),
        "R_fit": R_fit,
        "R_std": R_std,
        "total_samples": int(data["points"].shape[0]),
        **result,
    }

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/sphere_sign_mismatch_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nJSON 报告已写出: outputs/sphere_sign_mismatch_report.json")


if __name__ == "__main__":
    main()
