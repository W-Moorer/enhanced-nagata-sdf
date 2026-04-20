"""
导出论文/汇报用验证图。

基于已有 npz/json 输出，生成 publication-quality 图表。
不重新生成教师场，只读取已有结果。
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
})

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from nagata_teacher_runtime.enhanced_nagata_backend_torch import load_nsm_lightweight

FIG_DIR = _PROJECT_DIR / "outputs" / "figures"


def check_file(path: str) -> Path:
    """检查文件是否存在"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"输入文件不存在: {p}")
    return p


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_npz(path: str):
    return np.load(path, allow_pickle=True)


def fig1_sphere_error_hist(sphere_data: dict, sphere_npz) -> Path:
    """图 1：sphere abs_err 直方图"""
    points = sphere_npz["points"]
    phi_teacher = sphere_npz["sdf"]
    center = np.array(sphere_data["center_fit"])
    R = sphere_data["R_fit"]

    phi_ref = np.linalg.norm(points - center[None, :], axis=1) - R
    err_same = np.mean(np.abs(phi_teacher - phi_ref))
    err_flip = np.mean(np.abs(phi_teacher + phi_ref))

    if err_flip < err_same:
        phi_ref_used = -phi_ref
    else:
        phi_ref_used = phi_ref

    abs_err = np.abs(phi_teacher - phi_ref_used)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(abs_err, bins=200, log=True, edgecolor="none", alpha=0.85)
    ax.set_xlabel("Absolute Error", fontsize=12)
    ax.set_ylabel("Count (log scale)", fontsize=12)
    ax.set_title("Sphere Teacher Field: Absolute Error Distribution", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    # 添加统计线
    mean_val = float(np.mean(abs_err))
    p95_val = float(np.percentile(abs_err, 95))
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2e}")
    ax.axvline(p95_val, color="orange", linestyle="--", linewidth=1.5, label=f"P95: {p95_val:.2e}")
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = FIG_DIR / "sphere_error_hist.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig2_sign_mismatch_scatter(mismatch_data: dict, sphere_npz) -> Path:
    """图 2：sign mismatch 散点图，按 |phi_ref| 着色"""
    if mismatch_data["mismatch_count"] == 0:
        print("无 sign mismatch 样本，跳过图 2。")
        return None

    sphere_npz_path = check_file("outputs/sphere_teacher_gpu.npz")
    sphere_data = load_json("outputs/sphere_teacher_gpu_report.json")
    npz = load_npz(str(sphere_npz_path))

    points = npz["points"]
    phi_teacher = npz["sdf"]
    center = np.array(sphere_data["center_fit"])
    R = sphere_data["R_fit"]

    phi_ref = np.linalg.norm(points - center[None, :], axis=1) - R
    err_same = np.mean(np.abs(phi_teacher - phi_ref))
    err_flip = np.mean(np.abs(phi_teacher + phi_ref))

    if err_flip < err_same:
        phi_ref_used = -phi_ref
    else:
        phi_ref_used = phi_ref

    teacher_sign = np.where(phi_teacher >= 0, 1, -1)
    ref_sign = np.where(phi_ref_used >= 0, 1, -1)
    mismatch_mask = teacher_sign != ref_sign

    mismatch_pts = points[mismatch_mask]
    mismatch_phi = np.abs(phi_ref_used[mismatch_mask])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        mismatch_pts[:, 0], mismatch_pts[:, 1], mismatch_pts[:, 2],
        c=mismatch_phi, cmap="viridis", s=15, alpha=0.8, edgecolors="none"
    )
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.set_zlabel("Z", fontsize=11)
    ax.set_title("Sphere Sign Mismatch Points (colored by |phi_ref|)", fontsize=12)
    cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("|phi_ref|", fontsize=11)

    plt.tight_layout()
    out = FIG_DIR / "sphere_sign_mismatch_scatter.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig3_radial_error_hist(sphere_npz, sphere_data) -> Path:
    """图 3：最近点径向误差分布"""
    points = sphere_npz["points"]
    nearest_points = sphere_npz["nearest_points"]
    center = np.array(sphere_data["center_fit"])
    R = sphere_data["R_fit"]

    radial_err = np.abs(np.linalg.norm(nearest_points - center[None, :], axis=1) - R)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(radial_err, bins=200, log=True, edgecolor="none", alpha=0.85, color="steelblue")
    ax.set_xlabel("Radial Error of Nearest Points", fontsize=12)
    ax.set_ylabel("Count (log scale)", fontsize=12)
    ax.set_title("Sphere Teacher Field: Nearest Point Radial Error", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    mean_val = float(np.mean(radial_err))
    max_val = float(np.max(radial_err))
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2e}")
    ax.axvline(max_val, color="orange", linestyle="--", linewidth=1.5, label=f"Max: {max_val:.2e}")
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = FIG_DIR / "sphere_nearest_radial_error_hist.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig4_cone_sample_distribution(cone_npz) -> Path:
    """图 4：cone 教师样本点云，按 |sdf| 着色"""
    points = cone_npz["points"]
    sdf = cone_npz["sdf"]
    abs_sdf = np.abs(sdf)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=abs_sdf, cmap="coolwarm", s=3, alpha=0.7, edgecolors="none"
    )
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.set_zlabel("Z", fontsize=11)
    ax.set_title("Cone Teacher Samples (colored by |SDF|)", fontsize=12)
    cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("|SDF|", fontsize=11)

    plt.tight_layout()
    out = FIG_DIR / "cone_sample_distribution.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig5_cone_feature_code_hist(cone_data: dict) -> Path:
    """图 5：cone feature_code 直方图"""
    fc_hist = cone_data.get("feature_code_hist", {})
    if not fc_hist:
        print("无 feature_code 数据，跳过图 5。")
        return None

    names = ["FACE", "EDGE", "SHARPEDGE", "VERTEX"]
    codes = ["0", "1", "2", "3"]
    counts = [int(fc_hist.get(c, 0)) for c in codes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, counts, color=["steelblue", "orange", "green", "red"], edgecolor="black", alpha=0.85)
    ax.set_xlabel("Feature Code", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Cone Teacher Field: Feature Code Distribution", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                f"{count:,}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out = FIG_DIR / "cone_feature_code_distribution.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig6_cone_report_summary(cone_data: dict) -> Path:
    """图 6：cone 报告文本摘要"""
    lines = [
        "=" * 60,
        "Cone GPU Teacher Field Verification Summary",
        "=" * 60,
        f"num_crease_edges: {cone_data.get('num_crease_edges', 'N/A')}",
        f"kept_samples: {cone_data.get('kept_samples', 'N/A'):,}",
        f"kept_ratio: {cone_data.get('kept_ratio', 'N/A'):.4f}",
        f"total_wall_time: {cone_data.get('total_wall_time', 'N/A'):.2f}s",
        f"abs_sdf_minus_unsigned_mean: {cone_data.get('abs_sdf_minus_unsigned_mean', 'N/A'):.6e}",
        f"abs_sdf_minus_unsigned_p95: {cone_data.get('abs_sdf_minus_unsigned_p95', 'N/A'):.6e}",
        "",
        "final_interpretation:",
        f"  {cone_data.get('final_interpretation', 'N/A')}",
        "",
        "=" * 60,
    ]
    text = "\n".join(lines)

    out = FIG_DIR / "cone_report_summary.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    return out


def main():
    print("=" * 60)
    print("导出论文/汇报用验证图")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 检查输入文件
    sphere_json_path = check_file("outputs/sphere_teacher_gpu_report.json")
    mismatch_json_path = check_file("outputs/sphere_sign_mismatch_report.json")
    cone_json_path = check_file("outputs/sharp_teacher_gpu_report.json")

    if os.path.exists("outputs/sphere_teacher_gpu.npz"):
        sphere_npz_path = check_file("outputs/sphere_teacher_gpu.npz")
    else:
        sphere_npz_path = check_file("outputs/sphere_teacher_parallel.npz")
    cone_npz_path = check_file("outputs/sharp_teacher_gpu.npz")

    sphere_data = load_json(str(sphere_json_path))
    mismatch_data = load_json(str(mismatch_json_path))
    cone_data = load_json(str(cone_json_path))
    sphere_npz = load_npz(str(sphere_npz_path))
    cone_npz = load_npz(str(cone_npz_path))

    out_files = []

    print("\n生成图 1: sphere_error_hist.pdf ...")
    out_files.append(fig1_sphere_error_hist(sphere_data, sphere_npz))

    print("生成图 2: sphere_sign_mismatch_scatter.pdf ...")
    r = fig2_sign_mismatch_scatter(mismatch_data, sphere_npz)
    if r:
        out_files.append(r)

    print("生成图 3: sphere_nearest_radial_error_hist.pdf ...")
    out_files.append(fig3_radial_error_hist(sphere_npz, sphere_data))

    print("生成图 4: cone_sample_distribution.pdf ...")
    out_files.append(fig4_cone_sample_distribution(cone_npz))

    print("生成图 5: cone_feature_code_distribution.pdf ...")
    r = fig5_cone_feature_code_hist(cone_data)
    if r:
        out_files.append(r)

    print("生成图 6: cone_report_summary.txt ...")
    out_files.append(fig6_cone_report_summary(cone_data))

    print("\n" + "=" * 60)
    print("导出完成，输出文件:")
    print("=" * 60)
    for f in out_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
