"""
GPU Enhanced Nagata Sparse SDF Backend - PyTorch 实现

设计原则:
1. CPU 负责预处理 (读取 NSM、加载 ENG 缓存、构造 patch 数据)
2. GPU 负责批量查询 (最近点 Newton 迭代、SDF 计算)
3. 输出格式与 CPU 版完全兼容

数学定义与 CPU 版严格一致:
- Nagata 曲面: x(u,v) = x00*(1-u) + x10*(u-v) + x11*v - c1*(1-u)*(u-v) - c2*(u-v)*v - c3*(1-u)*v
- 折痕修复: 使用高斯衰减 exp(-k*d^2) 修正折痕边
- 最近点求解: Newton-Raphson 多起点迭代 + 边界兜底
"""

from __future__ import annotations

import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# 允许脚本在独立目录中运行，同时引用用户现有代码文件。
from .nagata_storage import get_eng_filepath, has_cached_data, load_enhanced_data, save_enhanced_data


# =============================================================================
# NSM 读取 (复用 CPU 版逻辑)
# =============================================================================

@dataclass
class NSMMeshDataLite:
    """精简版 NSM 网格数据 (无 PyVista 依赖)"""
    vertices: np.ndarray           # (num_vertices, 3)
    triangles: np.ndarray          # (num_triangles, 3)
    tri_face_ids: np.ndarray       # (num_triangles,)
    tri_vertex_normals: np.ndarray # (num_triangles, 3, 3)


def load_nsm_lightweight(filepath: str) -> NSMMeshDataLite:
    """
    读取 NSM 文件，返回精简网格数据。

    参数:
        filepath: NSM 文件路径

    返回:
        NSMMeshDataLite: 精简网格数据
    """
    with open(filepath, "rb") as f:
        header_data = f.read(64)
        if len(header_data) != 64:
            raise ValueError(f"文件头不完整: {len(header_data)}")

        magic = header_data[0:4].decode("ascii", errors="ignore")
        version = struct.unpack("<I", header_data[4:8])[0]
        num_vertices = struct.unpack("<I", header_data[8:12])[0]
        num_triangles = struct.unpack("<I", header_data[12:16])[0]

        if magic != "NSM\x00":
            raise ValueError(f"无效的 NSM magic: {repr(magic)}")
        if version != 1:
            raise ValueError(f"不支持的 NSM 版本: {version}")

        vertices = np.fromfile(f, dtype=np.float64, count=num_vertices * 3).reshape(num_vertices, 3)
        triangles = np.fromfile(f, dtype=np.uint32, count=num_triangles * 3).reshape(num_triangles, 3)
        tri_face_ids = np.fromfile(f, dtype=np.uint32, count=num_triangles)
        tri_vertex_normals = np.fromfile(f, dtype=np.float64, count=num_triangles * 3 * 3).reshape(num_triangles, 3, 3)

    return NSMMeshDataLite(
        vertices=vertices,
        triangles=triangles,
        tri_face_ids=tri_face_ids,
        tri_vertex_normals=tri_vertex_normals,
    )


# =============================================================================
# 折痕检测 (复用 CPU 版逻辑，由 CPU 执行)
# =============================================================================

def _compute_curvature_cpu(d: np.ndarray, n0: np.ndarray, n1: np.ndarray) -> np.ndarray:
    """
    计算 Nagata 曲率系数向量 (CPU 版本)。

    参数:
        d: 方向向量 (x1 - x0), shape (3,)
        n0: 起点法向量, shape (3,)
        n1: 终点法向量, shape (3,)

    返回:
        cvec: 曲率系数向量, shape (3,)
    """
    angle_tol = np.cos(0.1 * np.pi / 180)

    v = 0.5 * (n0 + n1)
    delta_v = 0.5 * (n0 - n1)
    dv = float(np.dot(d, v))
    d_delta_v = float(np.dot(d, delta_v))
    delta_c = float(np.dot(n0, delta_v))
    c = 1.0 - 2.0 * delta_c

    if abs(c) > angle_tol:
        return np.zeros(3, dtype=np.float64)

    denom1 = 1.0 - delta_c
    denom2 = delta_c
    if abs(denom1) < 1e-12 or abs(denom2) < 1e-12:
        return np.zeros(3, dtype=np.float64)

    return (d_delta_v / denom1) * v + (dv / denom2) * delta_v


def _compute_crease_direction_cpu(n_L: np.ndarray, n_R: np.ndarray, e: np.ndarray) -> np.ndarray:
    """
    计算折痕切向单位方向 (CPU 版本)。

    参数:
        n_L: 左侧法向量
        n_R: 右侧法向量
        e: 边向量 (B - A)

    返回:
        d: 折痕切向单位方向
    """
    cross = np.cross(n_L, n_R)
    norm = float(np.linalg.norm(cross))
    if norm < 1e-10:
        d = e / np.linalg.norm(e)
    else:
        d = cross / norm
    if float(np.dot(d, e)) < 0:
        d = -d
    return d


def _compute_c_sharp_cpu(A: np.ndarray, B: np.ndarray, d_A: np.ndarray, d_B: np.ndarray) -> np.ndarray:
    """
    计算共享边界系数 c^sharp (CPU 版本)。

    参数:
        A, B: 端点坐标
        d_A, d_B: 折痕切向单位方向

    返回:
        c_sharp: 共享边界系数
    """
    e = B - A
    G = np.array([
        [np.dot(d_A, d_A), np.dot(d_A, d_B)],
        [np.dot(d_A, d_B), np.dot(d_B, d_B)]
    ])
    r = np.array([2 * np.dot(e, d_A), 2 * np.dot(e, d_B)])
    G_reg = G + 1e-6 * np.eye(2)
    try:
        ell = np.linalg.solve(G_reg, r)
    except np.linalg.LinAlgError:
        ell = np.array([np.linalg.norm(e), np.linalg.norm(e)])

    T_A = ell[0] * d_A
    T_B = ell[1] * d_B
    c_sharp = (T_B - T_A) / 2.0

    c_norm = float(np.linalg.norm(c_sharp))
    max_c = 2.0 * np.linalg.norm(e)
    if c_norm > max_c:
        c_sharp = c_sharp * (max_c / c_norm)
    return c_sharp


def detect_crease_edges_cpu(
    vertices: np.ndarray,
    triangles: np.ndarray,
    tri_vertex_normals: np.ndarray,
    gap_threshold: float = 1e-4,
) -> Dict[Tuple[int, int], dict]:
    """
    检测折痕边 (CPU 版本，与 CPU 后端一致)。

    参数:
        vertices: 网格顶点
        triangles: 三角形索引
        tri_vertex_normals: 顶点法向量
        gap_threshold: 裂隙检测阈值

    返回:
        折痕边字典
    """
    from collections import defaultdict

    edge_to_tris = defaultdict(list)
    for tri_idx, tri in enumerate(triangles):
        edges = [
            (tri[0], tri[1], 0, 1),
            (tri[1], tri[2], 1, 2),
            (tri[0], tri[2], 0, 2),
        ]
        for v0, v1, local0, local1 in edges:
            edge_key = tuple(sorted([int(v0), int(v1)]))
            edge_to_tris[edge_key].append((tri_idx, int(v0), int(v1), local0, local1))

    crease_edges: Dict[Tuple[int, int], dict] = {}

    def get_normal_at_vertex(tri_idx: int, global_v_idx: int) -> Optional[np.ndarray]:
        tri = triangles[tri_idx]
        for local_idx in range(3):
            if int(tri[local_idx]) == int(global_v_idx):
                return tri_vertex_normals[tri_idx, local_idx]
        return None

    for edge_key, tris_info in edge_to_tris.items():
        if len(tris_info) != 2:
            continue

        tri_L_info, tri_R_info = tris_info
        tri_L, *_ = tri_L_info
        tri_R, *_ = tri_R_info
        A_idx, B_idx = edge_key
        A = vertices[A_idx]
        B = vertices[B_idx]

        n_A_L = get_normal_at_vertex(tri_L, A_idx)
        n_B_L = get_normal_at_vertex(tri_L, B_idx)
        n_A_R = get_normal_at_vertex(tri_R, A_idx)
        n_B_R = get_normal_at_vertex(tri_R, B_idx)
        if n_A_L is None or n_B_L is None or n_A_R is None or n_B_R is None:
            continue

        e = B - A
        c_L = _compute_curvature_cpu(e, n_A_L, n_B_L)
        c_R = _compute_curvature_cpu(e, n_A_R, n_B_R)

        max_gap = 0.0
        for t in np.linspace(0.0, 1.0, 11):
            p_L = (1.0 - t) * A + t * B - c_L * t * (1.0 - t)
            p_R = (1.0 - t) * A + t * B - c_R * t * (1.0 - t)
            max_gap = max(max_gap, float(np.linalg.norm(p_L - p_R)))

        if max_gap > gap_threshold:
            crease_edges[edge_key] = {
                "A": A, "B": B,
                "n_A_L": n_A_L, "n_A_R": n_A_R,
                "n_B_L": n_B_L, "n_B_R": n_B_R,
                "tri_L": int(tri_L), "tri_R": int(tri_R),
                "max_gap": max_gap,
            }

    return crease_edges


def _edge_index_for_triangle(tri: np.ndarray, edge_key: Tuple[int, int]) -> int:
    """获取边在三角形中的局部索引。"""
    if tuple(sorted([int(tri[0]), int(tri[1])])) == edge_key:
        return 0
    if tuple(sorted([int(tri[1]), int(tri[2])])) == edge_key:
        return 1
    if tuple(sorted([int(tri[0]), int(tri[2])])) == edge_key:
        return 2
    return -1


def _sample_uv_for_edge(edge_idx: int, steps: int = 5, eps: float = 0.05) -> List[Tuple[float, float]]:
    """在边上采样 UV 坐标。"""
    ts = np.linspace(eps, 1.0 - eps, steps)
    uvs: List[Tuple[float, float]] = []
    if edge_idx == 0:
        for t in ts:
            uvs.append((float(t), float(eps)))
    elif edge_idx == 1:
        for t in ts:
            uvs.append((float(1.0 - eps), float(t * (1.0 - eps))))
    elif edge_idx == 2:
        for t in ts:
            uvs.append((float(t), float(t - eps)))
    return uvs


def _eval_nagata_patch_point(x00, x10, x11, c1, c2, c3, u, v):
    """评估 Nagata 曲面单点 (numpy)。"""
    one_minus_u = 1.0 - u
    u_minus_v = u - v
    linear = x00 * one_minus_u + x10 * u_minus_v + x11 * v
    quadratic = c1 * (one_minus_u * u_minus_v) + c2 * (u_minus_v * v) + c3 * (one_minus_u * v)
    return linear - quadratic


def _eval_nagata_derivatives(x00, x10, x11, c1, c2, c3, u, v,
                              is_crease=(False, False, False),
                              c_sharps=(None, None, None),
                              k_factor=0.0):
    """评估 Nagata 曲面导数 (numpy)，支持折痕修复。"""
    one = 1.0
    dLin_du = -x00 + x10
    dLin_dv = -x10 + x11

    d_params = [v, one - u, u - v]
    dd_du = [0.0, -1.0, 1.0]
    dd_dv = [1.0, 0.0, -1.0]

    coeffs = [c1, c2, c3]
    sharp_coeffs = [c if s is None else s for c, s in zip(coeffs, c_sharps)]

    b1 = (1.0 - u) * (u - v)
    db1_du = 1.0 - 2.0 * u + v
    db1_dv = u - 1.0

    b2 = (u - v) * v
    db2_du = v
    db2_dv = u - 2.0 * v

    b3 = (1.0 - u) * v
    db3_du = -v
    db3_dv = 1.0 - u

    bases = [b1, b2, b3]
    db_du_list = [db1_du, db2_du, db3_du]
    db_dv_list = [db1_dv, db2_dv, db3_dv]

    dQ_du = np.zeros(3)
    dQ_dv = np.zeros(3)
    for i in range(3):
        dQ_du += coeffs[i] * db_du_list[i]
        dQ_dv += coeffs[i] * db_dv_list[i]

    dCorr_du = np.zeros(3)
    dCorr_dv = np.zeros(3)
    for i in range(3):
        if not is_crease[i]:
            continue
        dist = d_params[i]
        delta_c = sharp_coeffs[i] - coeffs[i]
        damping = np.exp(-k_factor * dist * dist)
        ddamping_dd = -2.0 * k_factor * dist * damping
        ddamping_du = ddamping_dd * dd_du[i]
        ddamping_dv = ddamping_dd * dd_dv[i]
        dCorr_du -= delta_c * (db_du_list[i] * damping + bases[i] * ddamping_du)
        dCorr_dv -= delta_c * (db_dv_list[i] * damping + bases[i] * ddamping_dv)

    dXdu = dLin_du - dQ_du + dCorr_du
    dXdv = dLin_dv - dQ_dv + dCorr_dv
    return dXdu, dXdv


def _check_edge_constraints_for_triangle_cpu(
    x00, x10, x11, c1, c2, c3, edge_idx, c_sharp_edge, k_factor, eps=1e-10,
) -> bool:
    """检查边约束 (CPU 版本)。"""
    n_ref = np.cross(x10 - x00, x11 - x00)
    n_len = float(np.linalg.norm(n_ref))
    if n_len < 1e-12:
        return True
    n_ref = n_ref / n_len

    is_crease = [False, False, False]
    c_sharps = [c1, c2, c3]
    is_crease[edge_idx] = True
    c_sharps[edge_idx] = c_sharp_edge

    if edge_idx == 0:
        a, b, opp = x00, x10, x11
    elif edge_idx == 1:
        a, b, opp = x10, x11, x00
    else:
        a, b, opp = x00, x11, x10

    e = b - a
    side_dir = np.cross(n_ref, e)
    side_len = float(np.linalg.norm(side_dir))
    if side_len < 1e-12:
        return True
    side_dir = side_dir / side_len
    if float(np.dot(side_dir, opp - 0.5 * (a + b))) < 0.0:
        side_dir = -side_dir

    for uu, vv in _sample_uv_for_edge(edge_idx):
        p = _eval_nagata_patch_point(x00, x10, x11, c1, c2, c3, uu, vv)
        dXdu, dXdv = _eval_nagata_derivatives(
            x00, x10, x11, c1, c2, c3, uu, vv,
            is_crease=tuple(is_crease), c_sharps=tuple(c_sharps), k_factor=k_factor,
        )
        jac = float(np.dot(np.cross(dXdu, dXdv), n_ref))
        if jac <= 0.0:
            return False
        t = uu if edge_idx == 0 else vv
        edge_point = (1.0 - t) * a + t * b - c_sharp_edge * t * (1.0 - t)
        s = float(np.dot(p - edge_point, side_dir))
        if s < -eps:
            return False
    return True


def compute_c_sharps_for_edges_cpu(
    crease_edges: Dict[Tuple[int, int], dict],
    vertices: np.ndarray,
    triangles: np.ndarray,
    tri_vertex_normals: np.ndarray,
    k_factor: float = 0.0,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    计算所有折痕边的 c_sharp 值 (CPU 版本)。

    参数:
        crease_edges: 折痕边字典
        vertices: 网格顶点
        triangles: 三角形索引
        tri_vertex_normals: 顶点法向量
        k_factor: 高斯衰减参数

    返回:
        c_sharps: 折痕边到共享系数的映射
    """
    c_sharps: Dict[Tuple[int, int], np.ndarray] = {}

    for edge_key, info in crease_edges.items():
        A, B = info["A"], info["B"]
        e = B - A

        d_A = _compute_crease_direction_cpu(info["n_A_L"], info["n_A_R"], e)
        d_B = _compute_crease_direction_cpu(info["n_B_L"], info["n_B_R"], e)
        if float(np.dot(d_A, d_B)) < 0.0:
            d_B = -d_B

        c_sharp = _compute_c_sharp_cpu(A, B, d_A, d_B)

        tri_L = int(info["tri_L"])
        tri_R = int(info["tri_R"])
        tri_L_idx = triangles[tri_L]
        tri_R_idx = triangles[tri_R]
        edge_idx_L = _edge_index_for_triangle(tri_L_idx, edge_key)
        edge_idx_R = _edge_index_for_triangle(tri_R_idx, edge_key)

        if edge_idx_L < 0 or edge_idx_R < 0:
            c_sharps[edge_key] = c_sharp
            continue

        x00_L, x10_L, x11_L = vertices[tri_L_idx[0]], vertices[tri_L_idx[1]], vertices[tri_L_idx[2]]
        n00_L, n10_L, n11_L = tri_vertex_normals[tri_L]
        c1_L, c2_L, c3_L = (
            _compute_curvature_cpu(x10_L - x00_L, n00_L, n10_L),
            _compute_curvature_cpu(x11_L - x10_L, n10_L, n11_L),
            _compute_curvature_cpu(x11_L - x00_L, n00_L, n11_L),
        )
        c_orig_L = [c1_L, c2_L, c3_L][edge_idx_L]

        x00_R, x10_R, x11_R = vertices[tri_R_idx[0]], vertices[tri_R_idx[1]], vertices[tri_R_idx[2]]
        n00_R, n10_R, n11_R = tri_vertex_normals[tri_R]
        c1_R, c2_R, c3_R = (
            _compute_curvature_cpu(x10_R - x00_R, n00_R, n10_R),
            _compute_curvature_cpu(x11_R - x10_R, n10_R, n11_R),
            _compute_curvature_cpu(x11_R - x00_R, n00_R, n11_R),
        )
        c_orig_R = [c1_R, c2_R, c3_R][edge_idx_R]

        baseline = 0.5 * (c_orig_L + c_orig_R)
        low, high = 0.0, 1.0
        best = baseline
        for _ in range(12):
            mid = 0.5 * (low + high)
            candidate = baseline + mid * (c_sharp - baseline)
            ok_L = _check_edge_constraints_for_triangle_cpu(
                x00_L, x10_L, x11_L, c1_L, c2_L, c3_L, edge_idx_L, candidate, k_factor
            )
            ok_R = _check_edge_constraints_for_triangle_cpu(
                x00_R, x10_R, x11_R, c1_R, c2_R, c3_R, edge_idx_R, candidate, k_factor
            )
            if ok_L and ok_R:
                best = candidate
                low = mid
            else:
                high = mid
        c_sharps[edge_key] = best

    return c_sharps


# =============================================================================
# GPU 后端核心
# =============================================================================

@dataclass
class BackendBuildInfoTorch:
    nsm_path: str
    num_vertices: int
    num_triangles: int
    num_crease_edges: int
    used_cache: bool
    eng_path: str


class EnhancedNagataBackendTorch:
    """
    GPU 版增强 Nagata 后端。

    CPU 职责:
    - 读取 NSM 网格数据
    - 加载或重建 ENG (c_sharps) 缓存
    - 构造 patch 级静态张量并传输到 GPU

    GPU 职责:
    - 批量 query_points_gpu: 并行求解最近点、计算 SDF
    - 所有数学运算在 GPU 上完成，保证与 CPU 定义一致
    """

    def __init__(
        self,
        nsm_path: str,
        *,
        use_cache: bool = True,
        force_recompute: bool = False,
        bake_cache: bool = False,
        gap_threshold: float = 1e-4,
        k_factor: float = 0.0,
        device: Optional[str] = None,
    ) -> None:
        """
        初始化 GPU 后端。

        参数:
            nsm_path: NSM 文件路径
            use_cache: 是否使用 ENG 缓存
            force_recompute: 是否强制重建 c_sharps
            bake_cache: 是否写回 ENG 缓存
            gap_threshold: 折痕边检测阈值
            k_factor: 高斯衰减参数
            device: GPU 设备字符串，None 则自动选择
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，无法创建 GPU 后端。")

        self.nsm_path = str(nsm_path)
        self.gap_threshold = float(gap_threshold)
        self.k_factor = float(k_factor)
        self.eng_path = get_eng_filepath(self.nsm_path)

        # 选择 GPU 设备
        if device is None:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device(device)

        print(f"EnhancedNagataBackendTorch 使用设备: {self.device}")

        # CPU: 读取 NSM
        self.mesh = load_nsm_lightweight(self.nsm_path)

        # CPU: 加载或重建 c_sharps
        self.c_sharps = self._load_or_build_c_sharps(
            use_cache=use_cache,
            force_recompute=force_recompute,
            bake_cache=bake_cache,
        )
        self.used_cache = hasattr(self, '_used_cache') and self._used_cache

        # CPU: 计算所有 patch 的 Nagata 系数
        self._compute_patch_coefficients()

        # CPU -> GPU: 传输静态张量
        self._build_gpu_tensors()

        self.build_info = BackendBuildInfoTorch(
            nsm_path=self.nsm_path,
            num_vertices=int(self.mesh.vertices.shape[0]),
            num_triangles=int(self.mesh.triangles.shape[0]),
            num_crease_edges=int(len(self.c_sharps)),
            used_cache=bool(self.used_cache),
            eng_path=self.eng_path,
        )

    def _load_or_build_c_sharps(
        self,
        *,
        use_cache: bool,
        force_recompute: bool,
        bake_cache: bool,
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """加载或重建 c_sharps 折痕数据 (CPU)。"""
        if use_cache and (not force_recompute) and has_cached_data(self.nsm_path):
            cached = load_enhanced_data(self.eng_path)
            if cached is not None:
                self._used_cache = True
                return cached

        crease_edges = detect_crease_edges_cpu(
            self.mesh.vertices,
            self.mesh.triangles,
            self.mesh.tri_vertex_normals,
            gap_threshold=self.gap_threshold,
        )
        c_sharps = compute_c_sharps_for_edges_cpu(
            crease_edges,
            self.mesh.vertices,
            self.mesh.triangles,
            self.mesh.tri_vertex_normals,
            k_factor=self.k_factor,
        )
        if bake_cache and c_sharps:
            save_enhanced_data(self.eng_path, c_sharps)
        self._used_cache = False
        return c_sharps

    def _compute_patch_coefficients(self) -> None:
        """为每个三角形计算 Nagata 曲率系数和折痕信息 (CPU)。"""
        n_tri = self.mesh.triangles.shape[0]

        # 每个 patch 的系数: c1, c2, c3
        self.patch_c1 = np.zeros((n_tri, 3), dtype=np.float64)
        self.patch_c2 = np.zeros((n_tri, 3), dtype=np.float64)
        self.patch_c3 = np.zeros((n_tri, 3), dtype=np.float64)

        # 每个 patch 的顶点坐标
        self.patch_x00 = np.zeros((n_tri, 3), dtype=np.float64)
        self.patch_x10 = np.zeros((n_tri, 3), dtype=np.float64)
        self.patch_x11 = np.zeros((n_tri, 3), dtype=np.float64)

        # 每个 patch 的折痕信息: is_crease (3 bools) + c_sharps (3 vectors)
        self.patch_is_crease = np.zeros((n_tri, 3), dtype=np.int32)
        self.patch_c1_sharp = np.zeros((n_tri, 3), dtype=np.float64)
        self.patch_c2_sharp = np.zeros((n_tri, 3), dtype=np.float64)
        self.patch_c3_sharp = np.zeros((n_tri, 3), dtype=np.float64)

        for tri_idx in range(n_tri):
            tri = self.mesh.triangles[tri_idx]
            x00 = self.mesh.vertices[int(tri[0])]
            x10 = self.mesh.vertices[int(tri[1])]
            x11 = self.mesh.vertices[int(tri[2])]
            n00 = self.mesh.tri_vertex_normals[tri_idx, 0]
            n10 = self.mesh.tri_vertex_normals[tri_idx, 1]
            n11 = self.mesh.tri_vertex_normals[tri_idx, 2]

            c1 = _compute_curvature_cpu(x10 - x00, n00, n10)
            c2 = _compute_curvature_cpu(x11 - x10, n10, n11)
            c3 = _compute_curvature_cpu(x11 - x00, n00, n11)

            self.patch_x00[tri_idx] = x00
            self.patch_x10[tri_idx] = x10
            self.patch_x11[tri_idx] = x11
            self.patch_c1[tri_idx] = c1
            self.patch_c2[tri_idx] = c2
            self.patch_c3[tri_idx] = c3

            # 折痕信息
            edges = [
                tuple(sorted((int(tri[0]), int(tri[1])))),
                tuple(sorted((int(tri[1]), int(tri[2])))),
                tuple(sorted((int(tri[0]), int(tri[2])))),
            ]
            for k in range(3):
                if edges[k] in self.c_sharps:
                    self.patch_is_crease[tri_idx, k] = 1
                    sharp_val = self.c_sharps[edges[k]]
                    if k == 0:
                        self.patch_c1_sharp[tri_idx] = sharp_val
                    elif k == 1:
                        self.patch_c2_sharp[tri_idx] = sharp_val
                    else:
                        self.patch_c3_sharp[tri_idx] = sharp_val

        # 每个 patch 的中心 (用于候选选择)
        self.patch_centroids = (self.patch_x00 + self.patch_x10 + self.patch_x11) / 3.0

    def _build_gpu_tensors(self) -> None:
        """将所有 patch 静态数据从 CPU numpy 传输到 GPU 张量。"""
        dtype = torch.float64

        def to_tensor(arr):
            return torch.from_numpy(arr).to(self.device, dtype=dtype)

        self.g_x00 = to_tensor(self.patch_x00)
        self.g_x10 = to_tensor(self.patch_x10)
        self.g_x11 = to_tensor(self.patch_x11)
        self.g_c1 = to_tensor(self.patch_c1)
        self.g_c2 = to_tensor(self.patch_c2)
        self.g_c3 = to_tensor(self.patch_c3)
        self.g_is_crease = torch.from_numpy(self.patch_is_crease).to(self.device, dtype=torch.int32)
        self.g_c1_sharp = to_tensor(self.patch_c1_sharp)
        self.g_c2_sharp = to_tensor(self.patch_c2_sharp)
        self.g_c3_sharp = to_tensor(self.patch_c3_sharp)
        self.g_centroids = to_tensor(self.patch_centroids)

        self.n_tri = self.mesh.triangles.shape[0]
        self.g_tri_indices = torch.arange(self.n_tri, device=self.device, dtype=torch.int32)

        # KDTree 的替代方案：使用 patch 中心进行粗筛
        # 在 GPU 上计算 patch 中心到查询点的距离，选 top-K
        print(f"GPU 张量构建完成: {self.n_tri} 个 patch, "
              f"折痕边: {len(self.c_sharps)}, "
              f"设备: {self.device}")

    # -------------------------------------------------------------------------
    # GPU 核心运算: Nagata 曲面求值和导数
    # -------------------------------------------------------------------------

    def _eval_nagata_patch_batch(
        self,
        u: torch.Tensor,  # (n_batch, n_patch)
        v: torch.Tensor,  # (n_batch, n_patch)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        批量评估 Nagata 曲面位置和导数。

        参数:
            u, v: 参数坐标，shape (n_batch, n_patch)

        返回:
            points: 曲面位置 (n_batch, n_patch, 3)
            dXdu: u 方向导数 (n_batch, n_patch, 3)
            dXdv: v 方向导数 (n_batch, n_patch, 3)
        """
        # 线性项: x00*(1-u) + x10*(u-v) + x11*v
        one_minus_u = 1.0 - u
        u_minus_v = u - v

        # (n_batch, n_patch, 3) = (1, n_patch, 3) * (n_batch, n_patch, 1)
        pts = (self.g_x00[None] * one_minus_u[:, :, None] +
               self.g_x10[None] * u_minus_v[:, :, None] +
               self.g_x11[None] * v[:, :, None])

        # 二次修正项
        quad = (self.g_c1[None] * (one_minus_u * u_minus_v)[:, :, None] +
                self.g_c2[None] * (u_minus_v * v)[:, :, None] +
                self.g_c3[None] * (one_minus_u * v)[:, :, None])
        points = pts - quad

        # 导数: 线性部分
        dLin_du = -self.g_x00 + self.g_x10  # (n_patch, 3)
        dLin_dv = -self.g_x10 + self.g_x11

        # 二次项导数
        db1_du = 1.0 - 2.0 * u + v
        db1_dv = u - 1.0
        db2_du = v
        db2_dv = u - 2.0 * v
        db3_du = -v
        db3_dv = 1.0 - u

        b1 = (1.0 - u) * (u - v)
        b2 = (u - v) * v
        b3 = (1.0 - u) * v

        dQ_du = (self.g_c1[None] * db1_du[:, :, None] +
                 self.g_c2[None] * db2_du[:, :, None] +
                 self.g_c3[None] * db3_du[:, :, None])
        dQ_dv = (self.g_c1[None] * db1_dv[:, :, None] +
                 self.g_c2[None] * db2_dv[:, :, None] +
                 self.g_c3[None] * db3_dv[:, :, None])

        dXdu = dLin_du[None] - dQ_du
        dXdv = dLin_dv[None] - dQ_dv

        # 折痕修正
        is_crease = self.g_is_crease.float()  # (n_patch, 3)
        # 距离参数: d1=v, d2=1-u, d3=u-v
        d_params = torch.stack([v, one_minus_u, u_minus_v], dim=-1)  # (n_batch, n_patch, 3)
        # dd/du, dd/dv
        dd_du = torch.tensor([0.0, -1.0, 1.0], device=self.device, dtype=torch.float64)
        dd_dv = torch.tensor([1.0, 0.0, -1.0], device=self.device, dtype=torch.float64)

        # delta_c = c_sharp - c_orig
        delta_c = torch.stack([
            self.g_c1_sharp - self.g_c1,
            self.g_c2_sharp - self.g_c2,
            self.g_c3_sharp - self.g_c3,
        ], dim=-1)  # (n_patch, 3, 3)
        delta_c = delta_c.permute(0, 2, 1)  # (n_patch, 3, 3) -> (n_patch, 3, 3) 已经是 (n_patch, edge_idx, xyz)
        # 重新排列: delta_c[edge_idx, xyz] -> (n_patch, edge_idx, xyz)
        # 实际上我们已经是 (n_patch, 3, 3) = (n_patch, edge_idx, xyz)
        # 需要 (1, n_patch, 3, 3) 乘以 mask

        bases = torch.stack([b1, b2, b3], dim=-1)  # (n_batch, n_patch, 3)
        db_du_list = torch.stack([db1_du, db2_du, db3_du], dim=-1)  # (n_batch, n_patch, 3)
        db_dv_list = torch.stack([db1_dv, db2_dv, db3_dv], dim=-1)

        # 高斯衰减: exp(-k*d^2)
        damping = torch.exp(-self.k_factor * d_params * d_params)  # (n_batch, n_patch, 3)
        ddamping_dd = -2.0 * self.k_factor * d_params * damping

        # ddamping/du = ddamping/dd * dd/du[edge_idx]
        ddamping_du = ddamping_dd * dd_du[None, None, :]  # (n_batch, n_patch, 3)
        ddamping_dv = ddamping_dd * dd_dv[None, None, :]

        # 修正项导数: dCorr/du = -delta_c * (dbasis/du * damping + basis * ddamping/du)
        # delta_c: (n_patch, 3, 3), 需要变成 (n_batch, n_patch, 3, 3)
        dc = delta_c[None]  # (1, n_patch, 3, 3)
        # db_du: (n_batch, n_patch, 3), damping: (n_batch, n_patch, 3)
        # 需要逐 edge 处理

        # 对每个 edge_idx 计算修正
        # dCorr_du = sum over edges of: -delta_c_e * (db_du_e * damping_e + basis_e * ddamping_du_e)
        # delta_c_e shape (n_patch, 3), db_du_e shape (n_batch, n_patch), damping_e (n_batch, n_patch)

        dCorr_du = torch.zeros_like(dLin_du[None])  # (n_batch, n_patch, 3)
        dCorr_dv = torch.zeros_like(dLin_dv[None])

        for edge_idx in range(3):
            mask = is_crease[:, edge_idx]  # (n_patch,)
            if not mask.any():
                continue
            mask_3d = mask[:, None]  # (n_patch, 3)
            dc_e = delta_c[:, edge_idx, :]  # (n_patch, 3)
            db_du_e = db_du_list[:, :, edge_idx]  # (n_batch, n_patch)
            db_dv_e = db_dv_list[:, :, edge_idx]
            basis_e = bases[:, :, edge_idx]
            damp_e = damping[:, :, edge_idx]
            ddamp_du_e = ddamping_du[:, :, edge_idx]
            ddamp_dv_e = ddamping_dv[:, :, edge_idx]

            term_du = (db_du_e[:, :, None] * damp_e[:, :, None] +
                       basis_e[:, :, None] * ddamp_du_e[:, :, None])
            term_dv = (db_dv_e[:, :, None] * damp_e[:, :, None] +
                       basis_e[:, :, None] * ddamp_dv_e[:, :, None])

            dCorr_du -= dc_e[None] * term_du * mask_3d[None, :, :]
            dCorr_dv -= dc_e[None] * term_dv * mask_3d[None, :, :]

        dXdu = dXdu + dCorr_du
        dXdv = dXdv + dCorr_dv

        return points, dXdu, dXdv

    def _eval_nagata_patch_single(
        self,
        tri_idx: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对单个 patch 评估位置和导数 (标量接口)。

        参数:
            tri_idx: 三角形索引 (标量)
            u, v: 参数坐标 (标量)

        返回:
            point: (3,)
            dXdu: (3,)
            dXdv: (3,)
        """
        x00 = self.g_x00[tri_idx]
        x10 = self.g_x10[tri_idx]
        x11 = self.g_x11[tri_idx]
        c1 = self.g_c1[tri_idx]
        c2 = self.g_c2[tri_idx]
        c3 = self.g_c3[tri_idx]
        is_crease = self.g_is_crease[tri_idx]

        one_minus_u = 1.0 - u
        u_minus_v = u - v

        # 位置
        linear = x00 * one_minus_u + x10 * u_minus_v + x11 * v
        quad = (c1 * (one_minus_u * u_minus_v) +
                c2 * (u_minus_v * v) +
                c3 * (one_minus_u * v))
        point = linear - quad

        # 导数: 线性部分
        dLin_du = -x00 + x10
        dLin_dv = -x10 + x11

        # 二次项导数
        b1 = (1.0 - u) * (u - v)
        b2 = (u - v) * v
        b3 = (1.0 - u) * v

        db1_du = 1.0 - 2.0 * u + v
        db1_dv = u - 1.0
        db2_du = v
        db2_dv = u - 2.0 * v
        db3_du = -v
        db3_dv = 1.0 - u

        dQ_du = c1 * db1_du + c2 * db2_du + c3 * db3_du
        dQ_dv = c1 * db1_dv + c2 * db2_dv + c3 * db3_dv

        dXdu = dLin_du - dQ_du
        dXdv = dLin_dv - dQ_dv

        # 折痕修正
        d_params = [v, 1.0 - u, u - v]
        dd_du = [0.0, -1.0, 1.0]
        dd_dv = [1.0, 0.0, -1.0]
        bases_list = [b1, b2, b3]
        db_du_list = [db1_du, db2_du, db3_du]
        db_dv_list = [db1_dv, db2_dv, db3_dv]
        coeffs_list = [c1, c2, c3]
        sharps_list = [self.g_c1_sharp[tri_idx], self.g_c2_sharp[tri_idx], self.g_c3_sharp[tri_idx]]

        for i in range(3):
            if not is_crease[i]:
                continue
            dist = d_params[i]
            delta_c = sharps_list[i] - coeffs_list[i]
            damping = torch.exp(-self.k_factor * dist * dist)
            ddamping_dd = -2.0 * self.k_factor * dist * damping
            ddamping_du = ddamping_dd * dd_du[i]
            ddamping_dv = ddamping_dd * dd_dv[i]

            dXdu -= delta_c * (db_du_list[i] * damping + bases_list[i] * ddamping_du)
            dXdv -= delta_c * (db_dv_list[i] * damping + bases_list[i] * ddamping_dv)

        return point, dXdu, dXdv

    # -------------------------------------------------------------------------
    # GPU 最近点求解: Newton 迭代 + 边界兜底
    # -------------------------------------------------------------------------

    def _newton_solve_single_patch(
        self,
        points: torch.Tensor,       # (n_points, 3)
        tri_idx: int,
        max_iter: int = 15,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对单个 patch 使用 Newton-Raphson 求解多个点的最近点。

        参数:
            points: 查询点 (n_points, 3)
            tri_idx: 三角形索引
            max_iter: 最大迭代次数

        返回:
            best_points: 最近点 (n_points, 3)
            best_dists: 距离 (n_points,)
            best_us: 最优 u (n_points,)
            best_vs: 最优 v (n_points,)
        """
        n_pts = points.shape[0]

        # 提取单个 patch 数据到局部变量
        x00 = self.g_x00[tri_idx]  # (3,)
        x10 = self.g_x10[tri_idx]
        x11 = self.g_x11[tri_idx]
        c1 = self.g_c1[tri_idx]
        c2 = self.g_c2[tri_idx]
        c3 = self.g_c3[tri_idx]
        is_crease_i = self.g_is_crease[tri_idx]  # (3,)
        c1_sharp = self.g_c1_sharp[tri_idx]
        c2_sharp = self.g_c2_sharp[tri_idx]
        c3_sharp = self.g_c3_sharp[tri_idx]

        def _eval_single(u: torch.Tensor, v: torch.Tensor):
            """对 (n_pts,) 形状的 u,v 求值和求导。"""
            one_minus_u = 1.0 - u
            u_minus_v = u - v

            # 位置 (n_pts, 3)
            linear = x00[None, :] * one_minus_u[:, None] + x10[None, :] * u_minus_v[:, None] + x11[None, :] * v[:, None]
            quad = (c1[None, :] * (one_minus_u * u_minus_v)[:, None] +
                    c2[None, :] * (u_minus_v * v)[:, None] +
                    c3[None, :] * (one_minus_u * v)[:, None])
            point = linear - quad

            # 导数: 线性部分
            dLin_du = -x00 + x10
            dLin_dv = -x10 + x11

            # 二次项导数
            db1_du = 1.0 - 2.0 * u + v
            db1_dv = u - 1.0
            db2_du = v
            db2_dv = u - 2.0 * v
            db3_du = -v
            db3_dv = 1.0 - u

            dQ_du = c1[None, :] * db1_du[:, None] + c2[None, :] * db2_du[:, None] + c3[None, :] * db3_du[:, None]
            dQ_dv = c1[None, :] * db1_dv[:, None] + c2[None, :] * db2_dv[:, None] + c3[None, :] * db3_dv[:, None]

            dXdu = dLin_du[None, :] - dQ_du
            dXdv = dLin_dv[None, :] - dQ_dv

            # 折痕修正
            if is_crease_i.any():
                d_params = [v, 1.0 - u, u - v]
                dd_du = [0.0, -1.0, 1.0]
                dd_dv = [1.0, 0.0, -1.0]
                bases_list = [(1.0 - u) * (u - v), (u - v) * v, (1.0 - u) * v]
                db_du_list = [db1_du, db2_du, db3_du]
                db_dv_list = [db1_dv, db2_dv, db3_dv]
                coeffs_list = [c1, c2, c3]
                sharps_list = [c1_sharp, c2_sharp, c3_sharp]

                for i in range(3):
                    if not is_crease_i[i]:
                        continue
                    dist = d_params[i]
                    delta_c = sharps_list[i] - coeffs_list[i]
                    damping = torch.exp(-self.k_factor * dist * dist)
                    ddamping_dd = -2.0 * self.k_factor * dist * damping
                    ddamping_du = ddamping_dd * dd_du[i]
                    ddamping_dv = ddamping_dd * dd_dv[i]

                    dXdu -= delta_c[None, :] * (db_du_list[i][:, None] * damping[:, None] +
                                                  bases_list[i][:, None] * ddamping_du[:, None])
                    dXdv -= delta_c[None, :] * (db_dv_list[i][:, None] * damping[:, None] +
                                                  bases_list[i][:, None] * ddamping_dv[:, None])

            return point, dXdu, dXdv

        # 候选初始点
        candidates = [
            (0.666, 0.333),
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.5, 0.0),
            (1.0, 0.5),
            (0.5, 0.5),
        ]

        # 平面投影
        edge1 = x10 - x00
        edge2 = x11 - x00
        normal = torch.cross(edge1, edge2, dim=-1)
        area_sq = torch.dot(normal, normal)
        if area_sq > 1e-12:
            w = points - x00[None]
            s = torch.sum(torch.cross(w, edge2[None], dim=-1) * normal[None], dim=-1) / area_sq
            t = torch.sum(torch.cross(edge1[None], w, dim=-1) * normal[None], dim=-1) / area_sq
            u_proj = torch.clamp(s + t, 0.0, 1.0)
            v_proj = torch.minimum(torch.maximum(t, torch.zeros_like(t)), u_proj)
            candidates.insert(0, (u_proj, v_proj))

        # 运行 Newton 迭代
        best_dist_sq = torch.full((n_pts,), float('inf'), device=self.device, dtype=torch.float64)
        best_result_pts = x00[None].expand(n_pts, -1).clone()
        best_us = torch.zeros(n_pts, device=self.device, dtype=torch.float64)
        best_vs = torch.zeros(n_pts, device=self.device, dtype=torch.float64)

        for cand in candidates:
            if isinstance(cand[0], torch.Tensor):
                u = cand[0].clone()
                v = cand[1].clone()
            else:
                u = torch.full((n_pts,), cand[0], device=self.device, dtype=torch.float64)
                v = torch.full((n_pts,), cand[1], device=self.device, dtype=torch.float64)

            for _ in range(max_iter):
                u = torch.clamp(u, 0.0, 1.0)
                v = torch.minimum(torch.maximum(v, torch.zeros_like(v)), u)

                p_surf, dXdu, dXdv = _eval_single(u, v)

                diff = p_surf - points

                F_u = torch.sum(diff * dXdu, dim=-1)
                F_v = torch.sum(diff * dXdv, dim=-1)

                H_uu = torch.sum(dXdu * dXdu, dim=-1)
                H_uv = torch.sum(dXdu * dXdv, dim=-1)
                H_vv = torch.sum(dXdv * dXdv, dim=-1)

                det = H_uu * H_vv - H_uv * H_uv
                mask_singular = torch.abs(det) < 1e-9

                du = torch.zeros_like(u)
                dv = torch.zeros_like(v)

                # 非奇异: Newton 步
                inv_det = 1.0 / det
                du[~mask_singular] = (H_vv[~mask_singular] * (-F_u[~mask_singular]) -
                                       H_uv[~mask_singular] * (-F_v[~mask_singular])) * inv_det[~mask_singular]
                dv[~mask_singular] = (-H_uv[~mask_singular] * (-F_u[~mask_singular]) +
                                       H_uu[~mask_singular] * (-F_v[~mask_singular])) * inv_det[~mask_singular]

                # 奇异: 梯度下降
                du[mask_singular] = -F_u[mask_singular] * 0.1
                dv[mask_singular] = -F_v[mask_singular] * 0.1

                # 步长限制
                step_len = torch.sqrt(du * du + dv * dv)
                max_step = torch.tensor(0.3, device=self.device)
                scale = torch.where(step_len > max_step, max_step / step_len, torch.ones_like(step_len))
                du = du * scale
                dv = dv * scale

                u_new = u + du
                v_new = v + dv

                # 域约束
                v_new = torch.minimum(torch.maximum(v_new, torch.zeros_like(v_new)), u_new)
                u_new = torch.clamp(u_new, 0.0, 1.0)

                change = torch.abs(u_new - u) + torch.abs(v_new - v)
                u = u_new
                v = v_new

                if torch.max(change) < 1e-6:
                    break

            # 最终 clamp
            u = torch.clamp(u, 0.0, 1.0)
            v = torch.minimum(torch.maximum(v, torch.zeros_like(v)), u)

            # 边界投影
            u_final, v_final = self._project_to_domain_gpu(points, tri_idx, u, v)

            # 评估最终点
            p_final, _, _ = _eval_single(u_final, v_final)

            dist_sq = torch.sum((p_final - points) ** 2, dim=-1)

            # 更新最优解
            improved = dist_sq < best_dist_sq
            if improved.any():
                best_dist_sq[improved] = dist_sq[improved]
                best_result_pts[improved] = p_final[improved]
                best_us[improved] = u_final[improved]
                best_vs[improved] = v_final[improved]

        return best_result_pts, torch.sqrt(best_dist_sq), best_us, best_vs

    def _project_to_domain_gpu(
        self,
        points: torch.Tensor,
        tri_idx: int,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 (u, v) 投影到参数域三角形 [0<=v<=u<=1]。

        参数:
            points: 查询点 (n_pts, 3)
            tri_idx: 三角形索引
            u, v: 自由 (u, v) (n_pts,)

        返回:
            u_proj, v_proj: 投影后的参数坐标
        """
        n_pts = points.shape[0]
        inside = (v >= 0.0) & (u <= 1.0) & (v <= u) & (u >= 0.0)

        u_out = u.clone()
        v_out = v.clone()

        if not inside.all():
            mask_out = ~inside
            n_out = int(mask_out.sum().item())
            if n_out == 0:
                return u_out, v_out

            # 对域外点做边界投影
            a = self.g_x00[tri_idx]
            b = self.g_x10[tri_idx]
            c = self.g_x11[tri_idx]
            c1 = self.g_c1[tri_idx]
            c2 = self.g_c2[tri_idx]
            c3 = self.g_c3[tri_idx]

            pts_out = points[mask_out]

            def _eval_uv(uv: torch.Tensor, vv: torch.Tensor) -> torch.Tensor:
                """评估单 patch 在给定 (u,v) 处的位置 (n_out, 3)。"""
                one_minus_u = 1.0 - uv
                u_minus_v = uv - vv
                linear = a[None, :] * one_minus_u[:, None] + b[None, :] * u_minus_v[:, None] + c[None, :] * vv[:, None]
                quad = (c1[None, :] * (one_minus_u * u_minus_v)[:, None] +
                        c2[None, :] * (u_minus_v * vv)[:, None] +
                        c3[None, :] * (one_minus_u * vv)[:, None])
                return linear - quad

            edges_uv = []
            edges_dist = []

            # 边 0: a->b, t in [0,1], u=t, v=0
            ab = b - a
            ab_len2 = torch.dot(ab, ab)
            t0 = torch.sum((pts_out - a[None]) * ab[None], dim=-1) / ab_len2
            t0 = torch.clamp(t0, 0.0, 1.0)
            p_e0 = _eval_uv(t0, torch.zeros_like(t0))
            d_e0 = torch.sqrt(torch.sum((p_e0 - pts_out) ** 2, dim=-1))
            edges_uv.append((t0, torch.zeros_like(t0)))
            edges_dist.append(d_e0)

            # 边 1: b->c, t in [0,1], u=1, v=t
            bc = c - b
            bc_len2 = torch.dot(bc, bc)
            t1 = torch.sum((pts_out - b[None]) * bc[None], dim=-1) / bc_len2
            t1 = torch.clamp(t1, 0.0, 1.0)
            p_e1 = _eval_uv(torch.ones_like(t1), t1)
            d_e1 = torch.sqrt(torch.sum((p_e1 - pts_out) ** 2, dim=-1))
            edges_uv.append((torch.ones_like(t1), t1))
            edges_dist.append(d_e1)

            # 边 2: a->c, t in [0,1], u=t, v=t
            ac = c - a
            ac_len2 = torch.dot(ac, ac)
            t2 = torch.sum((pts_out - a[None]) * ac[None], dim=-1) / ac_len2
            t2 = torch.clamp(t2, 0.0, 1.0)
            p_e2 = _eval_uv(t2, t2)
            d_e2 = torch.sqrt(torch.sum((p_e2 - pts_out) ** 2, dim=-1))
            edges_uv.append((t2, t2))
            edges_dist.append(d_e2)

            # 三个角点
            corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
            for uc, vc in corners:
                p_c = _eval_uv(
                    torch.full((n_out,), uc, device=self.device, dtype=torch.float64),
                    torch.full((n_out,), vc, device=self.device, dtype=torch.float64),
                )
                d_c = torch.sqrt(torch.sum((p_c - pts_out) ** 2, dim=-1))
                edges_uv.append((torch.full((n_out,), uc, device=self.device, dtype=torch.float64),
                                 torch.full((n_out,), vc, device=self.device, dtype=torch.float64)))
                edges_dist.append(d_c)

            # 选择最近边界
            dist_stack = torch.stack(edges_dist, dim=-1)  # (n_out, n_candidates)
            best_idx = torch.argmin(dist_stack, dim=-1)
            best_u = torch.zeros(n_out, device=self.device, dtype=torch.float64)
            best_v = torch.zeros(n_out, device=self.device, dtype=torch.float64)
            for ci, (eu, ev) in enumerate(edges_uv):
                mask_ci = best_idx == ci
                best_u[mask_ci] = eu[mask_ci]
                best_v[mask_ci] = ev[mask_ci]

            u_out[mask_out] = best_u
            v_out[mask_out] = best_v

        return u_out, v_out

    # -------------------------------------------------------------------------
    # GPU 候选 patch 筛选 (粗筛)
    # -------------------------------------------------------------------------

    def _select_candidate_patches(
        self,
        points: torch.Tensor,    # (n_points, 3)
        k_nearest: int = 16,
    ) -> torch.Tensor:
        """
        为每个查询点选择候选 patch。

        策略: 基于 patch 中心距离的 top-K 筛选。

        参数:
            points: 查询点 (n_points, 3)
            k_nearest: 候选 patch 数量

        返回:
            tri_indices: 候选三角形索引 (n_points, k_nearest)
        """
        k = min(k_nearest, self.n_tri)

        # 计算每个 patch 中心到每个点的距离 (n_points, n_patch)
        diff = points[:, None, :] - self.g_centroids[None]  # (n_points, n_patch, 3)
        dist_sq = torch.sum(diff * diff, dim=-1)  # (n_points, n_patch)

        # top-K
        _, indices = torch.topk(dist_sq, k, dim=-1, largest=False)
        return indices  # (n_points, k)

    # -------------------------------------------------------------------------
    # 公开查询接口
    # -------------------------------------------------------------------------

    def query_points_gpu(
        self,
        points: np.ndarray,
        k_nearest: int = 16,
        batch_size: int = 4096,
    ) -> Dict[str, np.ndarray]:
        """
        GPU 批量查询接口。

        核心策略: 将 (点, patch, 起点) 三元组展开为批量张量,
        在 GPU 上并行执行多起点 Newton 迭代，然后 per-point 取最小距离。

        参数:
            points: 查询点 (n_points, 3), numpy 数组
            k_nearest: 候选 patch 数量
            batch_size: GPU 批次大小 (控制显存使用)

        返回:
            与 CPU 版兼容的结果字典
        """
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points 应为 (n, 3) 形状，当前为 {points.shape}")

        n_points = points.shape[0]
        if n_points == 0:
            return self._empty_result()

        points_t = torch.from_numpy(points).to(self.device, dtype=torch.float64)

        # 多起点配置
        fixed_starts = [
            (0.666, 0.333),
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.5, 0.0),
            (1.0, 0.5),
            (0.5, 0.5),
        ]

        # 分批处理
        all_nearest_pts = []
        all_dists = []
        all_normals = []
        all_tri_indices = []
        all_uvs = []

        for batch_start in range(0, n_points, batch_size):
            batch_end = min(batch_start + batch_size, n_points)
            batch_pts = points_t[batch_start:batch_end]
            n_batch = batch_pts.shape[0]

            # 候选 patch 筛选 (n_batch, k)
            k = min(k_nearest, self.n_tri)
            cand_tris = self._select_candidate_patches(batch_pts, k_nearest)  # (n_batch, k)

            # Gather patch 数据 (n_batch, k, 3)
            idx_flat = cand_tris  # (n_batch, k)
            x00_p = self.g_x00[idx_flat]  # (n_batch, k, 3)
            x10_p = self.g_x10[idx_flat]
            x11_p = self.g_x11[idx_flat]
            c1_p = self.g_c1[idx_flat]
            c2_p = self.g_c2[idx_flat]
            c3_p = self.g_c3[idx_flat]

            # 平面投影作为主要起点
            edge1 = x10_p - x00_p  # (n_batch, k, 3)
            edge2 = x11_p - x00_p
            normal = torch.cross(edge1, edge2, dim=-1)  # (n_batch, k, 3)
            area_sq = torch.sum(normal * normal, dim=-1)  # (n_batch, k)
            w = batch_pts[:, None, :] - x00_p  # (n_batch, k, 3)
            s = torch.sum(torch.cross(w, edge2, dim=-1) * normal, dim=-1) / area_sq
            t = torch.sum(torch.cross(edge1, w, dim=-1) * normal, dim=-1) / area_sq
            u_proj = torch.clamp(s + t, 0.0, 1.0)
            v_proj = torch.minimum(torch.maximum(t, torch.zeros_like(t)), u_proj)

            # 多起点: 第一个是平面投影，其余是固定起点
            n_starts = 1 + len(fixed_starts)

            # 构建所有起点的 (u, v): (n_starts, n_batch, k)
            u_starts = torch.zeros((n_starts, n_batch, k), device=self.device, dtype=torch.float64)
            v_starts = torch.zeros((n_starts, n_batch, k), device=self.device, dtype=torch.float64)
            u_starts[0] = u_proj
            v_starts[0] = v_proj
            for si, (us, vs) in enumerate(fixed_starts, 1):
                u_starts[si] = us
                v_starts[si] = vs

            # 展开: (n_starts * n_batch * k,)
            u = u_starts.reshape(-1)
            v = v_starts.reshape(-1)
            N = u.shape[0]

            # 展平 patch 数据到 (N, 3)
            x00_all = x00_p[None].expand(n_starts, -1, -1, -1).reshape(N, 3)
            x10_all = x10_p[None].expand(n_starts, -1, -1, -1).reshape(N, 3)
            x11_all = x11_p[None].expand(n_starts, -1, -1, -1).reshape(N, 3)
            c1_all = c1_p[None].expand(n_starts, -1, -1, -1).reshape(N, 3)
            c2_all = c2_p[None].expand(n_starts, -1, -1, -1).reshape(N, 3)
            c3_all = c3_p[None].expand(n_starts, -1, -1, -1).reshape(N, 3)

            # 展平查询点: 每个起点需要对应的点
            pts_all = batch_pts[:, None, :].expand(-1, k, -1)  # (n_batch, k, 3)
            pts_all = pts_all[None].expand(n_starts, -1, -1, -1).reshape(N, 3)

            # 线性项导数
            dLin_du = -x00_all + x10_all
            dLin_dv = -x10_all + x11_all

            # Newton 迭代
            for _ in range(15):
                u = torch.clamp(u, 0.0, 1.0)
                v = torch.minimum(torch.maximum(v, torch.zeros_like(v)), u)

                one_minus_u = 1.0 - u
                u_minus_v = u - v

                linear = x00_all * one_minus_u[:, None] + x10_all * u_minus_v[:, None] + x11_all * v[:, None]
                quad = (c1_all * (one_minus_u * u_minus_v)[:, None] +
                        c2_all * (u_minus_v * v)[:, None] +
                        c3_all * (one_minus_u * v)[:, None])
                p_surf = linear - quad

                db1_du = 1.0 - 2.0 * u + v
                db1_dv = u - 1.0
                db2_du = v
                db2_dv = u - 2.0 * v
                db3_du = -v
                db3_dv = 1.0 - u

                dQ_du = c1_all * db1_du[:, None] + c2_all * db2_du[:, None] + c3_all * db3_du[:, None]
                dQ_dv = c1_all * db1_dv[:, None] + c2_all * db2_dv[:, None] + c3_all * db3_dv[:, None]

                dXdu = dLin_du - dQ_du
                dXdv = dLin_dv - dQ_dv

                diff = p_surf - pts_all

                F_u = torch.sum(diff * dXdu, dim=-1)
                F_v = torch.sum(diff * dXdv, dim=-1)

                H_uu = torch.sum(dXdu * dXdu, dim=-1)
                H_uv = torch.sum(dXdu * dXdv, dim=-1)
                H_vv = torch.sum(dXdv * dXdv, dim=-1)

                det = H_uu * H_vv - H_uv * H_uv
                mask_singular = torch.abs(det) < 1e-9

                du = torch.zeros_like(u)
                dv = torch.zeros_like(v)

                inv_det = torch.where(mask_singular, torch.tensor(1.0, device=self.device), 1.0 / det)
                du[~mask_singular] = (H_vv[~mask_singular] * (-F_u[~mask_singular]) -
                                       H_uv[~mask_singular] * (-F_v[~mask_singular])) * inv_det[~mask_singular]
                dv[~mask_singular] = (-H_uv[~mask_singular] * (-F_u[~mask_singular]) +
                                       H_uu[~mask_singular] * (-F_v[~mask_singular])) * inv_det[~mask_singular]

                du[mask_singular] = -F_u[mask_singular] * 0.1
                dv[mask_singular] = -F_v[mask_singular] * 0.1

                step_len = torch.sqrt(du * du + dv * dv)
                scale = torch.where(step_len > 0.3, 0.3 / step_len, torch.ones_like(step_len))

                u_new = u + du * scale
                v_new = v + dv * scale
                v_new = torch.minimum(torch.maximum(v_new, torch.zeros_like(v_new)), u_new)
                u_new = torch.clamp(u_new, 0.0, 1.0)

                change = torch.abs(u_new - u) + torch.abs(v_new - v)
                u = u_new
                v = v_new

                if torch.max(change) < 1e-6:
                    break

            # 最终 clamp
            u = torch.clamp(u, 0.0, 1.0)
            v = torch.minimum(torch.maximum(v, torch.zeros_like(v)), u)

            # 评估最终点
            one_minus_u = 1.0 - u
            u_minus_v = u - v
            final_pts = x00_all * one_minus_u[:, None] + x10_all * u_minus_v[:, None] + x11_all * v[:, None] - (
                c1_all * (one_minus_u * u_minus_v)[:, None] +
                c2_all * (u_minus_v * v)[:, None] +
                c3_all * (one_minus_u * v)[:, None]
            )

            dist_sq = torch.sum((final_pts - pts_all) ** 2, dim=-1)  # (N,)

            # Reshape: (n_starts, n_batch, k)
            dist_sq_r = dist_sq.view(n_starts, n_batch, k)

            # Per-point 选最小 (across all starts and all patches)
            best_overall = dist_sq_r.amin(dim=(0, 2))  # (n_batch,) 最小值
            best_idx_flat = dist_sq_r.view(n_starts * n_batch * k).argmin()

            # 更精细: 先选 best per-point across (start, patch)
            dist_sq_2d = dist_sq_r.view(n_starts * n_batch, k).min(dim=0)[0]  # 不对

            # 正确做法: reshape 到 (n_batch, n_starts * k) 然后取 argmin
            dist_sq_flat = dist_sq_r.permute(1, 0, 2).reshape(n_batch, n_starts * k)  # (n_batch, n_starts * k)
            best_flat_idx = torch.argmin(dist_sq_flat, dim=-1)  # (n_batch,)

            # 映射回 (start, batch, k) 索引
            best_k_global = best_flat_idx % k  # 在 k 维度
            best_comb = best_flat_idx // k  # 在 (n_starts) 维度

            # 收集结果
            batch_arange = torch.arange(n_batch, device=self.device)
            # 在展开数组中的 flat 索引
            flat_result_idx = best_comb * n_batch * k + batch_arange * k + best_k_global

            best_nearest = final_pts[flat_result_idx]  # (n_batch, 3)
            best_dist = torch.sqrt(dist_sq[flat_result_idx])  # (n_batch,)
            best_u_final = u[flat_result_idx]
            best_v_final = v[flat_result_idx]

            # 找到对应的 patch 索引
            # tris_expanded 需要先构建
            tris_expanded_all = cand_tris[None].expand(n_starts, -1, -1).reshape(N)  # (N,)
            best_tri = tris_expanded_all[flat_result_idx]

            # 计算法向
            x00_best = x00_all[flat_result_idx]
            x10_best = x10_all[flat_result_idx]
            x11_best = x11_all[flat_result_idx]
            c1_best = c1_all[flat_result_idx]
            c2_best = c2_all[flat_result_idx]
            c3_best = c3_all[flat_result_idx]

            u_b = best_u_final
            v_b = best_v_final
            one_minus_u = 1.0 - u_b
            u_minus_v = u_b - v_b

            db1_du = 1.0 - 2.0 * u_b + v_b
            db1_dv = u_b - 1.0
            db2_du = v_b
            db2_dv = u_b - 2.0 * v_b
            db3_du = -v_b
            db3_dv = 1.0 - u_b

            dXdu = (-x00_best + x10_best) - (c1_best * db1_du[:, None] + c2_best * db2_du[:, None] + c3_best * db3_du[:, None])
            dXdv = (-x10_best + x11_best) - (c1_best * db1_dv[:, None] + c2_best * db2_dv[:, None] + c3_best * db3_dv[:, None])

            n_vec = torch.cross(dXdu, dXdv, dim=-1)
            n_len = torch.norm(n_vec, dim=-1, keepdim=True)
            n_vec = torch.where(n_len > 1e-12, n_vec / n_len,
                                torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float64)[None])

            # SDF 符号
            diff_vec = batch_pts - best_nearest
            sign = torch.sign(torch.sum(diff_vec * n_vec, dim=-1))
            sign[sign == 0] = 1.0
            sdf = sign * best_dist

            # 法向与 SDF 梯度一致
            grad_dir = torch.where(sdf[:, None] >= 0, diff_vec, -diff_vec)
            grad_len = torch.norm(grad_dir, dim=-1, keepdim=True)
            best_normal = torch.where(grad_len > 1e-12, grad_dir / grad_len, n_vec)

            all_nearest_pts.append(best_nearest.cpu().numpy())
            all_dists.append(best_dist.cpu().numpy())
            all_normals.append(best_normal.cpu().numpy())
            all_tri_indices.append(best_tri.cpu().numpy())
            all_uvs.append(torch.stack([best_u_final, best_v_final], dim=-1).cpu().numpy())

        # 合并结果
        nearest_points = np.concatenate(all_nearest_pts, axis=0)
        unsigned_distance = np.concatenate(all_dists, axis=0)
        normals = np.concatenate(all_normals, axis=0)
        tri_indices = np.concatenate(all_tri_indices, axis=0)
        uvs = np.concatenate(all_uvs, axis=0)

        # SDF 重新计算
        diff_vec_all = points - nearest_points
        sign_all = np.sign(np.sum(diff_vec_all * normals, axis=-1))
        sign_all[sign_all == 0] = 1.0
        sdf = sign_all * unsigned_distance

        return {
            "points": points,
            "sdf": sdf,
            "unsigned_distance": unsigned_distance,
            "nearest_points": nearest_points,
            "normals": normals,
            "triangle_index": tri_indices.astype(np.int32),
            "uv": uvs,
            "feature_code": np.zeros(n_points, dtype=np.int8),
        }

    def _empty_result(self) -> Dict[str, np.ndarray]:
        """返回空结果字典。"""
        return {
            "points": np.zeros((0, 3), dtype=np.float64),
            "sdf": np.zeros((0,), dtype=np.float64),
            "unsigned_distance": np.zeros((0,), dtype=np.float64),
            "nearest_points": np.zeros((0, 3), dtype=np.float64),
            "normals": np.zeros((0, 3), dtype=np.float64),
            "triangle_index": np.zeros((0,), dtype=np.int32),
            "uv": np.zeros((0, 2), dtype=np.float64),
            "feature_code": np.zeros((0,), dtype=np.int8),
        }

    # -------------------------------------------------------------------------
    # 活跃块枚举 (复用 CPU 版逻辑，由 CPU 执行)
    # -------------------------------------------------------------------------

    def estimate_patch_aabb(self, tri_idx: int, pad: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        估计单个 patch 的 AABB (CPU 版本)。

        参数:
            tri_idx: 三角形索引
            pad: 额外扩展

        返回:
            aabb_min, aabb_max
        """
        tri = self.mesh.triangles[tri_idx]
        pts = [
            self.mesh.vertices[int(tri[0])],
            self.mesh.vertices[int(tri[1])],
            self.mesh.vertices[int(tri[2])],
        ]
        sample_uvs = [
            (0.666, 0.333), (0.5, 0.0), (1.0, 0.5),
            (0.5, 0.5), (0.25, 0.05), (0.95, 0.25), (0.25, 0.22),
        ]
        for u, v in sample_uvs:
            pts.append(self._eval_patch_numpy(tri_idx, u, v))

        pts_arr = np.stack(pts, axis=0)
        return np.min(pts_arr, axis=0) - pad, np.max(pts_arr, axis=0) + pad

    def _eval_patch_numpy(self, tri_idx: int, u: float, v: float) -> np.ndarray:
        """在 CPU 上评估单个 patch 的 numpy 版本。"""
        x00 = self.patch_x00[tri_idx]
        x10 = self.patch_x10[tri_idx]
        x11 = self.patch_x11[tri_idx]
        c1 = self.patch_c1[tri_idx]
        c2 = self.patch_c2[tri_idx]
        c3 = self.patch_c3[tri_idx]

        is_crease = (bool(self.patch_is_crease[tri_idx, 0]),
                     bool(self.patch_is_crease[tri_idx, 1]),
                     bool(self.patch_is_crease[tri_idx, 2]))
        c_sharps = (self.patch_c1_sharp[tri_idx],
                    self.patch_c2_sharp[tri_idx],
                    self.patch_c3_sharp[tri_idx])

        return _eval_nagata_patch_point(x00, x10, x11, c1, c2, c3, u, v)

    def enumerate_active_blocks(self, tau: float, block_size: float) -> List[Tuple[int, int, int]]:
        """
        枚举活跃块 (CPU 版本)。

        参数:
            tau: 窄带半宽
            block_size: 块大小

        返回:
            活跃块列表
        """
        tau = float(tau)
        block_size = float(block_size)
        active = set()
        for tri_idx in range(self.n_tri):
            aabb_min, aabb_max = self.estimate_patch_aabb(tri_idx, pad=tau)
            lo = np.floor(aabb_min / block_size).astype(int)
            hi = np.floor(aabb_max / block_size).astype(int)
            for i in range(int(lo[0]), int(hi[0]) + 1):
                for j in range(int(lo[1]), int(hi[1]) + 1):
                    for k in range(int(lo[2]), int(hi[2]) + 1):
                        active.add((i, j, k))
        return sorted(active)


__all__ = [
    "EnhancedNagataBackendTorch",
    "load_nsm_lightweight",
    "BackendBuildInfoTorch",
]
