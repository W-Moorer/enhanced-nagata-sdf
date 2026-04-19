"""
Nagata Patch计算模块
基于 Nagata (2005) 插值算法

参考: Matlab实现 (temps/ComputeCurvature.m, NagataPatch.m, PlotNagataPatch.m)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any


# 角度容差: 当法向量夹角小于0.1度时，退化为线性插值
ANGLE_TOL = np.cos(0.1 * np.pi / 180)

def _aabb_distance_sq(point: np.ndarray, aabb_min: np.ndarray, aabb_max: np.ndarray) -> float:
    d = np.maximum(0.0, np.maximum(aabb_min - point, point - aabb_max))
    return float(np.dot(d, d))


@dataclass(eq=False)
class _BvhNode:
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    left: Optional["_BvhNode"] = None
    right: Optional["_BvhNode"] = None
    indices: Optional[np.ndarray] = None


class _FeatureBvh:
    def __init__(self, aabb_mins: np.ndarray, aabb_maxs: np.ndarray, centers: np.ndarray, leaf_size: int = 16):
        self.aabb_mins = aabb_mins
        self.aabb_maxs = aabb_maxs
        self.centers = centers
        self.leaf_size = int(max(1, leaf_size))
        self.root = self._build(np.arange(centers.shape[0], dtype=np.int32))

    def _build(self, indices: np.ndarray) -> _BvhNode:
        mins = np.min(self.aabb_mins[indices], axis=0)
        maxs = np.max(self.aabb_maxs[indices], axis=0)

        if indices.shape[0] <= self.leaf_size:
            return _BvhNode(aabb_min=mins, aabb_max=maxs, indices=indices)

        span = maxs - mins
        axis = int(np.argmax(span))
        order = np.argsort(self.centers[indices, axis])
        indices = indices[order]
        mid = indices.shape[0] // 2

        left = self._build(indices[:mid])
        right = self._build(indices[mid:])
        return _BvhNode(aabb_min=mins, aabb_max=maxs, left=left, right=right)

    def query_k(self, point: np.ndarray, k: int) -> List[int]:
        import heapq

        k = int(max(1, k))
        best = []
        seq = 0
        heap = [(0.0, seq, self.root)]
        seq += 1

        worst_best = float("inf")

        while heap:
            dist_lb, _, node = heapq.heappop(heap)
            if dist_lb > worst_best:
                break

            if node.indices is not None:
                for idx in node.indices:
                    idx_i = int(idx)
                    d = _aabb_distance_sq(point, self.aabb_mins[idx_i], self.aabb_maxs[idx_i])
                    if len(best) < k:
                        best.append((d, idx_i))
                        if len(best) == k:
                            best.sort()
                            worst_best = best[-1][0]
                    else:
                        if d < worst_best:
                            best[-1] = (d, idx_i)
                            best.sort()
                            worst_best = best[-1][0]
                continue

            if node.left is not None:
                d_left = _aabb_distance_sq(point, node.left.aabb_min, node.left.aabb_max)
                if d_left <= worst_best:
                    heapq.heappush(heap, (d_left, seq, node.left))
                    seq += 1
            if node.right is not None:
                d_right = _aabb_distance_sq(point, node.right.aabb_min, node.right.aabb_max)
                if d_right <= worst_best:
                    heapq.heappush(heap, (d_right, seq, node.right))
                    seq += 1

        best.sort()
        return [idx for _, idx in best[:k]]


@dataclass
class _Feature:
    feature_type: str
    ref: Any
    aabb_min: np.ndarray
    aabb_max: np.ndarray


def compute_curvature(d: np.ndarray, n0: np.ndarray, n1: np.ndarray) -> np.ndarray:
    """
    计算Nagata插值的曲率系数向量
    
    基于Nagata 2005论文的曲率系数计算方法。
    当两个法向量接近平行时，退化为线性插值（返回零向量）。
    
    Args:
        d: 方向向量 (x1 - x0), shape (3,)
        n0: 起点法向量, shape (3,)
        n1: 终点法向量, shape (3,)
        
    Returns:
        cvec: 曲率系数向量, shape (3,)
    """
    # 3D情况的几何方法计算
    v = 0.5 * (n0 + n1)          # 法向量平均
    delta_v = 0.5 * (n0 - n1)    # 法向量差
    
    dv = np.dot(d, v)            # d在v方向的投影
    d_delta_v = np.dot(d, delta_v)  # d在delta_v方向的投影
    
    delta_c = np.dot(n0, delta_v)   # 法向量相关性
    c = 1 - 2 * delta_c             # 角度相关系数
    
    # 检查法向量是否接近平行
    if abs(c) > ANGLE_TOL:
        # 法向量接近平行，退化为线性插值
        return np.zeros(3)
    
    # 避免除零
    denom1 = 1 - delta_c
    denom2 = delta_c
    
    if abs(denom1) < 1e-12 or abs(denom2) < 1e-12:
        return np.zeros(3)
    
    cvec = (d_delta_v / denom1) * v + (dv / denom2) * delta_v
    return cvec


def nagata_patch(
    x00: np.ndarray, x10: np.ndarray, x11: np.ndarray,
    n00: np.ndarray, n10: np.ndarray, n11: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算Nagata曲面片的三个曲率系数
    
    对于三角形的三条边，分别计算曲率系数:
    - c1: 边 x00 -> x10
    - c2: 边 x10 -> x11  
    - c3: 边 x00 -> x11
    
    Args:
        x00, x10, x11: 三角形三个顶点坐标, shape (3,)
        n00, n10, n11: 三个顶点的法向量, shape (3,)
        
    Returns:
        c1, c2, c3: 三条边的曲率系数向量, shape (3,)
    """
    c1 = compute_curvature(x10 - x00, n00, n10)
    c2 = compute_curvature(x11 - x10, n10, n11)
    c3 = compute_curvature(x11 - x00, n00, n11)
    
    return c1, c2, c3


def evaluate_nagata_patch(
    x00: np.ndarray, x10: np.ndarray, x11: np.ndarray,
    c1: np.ndarray, c2: np.ndarray, c3: np.ndarray,
    u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """
    在参数域采样点计算Nagata曲面坐标
    
    Nagata曲面参数方程:
    x(u,v) = x00*(1-u) + x10*(u-v) + x11*v 
           - c1*(1-u)*(u-v) - c2*(u-v)*v - c3*(1-u)*v
           
    参数域: u,v ∈ [0,1] 且 v ≤ u (三角形区域)
    
    Args:
        x00, x10, x11: 三角形顶点坐标, shape (3,)
        c1, c2, c3: 曲率系数, shape (3,)
        u, v: 参数坐标, 可以是标量或数组
        
    Returns:
        points: 曲面采样点坐标, shape (..., 3)
    """
    dtype = np.asarray(x00).dtype
    u = np.atleast_1d(np.asarray(u, dtype=dtype))
    v = np.atleast_1d(np.asarray(v, dtype=dtype))
    
    # 计算各项
    one_minus_u = np.array(1.0, dtype=dtype) - u
    u_minus_v = u - v
    
    # 线性项
    linear = (x00[:, None] * one_minus_u + 
              x10[:, None] * u_minus_v + 
              x11[:, None] * v)
    
    # 二次修正项
    quadratic = (c1[:, None] * (one_minus_u * u_minus_v) +
                 c2[:, None] * (u_minus_v * v) +
                 c3[:, None] * (one_minus_u * v))
    
    # 最终坐标
    points = linear - quadratic
    
    return points.T  # shape (..., 3)


def sample_nagata_triangle(
    x00: np.ndarray, x10: np.ndarray, x11: np.ndarray,
    n00: np.ndarray, n10: np.ndarray, n11: np.ndarray,
    resolution: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对单个三角形采样Nagata曲面，返回点云和三角形面片
    
    Args:
        x00, x10, x11: 三角形顶点坐标
        n00, n10, n11: 顶点法向量
        resolution: 参数域采样密度 (M x M 网格)
        
    Returns:
        vertices: 采样点坐标, shape (N, 3)
        faces: 三角形索引, shape (M, 3)
    """
    # 计算曲率系数
    c1, c2, c3 = nagata_patch(x00, x10, x11, n00, n10, n11)
    
    # 在参数域采样 (只取下三角区域 v <= u)
    u_vals = np.linspace(0, 1, resolution)
    v_vals = np.linspace(0, 1, resolution)
    
    # 收集所有采样点
    vertices = []
    vertex_map = {}  # (i, j) -> vertex_index
    
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            if v <= u + 1e-10:  # 下三角区域
                point = evaluate_nagata_patch(x00, x10, x11, c1, c2, c3, u, v)
                vertex_map[(i, j)] = len(vertices)
                vertices.append(point.flatten())
    
    vertices = np.array(vertices)
    
    # 生成三角形面片
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            # 检查是否在有效区域内
            if (i, j) in vertex_map and (i+1, j) in vertex_map and (i+1, j+1) in vertex_map:
                # 下三角形
                faces.append([vertex_map[(i, j)], 
                            vertex_map[(i+1, j)], 
                            vertex_map[(i+1, j+1)]])
            
            if (i, j) in vertex_map and (i+1, j+1) in vertex_map and (i, j+1) in vertex_map:
                # 上三角形 (如果在有效区域)
                faces.append([vertex_map[(i, j)], 
                            vertex_map[(i+1, j+1)], 
                            vertex_map[(i, j+1)]])
    
    faces = np.array(faces) if faces else np.zeros((0, 3), dtype=int)
    
    return vertices, faces


def sample_all_nagata_patches(
    vertices: np.ndarray,
    triangles: np.ndarray,
    tri_vertex_normals: np.ndarray,
    resolution: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对整个网格的所有三角形采样Nagata曲面
    
    Args:
        vertices: 网格顶点, shape (num_vertices, 3)
        triangles: 三角形索引, shape (num_triangles, 3)
        tri_vertex_normals: 每个三角形的顶点法向量, shape (num_triangles, 3, 3)
        resolution: 每个三角形的采样密度
        
    Returns:
        all_vertices: 所有采样点, shape (N, 3)
        all_faces: 所有三角形面片, shape (M, 3)
        face_to_original: 每个采样三角形对应的原始三角形索引
    """
    all_vertices = []
    all_faces = []
    face_to_original = []
    
    vertex_offset = 0
    
    for tri_idx in range(triangles.shape[0]):
        # 获取三角形顶点
        i00, i10, i11 = triangles[tri_idx]
        x00 = vertices[i00]
        x10 = vertices[i10]
        x11 = vertices[i11]
        
        # 获取顶点法向量
        n00 = tri_vertex_normals[tri_idx, 0]
        n10 = tri_vertex_normals[tri_idx, 1]
        n11 = tri_vertex_normals[tri_idx, 2]
        
        # 采样该三角形
        tri_verts, tri_faces = sample_nagata_triangle(
            x00, x10, x11, n00, n10, n11, resolution
        )
        
        if len(tri_verts) > 0:
            all_vertices.append(tri_verts)
            all_faces.append(tri_faces + vertex_offset)
            face_to_original.extend([tri_idx] * len(tri_faces))
            vertex_offset += len(tri_verts)
    
    if all_vertices:
        all_vertices = np.vstack(all_vertices)
        all_faces = np.vstack(all_faces)
        face_to_original = np.array(face_to_original)
    else:
        all_vertices = np.zeros((0, 3))
        all_faces = np.zeros((0, 3), dtype=int)
        face_to_original = np.array([], dtype=int)
    
    return all_vertices, all_faces, face_to_original


# =============================================================================
# 折痕裂隙修复相关函数 (Crease-aware Nagata patch)
# =============================================================================

def compute_crease_direction(n_L: np.ndarray, n_R: np.ndarray, e: np.ndarray) -> np.ndarray:
    """
    计算折痕切向单位方向
    
    折痕方向位于两侧切平面的交线上，即两侧法向的叉积方向。
    
    Args:
        n_L: 左侧法向量, shape (3,)
        n_R: 右侧法向量, shape (3,)
        e: 边向量 (B - A), shape (3,)
        
    Returns:
        d: 折痕切向单位方向, shape (3,)
    """
    cross = np.cross(n_L, n_R)
    norm = np.linalg.norm(cross)
    
    if norm < 1e-10:
        # 退化情况：法向量近乎平行，使用边方向
        d = e / np.linalg.norm(e)
    else:
        d = cross / norm
    
    # 确保方向与边向量同向
    if np.dot(d, e) < 0:
        d = -d
    
    return d


def compute_c_sharp(A: np.ndarray, B: np.ndarray, 
                    d_A: np.ndarray, d_B: np.ndarray,
                    reg_lambda: float = 1e-6,
                    kappa: float = 2.0) -> np.ndarray:
    """
    计算共享边界系数 c^{sharp}
    
    通过最小二乘求解满足二次边界约束的端点切向长度。
    
    Args:
        A: 端点A坐标, shape (3,)
        B: 端点B坐标, shape (3,)
        d_A: 端点A的折痕切向单位方向, shape (3,)
        d_B: 端点B的折痕切向单位方向, shape (3,)
        reg_lambda: 正则化参数
        kappa: 过冲钳制系数
        
    Returns:
        c_sharp: 共享边界系数, shape (3,)
    """
    e = B - A
    e_norm = np.linalg.norm(e)
    
    # 构建 2x2 Gram 矩阵
    G = np.array([
        [np.dot(d_A, d_A), np.dot(d_A, d_B)],
        [np.dot(d_A, d_B), np.dot(d_B, d_B)]
    ])
    
    # 右端项
    r = np.array([
        2 * np.dot(e, d_A),
        2 * np.dot(e, d_B)
    ])
    
    # 正则化求解
    G_reg = G + reg_lambda * np.eye(2)
    try:
        ell = np.linalg.solve(G_reg, r)
    except np.linalg.LinAlgError:
        # 矩阵奇异，回退
        ell = np.array([e_norm, e_norm])
    
    ell_A, ell_B = ell
    T_A = ell_A * d_A
    T_B = ell_B * d_B
    
    # 计算 c^sharp
    c_sharp = (T_B - T_A) / 2
    
    # 过冲钳制
    c_norm = np.linalg.norm(c_sharp)
    max_c = kappa * e_norm
    if c_norm > max_c:
        c_sharp = c_sharp * (max_c / c_norm)
    
    return c_sharp


def smoothstep(t: np.ndarray) -> np.ndarray:
    """
    五次光滑过渡函数 (已弃用，保留用于兼容)
    
    w = 6t^5 - 15t^4 + 10t^3
    满足 w(0)=0, w(1)=1, w'(0)=w'(1)=0
    
    注意: 此函数存在"平顶"问题，建议使用 quartic_bell 替代
    
    Args:
        t: 输入参数, 可以是标量或数组, 应在 [0, 1] 范围内
        
    Returns:
        w: 过渡值
    """
    t = np.clip(t, 0, 1)
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def quartic_bell(s: np.ndarray) -> np.ndarray:
    """
    四次钟形衰减权重函数 (Quartic Bell Decay) - v2 方案
    
    w(s) = (1 - s²)², 当 0 ≤ s ≤ 1
    w(s) = 0, 当 s > 1
    
    特性:
    - w(0) = 1: 边界处完全替换
    - w'(0) = 0: 边界处导数为零，保持法向连续
    - 无平顶: 离开边界后立即衰减，避免凹槽伪影
    - C² 连续: 在 s=1 处光滑过渡到零
    
    注意: 此函数仍使用硬截断，建议使用 gaussian_decay (v3) 替代
    
    Args:
        s: 归一化距离参数 (s = d / d_threshold)
        
    Returns:
        w: 权重值，范围 [0, 1]
    """
    s = np.clip(s, 0, 1)
    return (1.0 - s * s) ** 2


def gaussian_decay(d: np.ndarray, k: float = 0.0) -> np.ndarray:
    """
    高斯指数衰减函数 (v3 方案 - 推荐)
    
    w(d) = exp(-k * d²)
    
    特性:
    - w(0) = 1: 边界处完全修复裂隙
    - w'(0) = 0: 边界处导数为零，保持法向连续
    - 无截断: 全局光滑，C^∞ 可导
    - 无拐点: 不会产生"肩部"或"坑洼"
    
    参数 k 的选择:
    - k = 0: 纯净几何模式，最光滑的结果（推荐）
    - k = 10~20: 局部约束模式，限制修正范围
    
    相比 quartic_bell 的优势:
    - 无硬截断: 没有 if dist < threshold 的判断
    - 无限光滑: C^∞ 连续，不会产生任何视觉伪影
    - 自然衰减: 像弹簧钢板受力后的自然形变
    
    Args:
        d: 到边界的距离
        k: 衰减系数 (0 = 无衰减，10~20 = 局部化)
        
    Returns:
        w: 权重值，范围 (0, 1]
    """
    return np.exp(-k * d * d)


def evaluate_nagata_patch_with_crease(
    x00: np.ndarray, x10: np.ndarray, x11: np.ndarray,
    c1_orig: np.ndarray, c2_orig: np.ndarray, c3_orig: np.ndarray,
    c1_sharp: np.ndarray, c2_sharp: np.ndarray, c3_sharp: np.ndarray,
    is_crease: tuple,
    u: np.ndarray, v: np.ndarray,
    k_factor: float = 0.0,
    enforce_constraints: bool = True
) -> np.ndarray:
    """
    带折痕修复的 Nagata 曲面求值 (v3: 自然结构传播)
    
    采用结构化补偿形式:
    x_final(u,v) = x_orig(u,v) - Sum(Psi_i)
    
    其中 Psi_i 是第 i 条边的结构化修正项:
    Psi_i = delta_c * Basis * exp(-k * d²)
    
    核心改进 (v3):
    - 无硬截断: 废弃 if dist < threshold 逻辑
    - 高斯衰减: 使用 exp(-k*d²)，C^∞ 光滑
    - 自然同调: 利用 Nagata 基函数作为传播载体
    
    参数 k_factor 的选择:
    - k = 0: 纯净几何模式，最光滑结果（推荐）
    - k = 10~20: 局部约束模式，限制修正范围
    
    Args:
        x00, x10, x11: 三角形顶点坐标
        c1_orig, c2_orig, c3_orig: 原始曲率系数
        c1_sharp, c2_sharp, c3_sharp: 裂隙修复系数
        is_crease: 三条边是否为裂隙边
        u, v: 参数坐标
        k_factor: 高斯衰减系数 (0=无衰减，10~20=局部化)
        
    Returns:
        points: 曲面采样点坐标
    """
    dtype = np.asarray(x00).dtype
    u = np.atleast_1d(np.asarray(u, dtype=dtype))
    v = np.atleast_1d(np.asarray(v, dtype=dtype))
    
    # 线性项
    one_minus_u = np.array(1.0, dtype=dtype) - u
    u_minus_v = u - v
    
    linear = (x00 * one_minus_u[:, None] + 
              x10 * u_minus_v[:, None] + 
              x11 * v[:, None])
    
    # 原始二次修正项
    quadratic_orig = (c1_orig * (one_minus_u * u_minus_v)[:, None] +
                      c2_orig * (u_minus_v * v)[:, None] +
                      c3_orig * (one_minus_u * v)[:, None])
    
    # 计算修正项 (Correction) - v3: 高斯衰减，无硬截断
    correction = np.zeros((len(u), 3), dtype=dtype)
    
    # 边1: v=0, d1=v, 基函数 (1-u)(u-v)
    if is_crease[0]:
        d1 = v
        delta_c1 = c1_sharp - c1_orig
        basis1 = one_minus_u * u_minus_v
        damping1 = gaussian_decay(d1, k_factor)
        correction -= delta_c1 * basis1[:, None] * damping1[:, None]
    
    # 边2: u=1, d2=1-u, 基函数 (u-v)v
    if is_crease[1]:
        d2 = 1 - u
        delta_c2 = c2_sharp - c2_orig
        basis2 = u_minus_v * v
        damping2 = gaussian_decay(d2, k_factor)
        correction -= delta_c2 * basis2[:, None] * damping2[:, None]
    
    # 边3: u=v, d3=u-v, 基函数 (1-u)v
    if is_crease[2]:
        d3 = u - v
        delta_c3 = c3_sharp - c3_orig
        basis3 = one_minus_u * v
        damping3 = gaussian_decay(d3, k_factor)
        correction -= delta_c3 * basis3[:, None] * damping3[:, None]
    
    points = linear - quadratic_orig + correction
    if not enforce_constraints or not any(is_crease):
        return points

    n_ref = _compute_reference_normal(x00, x10, x11, None, None, None)
    if n_ref is None:
        return points

    for idx in range(len(u)):
        uu = float(u[idx])
        vv = float(v[idx])
        orig_point = evaluate_nagata_patch(
            x00, x10, x11,
            c1_orig, c2_orig, c3_orig,
            np.array([uu]), np.array([vv])
        ).flatten()
        dXdu, dXdv = evaluate_nagata_derivatives(
            x00, x10, x11, c1_orig, c2_orig, c3_orig,
            uu, vv,
            is_crease=is_crease,
            c_sharps=(c1_sharp, c2_sharp, c3_sharp),
            k_factor=k_factor
        )
        jacobian = float(np.dot(np.cross(dXdu, dXdv), n_ref))
        if jacobian <= 0.0:
            points[idx] = orig_point
            continue
        points[idx] = _apply_edge_crossing_guard(
            orig_point,
            points[idx],
            x00, x10, x11,
            c1_sharp, c2_sharp, c3_sharp,
            is_crease,
            uu, vv,
            n_ref
        )
    return points


def _safe_normalize(vec: np.ndarray, eps: float = 1e-12) -> Optional[np.ndarray]:
    norm = np.linalg.norm(vec)
    if norm < eps:
        return None
    return vec / norm


def _compute_reference_normal(
    x00: np.ndarray,
    x10: np.ndarray,
    x11: np.ndarray,
    n00: Optional[np.ndarray],
    n10: Optional[np.ndarray],
    n11: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    tri_normal = _safe_normalize(np.cross(x10 - x00, x11 - x00))
    if tri_normal is not None:
        return tri_normal
    if n00 is None or n10 is None or n11 is None:
        return None
    return _safe_normalize(n00 + n10 + n11)


def _apply_edge_crossing_guard(
    x_orig: np.ndarray,
    x_new: np.ndarray,
    x00: np.ndarray,
    x10: np.ndarray,
    x11: np.ndarray,
    c1_sharp: np.ndarray,
    c2_sharp: np.ndarray,
    c3_sharp: np.ndarray,
    is_crease: tuple,
    u: float,
    v: float,
    n_ref: Optional[np.ndarray],
) -> np.ndarray:
    if n_ref is None:
        return x_new

    edges = [
        (x00, x10, x11, c1_sharp, u),
        (x10, x11, x00, c2_sharp, v),
        (x00, x11, x10, c3_sharp, v),
    ]

    for edge_idx, (a, b, opp, c_sharp, t) in enumerate(edges):
        if not is_crease[edge_idx]:
            continue
        e = b - a
        side_dir = _safe_normalize(np.cross(n_ref, e))
        if side_dir is None:
            continue
        if np.dot(side_dir, opp - (a + b) * 0.5) < 0:
            side_dir = -side_dir
        t_clamped = float(np.clip(t, 0.0, 1.0))
        edge_point = (1.0 - t_clamped) * a + t_clamped * b - c_sharp * t_clamped * (1.0 - t_clamped)
        s_new = float(np.dot(x_new - edge_point, side_dir))
        if s_new < 0.0:
            x_new = x_new - s_new * side_dir

    return x_new


def sample_nagata_triangle_with_crease(
    x00: np.ndarray, x10: np.ndarray, x11: np.ndarray,
    n00: np.ndarray, n10: np.ndarray, n11: np.ndarray,
    c_sharps: dict,
    edge_keys: tuple,
    resolution: int = 10,
    k_factor: float = 0.0,
    guard_edge_crossing: bool = True,
    enforce_jacobian: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    带折痕修复的单三角形 Nagata 采样 (v3: 自然结构传播)
    
    Args:
        x00, x10, x11: 三角形顶点坐标
        n00, n10, n11: 顶点法向量
        c_sharps: 裂隙边的共享系数字典
        edge_keys: 三条边的全局键 (边1, 边2, 边3)
        resolution: 采样密度
        k_factor: 高斯衰减系数 (0=无衰减，10~20=局部化)
        
    Returns:
        vertices, faces: 采样结果
    """
    # 计算原始曲率系数
    c1_orig, c2_orig, c3_orig = nagata_patch(x00, x10, x11, n00, n10, n11)
    
    # 获取裂隙修复系数
    is_crease = [False, False, False]
    c1_sharp = c1_orig.copy()
    c2_sharp = c2_orig.copy()
    c3_sharp = c3_orig.copy()
    
    if edge_keys[0] in c_sharps:
        is_crease[0] = True
        c1_sharp = c_sharps[edge_keys[0]]
    if edge_keys[1] in c_sharps:
        is_crease[1] = True
        c2_sharp = c_sharps[edge_keys[1]]
    if edge_keys[2] in c_sharps:
        is_crease[2] = True
        c3_sharp = c_sharps[edge_keys[2]]
    
    # 生成参数域采样点
    u_vals = np.linspace(0, 1, resolution)
    v_vals = np.linspace(0, 1, resolution)
    
    vertices = []
    vertex_map = {}
    
    for i, u in enumerate(u_vals):
        for j, v in enumerate(v_vals):
            if v <= u + 1e-10:
                point = evaluate_nagata_patch_with_crease(
                    x00, x10, x11,
                    c1_orig, c2_orig, c3_orig,
                    c1_sharp, c2_sharp, c3_sharp,
                    tuple(is_crease),
                    np.array([u]), np.array([v]),
                    k_factor,
                    enforce_constraints=enforce_jacobian or guard_edge_crossing
                )
                vertex_map[(i, j)] = len(vertices)
                vertices.append(point.flatten())
    
    vertices = np.array(vertices) if vertices else np.zeros((0, 3))
    
    # 生成面片
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            if (i, j) in vertex_map and (i+1, j) in vertex_map and (i+1, j+1) in vertex_map:
                faces.append([vertex_map[(i, j)], 
                            vertex_map[(i+1, j)], 
                            vertex_map[(i+1, j+1)]])
            
            if (i, j) in vertex_map and (i+1, j+1) in vertex_map and (i, j+1) in vertex_map:
                faces.append([vertex_map[(i, j)], 
                            vertex_map[(i+1, j+1)], 
                            vertex_map[(i, j+1)]])
    
    faces = np.array(faces) if faces else np.zeros((0, 3), dtype=int)
    
    return vertices, faces





# =============================================================================
# 投影与查询逻辑 (Projection & Query)
# =============================================================================

def smoothstep_deriv(t: np.ndarray) -> np.ndarray:
    """
    五次光滑过渡函数的导数 (已弃用，保留用于兼容)
    
    w'(t) = 30t²(t-1)²
    
    注意: 建议使用 quartic_bell_deriv 替代
    """
    t = np.clip(t, 0, 1)
    term = t * (t - 1.0)
    return 30.0 * term * term


def quartic_bell_deriv(s: np.ndarray) -> np.ndarray:
    """
    四次钟形衰减权重函数的导数 (v2 方案)
    
    w(s) = (1 - s²)²
    w'(s) = -4s(1 - s²)
    
    注意: 建议使用 gaussian_decay_deriv (v3) 替代
    
    Args:
        s: 归一化距离参数
        
    Returns:
        dw/ds: 权重函数的导数
    """
    s = np.clip(s, 0, 1)
    return -4.0 * s * (1.0 - s * s)


def gaussian_decay_deriv(d: np.ndarray, k: float = 0.0) -> np.ndarray:
    """
    高斯指数衰减函数的导数 (v3 方案 - 推荐)
    
    w(d) = exp(-k * d²)
    w'(d) = -2k * d * exp(-k * d²)
    
    特性:
    - w'(0) = 0: 边界处导数为零
    - 全局光滑: C^∞ 可导
    
    Args:
        d: 到边界的距离
        k: 衰减系数
        
    Returns:
        dw/dd: 权重函数对距离的导数
    """
    w = np.exp(-k * d * d)
    return -2.0 * k * d * w


def evaluate_nagata_derivatives(
    x00: np.ndarray, x10: np.ndarray, x11: np.ndarray,
    c1: np.ndarray, c2: np.ndarray, c3: np.ndarray,
    u: float, v: float,
    is_crease: tuple = (False, False, False),
    c_sharps: tuple = (None, None, None), 
    k_factor: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Nagata 曲面的一阶偏导数 (dX/du, dX/dv)
    支持折痕融合逻辑 (v3: 自然结构传播)
    
    采用修正项形式的导数:
    dX/du = dX_orig/du + d(Correction)/du
    
    其中:
    Correction = -delta_c * basis * exp(-k * d²)
    d(Correction)/du = -delta_c * [dbasis/du * damping + basis * ddamping/du]
    ddamping/du = ddamping/dd * dd/du = -2k*d*exp(-k*d²) * dd/du
    
    Args:
        x00, x10, x11: 三角形顶点坐标
        c1, c2, c3: 原始曲率系数
        u, v: 参数坐标
        is_crease: 三条边是否为裂隙边
        c_sharps: 裂隙修复系数
        k_factor: 高斯衰减系数 (0=无衰减)
        
    Returns:
        dXdu, dXdv: 曲面在 u, v 方向的偏导数
    """
    dtype = np.asarray(x00).dtype
    u = dtype.type(u)
    v = dtype.type(v)
    one = dtype.type(1.0)
    # 基础几何导数 (线性部分)
    dLin_du = -x00 + x10
    dLin_dv = -x10 + x11
    
    # 距离参数定义
    d_params = [v, one - u, u - v]
    
    # 距离对 (u,v) 的导数 [dd/du, dd/dv]
    dd_du = [dtype.type(0.0), dtype.type(-1.0), dtype.type(1.0)]
    dd_dv = [dtype.type(1.0), dtype.type(0.0), dtype.type(-1.0)]
    
    # 准备系数
    coeffs = [c1, c2, c3]
    sharp_coeffs = [c if s is None else s for c, s in zip(coeffs, c_sharps)]
    
    # 二次基函数及其导数
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
    
    # 原始二次项导数
    dQ_du = np.zeros(3)
    dQ_dv = np.zeros(3)
    
    for i in range(3):
        dQ_du += coeffs[i] * db_du_list[i]
        dQ_dv += coeffs[i] * db_dv_list[i]
    
    # 修正项导数 - v3: 高斯衰减
    dCorr_du = np.zeros(3)
    dCorr_dv = np.zeros(3)
    
    for i in range(3):
        if not is_crease[i]:
            continue
        
        dist = d_params[i]
        delta_c = sharp_coeffs[i] - coeffs[i]
        
        # 高斯衰减及其导数
        damping = gaussian_decay(np.array([dist]), k_factor)[0]
        ddamping_dd = gaussian_decay_deriv(np.array([dist]), k_factor)[0]
        
        # ddamping/du = ddamping/dd * dd/du
        ddamping_du = ddamping_dd * dd_du[i]
        ddamping_dv = ddamping_dd * dd_dv[i]
        
        # d(Correction)/du = -delta_c * [dbasis/du * damping + basis * ddamping/du]
        dCorr_du -= delta_c * (db_du_list[i] * damping + bases[i] * ddamping_du)
        dCorr_dv -= delta_c * (db_dv_list[i] * damping + bases[i] * ddamping_dv)
    
    dXdu = dLin_du - dQ_du + dCorr_du
    dXdv = dLin_dv - dQ_dv + dCorr_dv
    
    return dXdu, dXdv

def find_nearest_point_on_patch(
    point: np.ndarray,
    x00: np.ndarray, x10: np.ndarray, x11: np.ndarray,
    c1: np.ndarray, c2: np.ndarray, c3: np.ndarray,
    is_crease: tuple = (False, False, False),
    c_sharps: tuple = (None, None, None),
    k_factor: float = 0.0,
    max_iter: int = 15
) -> Tuple[np.ndarray, float, float, float]:
    """
    使用 Newton-Raphson 算法寻找单个 Patch 上的最近点
    (Robust Version: Multiple Restarts, v3: 自然结构传播)
    
    Returns: (nearest_point, distance, u, v)
    """
    # 候选初始点列表 (u, v)
    # 包含: 
    # 1. 简单平面投影 (Primary Guess)
    # 2. 面片中心 (重心的Nagata参数大致也在中心附近, 选 u=0.66, v=0.33)
    # 3. 顶点 (0,0), (1,0), (1,1)
    # 4. 边中点 (0.5, 0), (1, 0.5), (0.5, 0.5)
    
    candidates = [
        (0.666, 0.333), # 重心 approximation
        (0.0, 0.0),     # x00
        (1.0, 0.0),     # x10
        (1.0, 1.0),     # x11
        (0.5, 0.0),     # Edge 1 mid
        (1.0, 0.5),     # Edge 2 mid
        (0.5, 0.5)      # Edge 3 mid
    ]

    # 添加平面投影作为候选 (优先级最高)
    edge1 = x10 - x00
    edge2 = x11 - x00
    normal = np.cross(edge1, edge2)
    area_sq = np.dot(normal, normal)
    
    if area_sq > 1e-12:
        w = point - x00
        s = np.dot(np.cross(w, edge2), normal) / area_sq
        t = np.dot(np.cross(edge1, w), normal) / area_sq
        
        # Mapping to u,v assuming linear transform structure approximations
        # u ~ s+t, v ~ t ? 
        # Ideally, we should invert the linear basis x00(1-u) + x10(u-v) + x11v
        # = x00 + u(x10-x00) + v(x11-x10)
        # s corresponds directly to u coeff? t corresponds to v coeff? 
        # No, edge1 = x10-x00, edge2 = x11-x00 (standard)
        # Ours linear: x00 + u*edge1 + v*(x11-x10)
        # x11-x10 = (x11-x00) - (x10-x00) = edge2 - edge1
        # P = x00 + u*edge1 + v*(edge2 - edge1)
        #   = x00 + (u-v)*edge1 + v*edge2
        # So s = u-v, t = v
        # => v = t
        # => u = s + v = s + t
        
        u_proj = np.clip(s + t, 0.0, 1.0)
        v_proj = np.clip(t, 0.0, u_proj)
        candidates.insert(0, (u_proj, v_proj))

    # 挑选最接近的顶点作为参考 (Heuristic)
    d00 = np.sum((point - x00)**2)
    d10 = np.sum((point - x10)**2)
    d11 = np.sum((point - x11)**2)
    if d00 <= d10 and d00 <= d11:
        candidates.insert(1, (0.01, 0.0))
    elif d10 <= d00 and d10 <= d11:
        candidates.insert(1, (0.99, 0.0))
    else:
        candidates.insert(1, (0.99, 0.99))

    # 去重
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = (round(c[0], 2), round(c[1], 2))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    best_dist_sq = float('inf')
    best_res = (x00.copy(), float('inf'), 0.0, 0.0)

    # 对每个候选起点运行优化
    for start_u, start_v in unique_candidates:
        u, v = start_u, start_v
        
        # Optimization Loop
        for _ in range(max_iter):
            # Clamping
            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, u)
            
            u_arr, v_arr = np.array([u]), np.array([v])
            
            # Eval
            if any(is_crease):
                P_surf = evaluate_nagata_patch_with_crease(
                    x00, x10, x11, c1, c2, c3, 
                    c_sharps[0], c_sharps[1], c_sharps[2],
                    is_crease, u_arr, v_arr, k_factor
                ).flatten()
                dXdu, dXdv = evaluate_nagata_derivatives(
                    x00, x10, x11, c1, c2, c3, u, v,
                    is_crease, c_sharps, k_factor
                )
            else:
                P_surf = evaluate_nagata_patch(
                    x00, x10, x11, c1, c2, c3, u_arr, v_arr
                ).flatten()
                dXdu, dXdv = evaluate_nagata_derivatives(
                    x00, x10, x11, c1, c2, c3, u, v
                )
            
            diff = P_surf - point
            
            # Gradient
            F_u = np.dot(diff, dXdu)
            F_v = np.dot(diff, dXdv)
            
            # Hessian
            H_uu = np.dot(dXdu, dXdu)
            H_uv = np.dot(dXdu, dXdv)
            H_vv = np.dot(dXdv, dXdv)
            
            det = H_uu * H_vv - H_uv * H_uv
            if abs(det) < 1e-9:
                # Hessian invalid, verify gradient descent
                du = -F_u * 0.1 
                dv = -F_v * 0.1
            else:
                inv_det = 1.0 / det
                du = (H_vv * (-F_u) - H_uv * (-F_v)) * inv_det
                dv = (-H_uv * (-F_u) + H_uu * (-F_v)) * inv_det
                
            # Step Limiting
            step_len = np.sqrt(du*du + dv*dv)
            if step_len > 0.3:
                scale = 0.3 / step_len
                du *= scale
                dv *= scale

            u_new = u + du
            v_new = v + dv
            
            # Simple Domain Wall
            if v_new < 0: v_new = 0
            if u_new > 1: u_new = 1
            if u_new < 0: u_new = 0
            if v_new > u_new: v_new = u_new 
                
            change = abs(u_new - u) + abs(v_new - v)
            u, v = u_new, v_new
            
            if change < 1e-6:
                break
                
        # Final Eval for this candidate
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, u)
        
        if any(is_crease):
             P_final = evaluate_nagata_patch_with_crease(
                x00, x10, x11, c1, c2, c3, c_sharps[0], c_sharps[1], c_sharps[2], is_crease, np.array([u]), np.array([v]), k_factor
            ).flatten()
        else:
            P_final = evaluate_nagata_patch(x00, x10, x11, c1, c2, c3, np.array([u]), np.array([v])).flatten()
            
        dist_sq = np.sum((P_final - point)**2)
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_res = (P_final, np.sqrt(dist_sq), u, v)

    return best_res


class NagataModelQuery:
    """
    提供对整个 NSM 模型的最近点查询功能
    (Enhanced: 支持折痕自动检测与 c_sharp 修复)
    """
    def __init__(self, vertices: np.ndarray, triangles: np.ndarray, tri_vertex_normals: np.ndarray, nsm_filepath: Optional[str] = None):
        self.vertices = vertices
        self.triangles = triangles
        self.normals = tri_vertex_normals
        
        # Precompute individual patch data
        self.patch_coeffs = [] # List[Tuple(c1, c2, c3)]
        self.centroids = []
        
        print(f"初始化查询模型: {len(triangles)} 三角形...")
        for i in range(len(triangles)):
            idx = triangles[i]
            x00=vertices[idx[0]]; x10=vertices[idx[1]]; x11=vertices[idx[2]]
            n00=tri_vertex_normals[i,0]; n10=tri_vertex_normals[i,1]; n11=tri_vertex_normals[i,2]
            
            c1, c2, c3 = nagata_patch(x00, x10, x11, n00, n10, n11)
            self.patch_coeffs.append((c1, c2, c3))
            
            # Approximate centroid
            center = (x00 + x10 + x11) / 3.0
            self.centroids.append(center)
            
        self.centroids = np.array(self.centroids)
        
        # Try to build KDTree
        try:
            from scipy.spatial import KDTree
            self.kdtree = KDTree(self.centroids)
            self.use_kdtree = True
            print("KDTree 构建成功，加速开启。")
        except ImportError:
            self.use_kdtree = False
            print("警告: 未找到 scipy.spatial.KDTree，将使用暴力搜索 (速度较慢)。")
            
        self.crease_map = {}

        print("正在构建边缘拓扑...")
        edge_to_tris = {} # (min_v, max_v) -> list of tri_idx
        
        for t_idx in range(len(triangles)):
            # Edge 1: 0-1
            # Edge 2: 1-2
            # Edge 3: 0-2 (注意 Nagata 定义的边序)
            tri = triangles[t_idx]
            edges = [
                tuple(sorted((tri[0], tri[1]))),
                tuple(sorted((tri[1], tri[2]))),
                tuple(sorted((tri[0], tri[2])))
            ]
            
            for e_key in edges:
                if e_key not in edge_to_tris:
                    edge_to_tris[e_key] = []
                edge_to_tris[e_key].append(t_idx)
                
        if nsm_filepath:
            try:
                from nagata_storage import get_eng_filepath, has_cached_data, load_enhanced_data
                if has_cached_data(nsm_filepath):
                    eng_path = get_eng_filepath(nsm_filepath)
                    cached = load_enhanced_data(eng_path)
                    if cached:
                        self.crease_map = cached
                        print(f"已加载 {len(self.crease_map)} 条折痕边数据从: {eng_path}")
                    else:
                        print("ENG 缓存为空，折痕修复关闭。")
                else:
                    print("未找到 ENG 缓存，折痕修复关闭。")
            except Exception as e:
                print(f"ENG 加载失败: {e}")

        self.edge_to_tris = edge_to_tris
        vertex_to_tris = {}
        for t_idx in range(len(triangles)):
            tri = triangles[t_idx]
            for local_idx in range(3):
                v_idx = int(tri[local_idx])
                if v_idx not in vertex_to_tris:
                    vertex_to_tris[v_idx] = []
                vertex_to_tris[v_idx].append((t_idx, local_idx))
        self.vertex_to_tris = vertex_to_tris
        self._features = []
        self._feature_bvh = None
        self._build_feature_bvh()
            
    def _get_patch_crease_info(self, idx: int):
        """Helper to get crease query params for a triangle"""
        tri = self.triangles[idx]
        # Edges order corresponding to c1, c2, c3:
        # 1: v0-v1
        # 2: v1-v2
        # 3: v0-v2
        
        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[0], tri[2])))
        ]
        
        is_crease = [False, False, False]
        c_sharps = [None, None, None]
        
        for k in range(3):
            if edges[k] in self.crease_map:
                is_crease[k] = True
                c_sharps[k] = self.crease_map[edges[k]]
                
        return tuple(is_crease), tuple(c_sharps)

    def _get_candidate_triangle_indices(self, point: np.ndarray, k_nearest: int) -> List[int]:
        if self.use_kdtree:
            _, indices = self.kdtree.query(point, k=min(k_nearest, len(self.centroids)))
            if isinstance(indices, (int, np.integer)):
                return [int(indices)]
            return [int(i) for i in indices]
        return [int(i) for i in range(len(self.centroids))]

    def _build_feature_bvh(self):
        features = []

        for tri_idx in range(len(self.triangles)):
            tri_v_idx = self.triangles[tri_idx]
            x00 = self.vertices[int(tri_v_idx[0])]
            x10 = self.vertices[int(tri_v_idx[1])]
            x11 = self.vertices[int(tri_v_idx[2])]

            pts = [x00, x10, x11]
            for u, v in [(0.666, 0.333), (0.5, 0.0), (1.0, 0.5), (0.5, 0.5)]:
                p, _, _ = self._eval_patch_point_and_derivatives(tri_idx, float(u), float(v))
                pts.append(p)
            pts = np.stack(pts, axis=0)

            aabb_min = np.min(pts, axis=0)
            aabb_max = np.max(pts, axis=0)
            features.append(_Feature("FACE", int(tri_idx), aabb_min, aabb_max))

        for edge_key, tri_indices in self.edge_to_tris.items():
            vA, vB = int(edge_key[0]), int(edge_key[1])
            posA = self.vertices[vA]
            posB = self.vertices[vB]
            pts = [posA, posB]

            tri_idx0 = int(tri_indices[0])
            local_edge_idx = self._local_edge_index_for_edge_key(tri_idx0, (vA, vB))
            if local_edge_idx is not None:
                u_m, v_m = self._uv_on_edge(int(local_edge_idx), 0.5)
                p_m, _, _ = self._eval_patch_point_and_derivatives(tri_idx0, float(u_m), float(v_m))
                pts.append(p_m)

            pts = np.stack(pts, axis=0)
            aabb_min = np.min(pts, axis=0)
            aabb_max = np.max(pts, axis=0)
            f_type = "SHARPEDGE" if edge_key in self.crease_map else "EDGE"
            features.append(_Feature(f_type, (vA, vB), aabb_min, aabb_max))

        for v_idx in range(self.vertices.shape[0]):
            p = self.vertices[int(v_idx)]
            features.append(_Feature("VERTEX", int(v_idx), p.copy(), p.copy()))

        self._features = features
        aabb_mins = np.stack([f.aabb_min for f in features], axis=0)
        aabb_maxs = np.stack([f.aabb_max for f in features], axis=0)
        centers = 0.5 * (aabb_mins + aabb_maxs)
        self._feature_bvh = _FeatureBvh(aabb_mins, aabb_maxs, centers, leaf_size=32)

    def _uv_on_edge(self, local_edge_idx: int, t: float) -> Tuple[float, float]:
        t = float(np.clip(t, 0.0, 1.0))
        if local_edge_idx == 0:
            return (t, 0.0)
        if local_edge_idx == 1:
            return (1.0, t)
        return (t, t)

    def _local_edge_index_for_edge_key(self, tri_idx: int, edge_key: Tuple[int, int]) -> Optional[int]:
        tri = self.triangles[tri_idx]
        edges = [
            tuple(sorted((int(tri[0]), int(tri[1])))),
            tuple(sorted((int(tri[1]), int(tri[2])))),
            tuple(sorted((int(tri[0]), int(tri[2])))),
        ]
        for k in range(3):
            if edges[k] == edge_key:
                return k
        return None

    def _eval_patch_point_and_derivatives(self, tri_idx: int, u: float, v: float, k_factor: float = 0.0):
        tri_v_idx = self.triangles[tri_idx]
        x00 = self.vertices[int(tri_v_idx[0])]
        x10 = self.vertices[int(tri_v_idx[1])]
        x11 = self.vertices[int(tri_v_idx[2])]
        c1, c2, c3 = self.patch_coeffs[tri_idx]
        is_crease, c_sharps = self._get_patch_crease_info(tri_idx)

        if any(is_crease):
            p = evaluate_nagata_patch_with_crease(
                x00,
                x10,
                x11,
                c1,
                c2,
                c3,
                c_sharps[0],
                c_sharps[1],
                c_sharps[2],
                is_crease,
                np.array([u]),
                np.array([v]),
                k_factor,
            ).flatten()
            dXdu, dXdv = evaluate_nagata_derivatives(
                x00, x10, x11, c1, c2, c3, u, v, is_crease=is_crease, c_sharps=c_sharps, k_factor=k_factor
            )
        else:
            p = evaluate_nagata_patch(x00, x10, x11, c1, c2, c3, np.array([u]), np.array([v])).flatten()
            dXdu, dXdv = evaluate_nagata_derivatives(x00, x10, x11, c1, c2, c3, u, v)

        return p, dXdu, dXdv

    def _eval_patch_normal(self, tri_idx: int, u: float, v: float) -> np.ndarray:
        _, dXdu, dXdv = self._eval_patch_point_and_derivatives(tri_idx, u, v)
        n = np.cross(dXdu, dXdv)
        n_len = float(np.linalg.norm(n))
        if n_len > 1e-12:
            return n / n_len
        return np.array([0.0, 0.0, 1.0], dtype=float)

    def _optimize_edge(self, point: np.ndarray, tri_idx: int, local_edge_idx: int, t0: float, max_iter: int = 20):
        t = float(np.clip(t0, 0.0, 1.0))
        best_t = t
        best_dist_sq = float("inf")
        best_p = None

        for _ in range(max_iter):
            u, v = self._uv_on_edge(local_edge_idx, t)
            p, dXdu, dXdv = self._eval_patch_point_and_derivatives(tri_idx, u, v)

            if local_edge_idx == 0:
                dXdt = dXdu
            elif local_edge_idx == 1:
                dXdt = dXdv
            else:
                dXdt = dXdu + dXdv

            diff = p - point
            F = float(np.dot(diff, dXdt))
            H = float(np.dot(dXdt, dXdt))
            if H < 1e-14:
                break

            dt = -F / H
            if abs(dt) > 0.25:
                dt = 0.25 * np.sign(dt)

            t_new = float(np.clip(t + dt, 0.0, 1.0))

            dist_sq = float(np.sum((p - point) ** 2))
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_t = t
                best_p = p

            if abs(t_new - t) < 1e-8:
                t = t_new
                break
            t = t_new

        if best_p is None:
            u, v = self._uv_on_edge(local_edge_idx, best_t)
            best_p, _, _ = self._eval_patch_point_and_derivatives(tri_idx, u, v)
            best_dist_sq = float(np.sum((best_p - point) ** 2))

        return best_p, float(np.sqrt(best_dist_sq)), best_t

    def _is_interior_uv(self, u: float, v: float, eps: float) -> bool:
        return (u > eps) and (u < 1.0 - eps) and (v > eps) and (v < u - eps)

    def _project_to_domain_geometry(self, point: np.ndarray, tri_idx: int, u_free: float, v_free: float):
        if (0.0 <= v_free <= u_free <= 1.0):
            p, _, _ = self._eval_patch_point_and_derivatives(tri_idx, float(u_free), float(v_free))
            dist = float(np.linalg.norm(p - point))
            return p, dist, float(u_free), float(v_free)

        tri_v_idx = self.triangles[tri_idx]
        a = self.vertices[int(tri_v_idx[0])]
        b = self.vertices[int(tri_v_idx[1])]
        c = self.vertices[int(tri_v_idx[2])]

        candidates = []

        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        t0 = float(np.dot(point - a, ab) / ab_len2) if ab_len2 > 1e-14 else 0.0
        p_e, d_e, t_e = self._optimize_edge(point, tri_idx, 0, t0)
        u_e, v_e = self._uv_on_edge(0, t_e)
        candidates.append((p_e, d_e, u_e, v_e))

        bc = c - b
        bc_len2 = float(np.dot(bc, bc))
        t1 = float(np.dot(point - b, bc) / bc_len2) if bc_len2 > 1e-14 else 0.0
        p_e, d_e, t_e = self._optimize_edge(point, tri_idx, 1, t1)
        u_e, v_e = self._uv_on_edge(1, t_e)
        candidates.append((p_e, d_e, u_e, v_e))

        ac = c - a
        ac_len2 = float(np.dot(ac, ac))
        t2 = float(np.dot(point - a, ac) / ac_len2) if ac_len2 > 1e-14 else 0.0
        p_e, d_e, t_e = self._optimize_edge(point, tri_idx, 2, t2)
        u_e, v_e = self._uv_on_edge(2, t_e)
        candidates.append((p_e, d_e, u_e, v_e))

        corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        for u_c, v_c in corners:
            p_c, _, _ = self._eval_patch_point_and_derivatives(tri_idx, u_c, v_c)
            d_c = float(np.linalg.norm(p_c - point))
            candidates.append((p_c, d_c, u_c, v_c))

        best = min(candidates, key=lambda x: x[1])
        return best

    def _face_project_multi_start(self, point: np.ndarray, tri_idx: int, max_iter: int = 15):
        tri_v_idx = self.triangles[tri_idx]
        x00 = self.vertices[int(tri_v_idx[0])]
        x10 = self.vertices[int(tri_v_idx[1])]
        x11 = self.vertices[int(tri_v_idx[2])]

        candidates = [
            (0.666, 0.333),
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.5, 0.0),
            (1.0, 0.5),
            (0.5, 0.5),
        ]

        edge1 = x10 - x00
        edge2 = x11 - x00
        normal = np.cross(edge1, edge2)
        area_sq = float(np.dot(normal, normal))
        if area_sq > 1e-12:
            w = point - x00
            s = float(np.dot(np.cross(w, edge2), normal) / area_sq)
            t = float(np.dot(np.cross(edge1, w), normal) / area_sq)
            u_proj = float(np.clip(s + t, 0.0, 1.0))
            v_proj = float(np.clip(t, 0.0, u_proj))
            candidates.insert(0, (u_proj, v_proj))

        best_dist_sq = float("inf")
        best_u = 0.0
        best_v = 0.0
        best_p = None

        seen = set()
        unique_candidates = []
        for u0, v0 in candidates:
            key = (round(float(u0), 3), round(float(v0), 3))
            if key not in seen:
                seen.add(key)
                unique_candidates.append((float(u0), float(v0)))

        for start_u, start_v in unique_candidates:
            u = start_u
            v = start_v
            for _ in range(max_iter):
                p, dXdu, dXdv = self._eval_patch_point_and_derivatives(tri_idx, u, v)

                diff = p - point
                F_u = float(np.dot(diff, dXdu))
                F_v = float(np.dot(diff, dXdv))

                H_uu = float(np.dot(dXdu, dXdu))
                H_uv = float(np.dot(dXdu, dXdv))
                H_vv = float(np.dot(dXdv, dXdv))

                det = H_uu * H_vv - H_uv * H_uv
                if abs(det) < 1e-12:
                    du = -0.1 * F_u
                    dv = -0.1 * F_v
                else:
                    inv_det = 1.0 / det
                    du = (H_vv * (-F_u) - H_uv * (-F_v)) * inv_det
                    dv = (-H_uv * (-F_u) + H_uu * (-F_v)) * inv_det

                step_len = float(np.sqrt(du * du + dv * dv))
                if step_len > 0.35:
                    scale = 0.35 / step_len
                    du *= scale
                    dv *= scale

                u_new = u + du
                v_new = v + dv

                change = abs(u_new - u) + abs(v_new - v)
                u = u_new
                v = v_new
                if change < 1e-6:
                    break

            p_proj, dist, u_c, v_c = self._project_to_domain_geometry(point, tri_idx, u, v)
            dist_sq = dist * dist
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_u = u_c
                best_v = v_c
                best_p = p_proj

        if best_p is None:
            best_p, dist, best_u, best_v = self._project_to_domain_geometry(point, tri_idx, 0.666, 0.333)
            best_dist_sq = dist * dist

        return best_p, float(np.sqrt(best_dist_sq)), float(best_u), float(best_v)

    def query_feature_aware(self, point: np.ndarray, k_nearest: int = 16) -> Optional[Dict[str, Any]]:
        candidate_triangles = set()
        candidate_edges = set()
        candidate_vertices = set()

        if self._feature_bvh is not None and self._features:
            k_features = max(64, int(k_nearest) * 16)
            hit_indices = self._feature_bvh.query_k(point, k_features)
            for fi in hit_indices:
                f = self._features[int(fi)]
                if f.feature_type == "FACE":
                    candidate_triangles.add(int(f.ref))
                elif f.feature_type in ("EDGE", "SHARPEDGE"):
                    candidate_edges.add(tuple(sorted((int(f.ref[0]), int(f.ref[1])))))
                    tri_indices = self.edge_to_tris.get(tuple(sorted((int(f.ref[0]), int(f.ref[1])))))
                    if tri_indices:
                        for t_idx in tri_indices:
                            candidate_triangles.add(int(t_idx))
                elif f.feature_type == "VERTEX":
                    v_idx = int(f.ref)
                    candidate_vertices.add(v_idx)
                    for t_idx, _ in self.vertex_to_tris.get(v_idx, []):
                        candidate_triangles.add(int(t_idx))

        if not candidate_triangles:
            for t_idx in self._get_candidate_triangle_indices(point, k_nearest):
                candidate_triangles.add(int(t_idx))

        candidate_triangles = sorted(candidate_triangles)

        candidates = []
        min_dist = float("inf")

        eps_interior = 1e-6
        for t_idx in candidate_triangles:
            p_surf, dist, u, v = self._face_project_multi_start(point, t_idx)
            if self._is_interior_uv(u, v, eps_interior):
                surf_n = self._eval_patch_normal(t_idx, u, v)
                diff_vec = point - p_surf
                diff_len = float(np.linalg.norm(diff_vec))
                if diff_len > 1e-12:
                    grad = diff_vec / diff_len
                    if float(np.dot(grad, surf_n)) < 0.0:
                        grad = -grad
                else:
                    grad = surf_n

                if dist < min_dist:
                    min_dist = dist
                candidates.append(
                    {
                        "nearest_point": p_surf,
                        "distance": float(dist),
                        "normal": grad,
                        "triangle_index": int(t_idx),
                        "uv": (float(u), float(v)),
                        "feature_type": "FACE",
                    }
                )

        if not candidate_edges:
            for t_idx in candidate_triangles:
                tri = self.triangles[int(t_idx)]
                v0 = int(tri[0])
                v1 = int(tri[1])
                v2 = int(tri[2])
                candidate_edges.add(tuple(sorted((v0, v1))))
                candidate_edges.add(tuple(sorted((v1, v2))))
                candidate_edges.add(tuple(sorted((v0, v2))))

        for edge_key in candidate_edges:
            tri_indices = self.edge_to_tris.get(edge_key)
            if not tri_indices:
                continue

            tri_idx0 = int(tri_indices[0])
            local_edge_idx = self._local_edge_index_for_edge_key(tri_idx0, edge_key)
            if local_edge_idx is None:
                continue

            vA, vB = edge_key
            posA = self.vertices[int(vA)]
            posB = self.vertices[int(vB)]
            edge_vec = posB - posA
            edge_len2 = float(np.dot(edge_vec, edge_vec))
            t0 = float(np.dot(point - posA, edge_vec) / edge_len2) if edge_len2 > 1e-14 else 0.0
            p_edge, dist, t_best = self._optimize_edge(point, tri_idx0, int(local_edge_idx), t0)
            u_e, v_e = self._uv_on_edge(int(local_edge_idx), t_best)

            if t_best <= 1e-6 or t_best >= 1.0 - 1e-6:
                continue

            n_sum = np.zeros(3)
            for tri_idx in tri_indices:
                tri_idx = int(tri_idx)
                local_idx = self._local_edge_index_for_edge_key(tri_idx, edge_key)
                if local_idx is None:
                    continue
                u_n, v_n = self._uv_on_edge(int(local_idx), t_best)
                n_sum += self._eval_patch_normal(tri_idx, u_n, v_n)

            n_len = float(np.linalg.norm(n_sum))
            if n_len > 1e-12:
                pseudo_n = n_sum / n_len
            else:
                pseudo_n = np.array([0.0, 0.0, 1.0], dtype=float)

            diff_vec = point - p_edge
            diff_len = float(np.linalg.norm(diff_vec))
            if diff_len > 1e-12:
                grad = diff_vec / diff_len
                if float(np.dot(grad, pseudo_n)) < 0.0:
                    grad = -grad
            else:
                grad = pseudo_n

            if dist < min_dist:
                min_dist = dist
            candidates.append(
                {
                    "nearest_point": p_edge,
                    "distance": float(dist),
                    "normal": grad,
                    "triangle_index": int(tri_idx0),
                    "uv": (float(u_e), float(v_e)),
                    "feature_type": "SHARPEDGE" if edge_key in self.crease_map else "EDGE",
                }
            )

        if not candidate_vertices:
            for t_idx in candidate_triangles:
                tri = self.triangles[int(t_idx)]
                candidate_vertices.add(int(tri[0]))
                candidate_vertices.add(int(tri[1]))
                candidate_vertices.add(int(tri[2]))

        for v_idx in candidate_vertices:
            v_idx = int(v_idx)
            p_v = self.vertices[v_idx]
            dist = float(np.linalg.norm(point - p_v))

            n_sum = np.zeros(3)
            for tri_idx, local_idx in self.vertex_to_tris.get(v_idx, []):
                n_sum += self.normals[int(tri_idx), int(local_idx)]

            n_len = float(np.linalg.norm(n_sum))
            if n_len > 1e-12:
                pseudo_n = n_sum / n_len
            else:
                pseudo_n = np.array([0.0, 0.0, 1.0], dtype=float)

            diff_vec = point - p_v
            diff_len = float(np.linalg.norm(diff_vec))
            if diff_len > 1e-12:
                grad = diff_vec / diff_len
                if float(np.dot(grad, pseudo_n)) < 0.0:
                    grad = -grad
            else:
                grad = pseudo_n

            if dist < min_dist:
                min_dist = dist
            candidates.append(
                {
                    "nearest_point": p_v,
                    "distance": float(dist),
                    "normal": grad,
                    "triangle_index": int(candidate_triangles[0]) if candidate_triangles else 0,
                    "uv": (0.0, 0.0),
                    "feature_type": "VERTEX",
                    "vertex_index": v_idx,
                }
            )

        if not candidates:
            return None

        epsilon = 1e-5 * (min_dist + 1.0)
        best_candidates = [c for c in candidates if c["distance"] <= min_dist + epsilon]
        if not best_candidates:
            return None

        avg_gradient = np.zeros(3)
        mean_dist = 0.0
        for c in best_candidates:
            avg_gradient += c["normal"]
            mean_dist += c["distance"]

        norm_len = float(np.linalg.norm(avg_gradient))
        if norm_len > 1e-12:
            avg_gradient /= norm_len
        else:
            avg_gradient = best_candidates[0]["normal"]

        mean_dist /= float(len(best_candidates))

        primary_res = dict(best_candidates[0])
        primary_res["normal"] = avg_gradient
        primary_res["distance"] = float(mean_dist)
        return primary_res

    def query(self, point: np.ndarray, k_nearest: int = 16) -> dict:
        """
        查询模型上距离 point 最近的点
        (Enhanced: 支持多解处法向平滑/平均, 支持折痕修复)
        """
        candidate_indices = []
        
        if self.use_kdtree:
            dists, indices = self.kdtree.query(point, k=min(k_nearest, len(self.centroids)))
            if isinstance(indices, (int, np.integer)): indices = [indices]
            candidate_indices = indices
        else:
            candidate_indices = range(len(self.centroids))
            
        # 1. 收集所有候选结果
        candidates = []
        min_dist = float('inf')
        
        for idx in candidate_indices:
            idx = int(idx)
            tri_v_idx = self.triangles[idx]
            x00=self.vertices[tri_v_idx[0]]
            x10=self.vertices[tri_v_idx[1]]
            x11=self.vertices[tri_v_idx[2]]
            c1, c2, c3 = self.patch_coeffs[idx]
            
            # 获取折痕信息
            is_crease, c_sharps = self._get_patch_crease_info(idx)
            
            p_surf, dist, u, v = find_nearest_point_on_patch(
                point, x00, x10, x11, c1, c2, c3,
                is_crease=is_crease, c_sharps=c_sharps
            )
            
            if dist < min_dist:
                min_dist = dist
            
            # 计算法向备用
            dXdu, dXdv = evaluate_nagata_derivatives(
                x00, x10, x11, c1, c2, c3, u, v,
                is_crease=is_crease, c_sharps=c_sharps
            )
            normal = np.cross(dXdu, dXdv)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-12:
                normal /= norm_len
            else:
                normal = np.array([0.,0.,1.])
                
            candidates.append({
                'nearest_point': p_surf,
                'distance': dist,
                'normal': normal,
                'triangle_index': idx,
                'uv': (u,v)
            })
            
        # 2. 筛选最优集合 (Tolerance for float errors)
        epsilon = 1e-5 * (min_dist + 1.0) # Relative + Absolute
        best_candidates = [c for c in candidates if c['distance'] <= min_dist + epsilon]
        
        if not best_candidates:
            return None
            
        # 3. 融合结果 (Selector implementation)
        # 核心逻辑: 平均所有"最近"候选者的梯度方向
        # - 对于外部 Corner (Same point): geometry direction 是一致的 (P-q)，平均无影响 (除了数值噪点)
        # - 对于内部 Medial Axis (Different points): geometry directions 不同，平均产生对称的角平分线 (Theory Section 5.2)
        
        avg_gradient = np.zeros(3)
        mean_dist = 0.0
        contributing_count = 0
        
        # 记录用于显示的元数据 (取第一个)
        primary_res = best_candidates[0]
        
        for c in best_candidates:
            p_surf = c['nearest_point']
            surf_normal = c['normal']
            dist = c['distance']
            
            # 计算该候选者的梯度方向 g_i
            diff_vec = point - p_surf
            dist_geo = np.linalg.norm(diff_vec)
            
            if dist_geo > 1e-6:
                # 几何方向 (P - q) / d
                g_i = diff_vec / dist_geo
                
                # 符号判断: SDF Gradient 永远指向"SDF值增加"的方向 (Outwards)
                # Check alignment with surface normal
                if np.dot(g_i, surf_normal) < 0:
                     # P is Inside. P-q points "Inwards".
                     # Gradient should point "Outwards" (-g_i).
                     g_i = -g_i
                else:
                     # P is Outside. P-q points "Outwards".
                     # Gradient = g_i
                     pass
            else:
                # On surface: use surface normal
                g_i = surf_normal
            
            avg_gradient += g_i
            mean_dist += dist
            contributing_count += 1
            
        if contributing_count > 0:
            # 归一化平均梯度
            norm_len = np.linalg.norm(avg_gradient)
            if norm_len > 1e-12:
                avg_gradient /= norm_len
            else:
                # Fallback (Theory Section 5.3): Maximum Inner Product or Parent
                # 这里简单处理: 取第一个的梯度
                avg_gradient = primary_res['normal'] # This is actually surf normal in raw data... 
                # Recompute exact gradient for primary
                diff = point - primary_res['nearest_point']
                if np.linalg.norm(diff) > 1e-6:
                    g0 = diff / np.linalg.norm(diff)
                    if np.dot(g0, primary_res['normal']) < 0: g0 = -g0
                    avg_gradient = g0
                
            mean_dist /= contributing_count
            
            # 更新结果
            primary_res['normal'] = avg_gradient
            primary_res['distance'] = mean_dist
            
            # Signed Distance (Re-evaluate sign based on final averaged gradient?)
            # Usually sign is determined by the "Winner".
            # If internal, sign is negative.
            # But the 'query' function returns unsigned distance + normal.
            # The calling script calculates signed distance.
            # We should ensure consistence. Use dot(P-q, N_final)?
            # No, `query` returns 'distance' (unsigned).
            # Caller does `sign = 1 if dot(diff, normal) >= 0 else -1`.
            # If we return N_final = (1,1).normalized(). P=(0.4,0.4). q=(0.5,0.4)?
            # diff = (-0.1, 0). dot((-0.1, 0), (0.7, 0.7)) = -0.07 < 0.
            # Sign = -1. Correct.
            
        return primary_res


if __name__ == '__main__':
    # 简单测试
    print("Nagata Patch 计算模块测试")
    
    # 测试一个简单三角形
    x00 = np.array([0.0, 0.0, 0.0])
    x10 = np.array([1.0, 0.0, 0.0])
    x11 = np.array([0.5, 1.0, 0.0])
    
    # 假设法向量都指向z轴正方向
    n00 = np.array([0.0, 0.0, 1.0])
    n10 = np.array([0.0, 0.0, 1.0])
    n11 = np.array([0.0, 0.0, 1.0])
    
    c1, c2, c3 = nagata_patch(x00, x10, x11, n00, n10, n11)
    print(f"曲率系数: c1={c1}, c2={c2}, c3={c3}")
    
    # 采样测试
    verts, faces = sample_nagata_triangle(x00, x10, x11, n00, n10, n11, resolution=5)
    print(f"采样结果: {len(verts)} 个顶点, {len(faces)} 个三角形")

    # 投影测试
    print("\n投影测试:")
    model = NagataModelQuery(
        vertices=np.array([x00, x10, x11]),
        triangles=np.array([[0, 1, 2]]),
        tri_vertex_normals=np.array([[n00, n10, n11]])
    )
    
    test_pt = np.array([0.5, 0.3, 1.0]) # 位于三角形上方
    res = model.query(test_pt)
    print(f"查询点: {test_pt}")
    print(f"最近点: {res['nearest_point']}")
    print(f"距离: {res['distance']:.6f}")
    print(f"法向: {res['normal']}")
    print(f"Face ID: {res['triangle_index']}")
