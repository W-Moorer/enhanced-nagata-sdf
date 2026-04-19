"""
Nagata Patch可视化工具
使用PyVista进行交互式可视化

功能:
- 读取NSM文件
- 计算Nagata曲面
- 并排对比原始网格与Nagata曲面
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import sys

# 导入本地模块
from nsm_reader import load_nsm, create_pyvista_mesh
from nagata_patch import (sample_all_nagata_patches, nagata_patch, 
                          compute_crease_direction, compute_c_sharp,
                          sample_nagata_triangle_with_crease,
                          evaluate_nagata_patch_with_crease,
                          evaluate_nagata_derivatives)
from nagata_storage import (save_enhanced_data, load_enhanced_data, 
                            get_eng_filepath, has_cached_data)


def create_nagata_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    tri_vertex_normals: np.ndarray,
    tri_face_ids: np.ndarray,
    resolution: int = 10
) -> pv.PolyData:
    """
    从NSM数据创建Nagata曲面的PyVista网格
    
    Args:
        vertices: 顶点坐标
        triangles: 三角形索引
        tri_vertex_normals: 顶点法向量
        tri_face_ids: 面片ID
        resolution: 采样密度
        
    Returns:
        pv.PolyData: Nagata曲面网格
    """
    # 采样所有Nagata patches
    nagata_verts, nagata_faces, face_to_original = sample_all_nagata_patches(
        vertices, triangles, tri_vertex_normals, resolution
    )
    
    if len(nagata_verts) == 0:
        print("警告: 无法生成Nagata曲面")
        return pv.PolyData()
    
    # 创建PyVista格式的faces数组
    pv_faces = np.hstack([
        np.full((nagata_faces.shape[0], 1), 3, dtype=np.int32),
        nagata_faces.astype(np.int32)
    ]).flatten()
    
    # 创建PolyData
    mesh = pv.PolyData(nagata_verts, pv_faces)
    
    # 添加面片ID (从原始三角形继承)
    if len(face_to_original) > 0:
        mesh.cell_data['face_id'] = tri_face_ids[face_to_original]
        mesh.cell_data['original_tri'] = face_to_original
    
    return mesh


def _polydata_to_triangles(mesh: pv.PolyData) -> tuple:
    if mesh is None or mesh.n_cells == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int)
    faces = mesh.faces.reshape(-1, 4)
    if np.any(faces[:, 0] != 3):
        raise ValueError("Only triangle faces are supported")
    return mesh.points.copy(), faces[:, 1:4].astype(np.int32)


def _segment_intersects_triangle(p0: np.ndarray, p1: np.ndarray, t0: np.ndarray, t1: np.ndarray, t2: np.ndarray, eps: float) -> bool:
    d = p1 - p0
    e1 = t1 - t0
    e2 = t2 - t0
    h = np.cross(d, e2)
    a = np.dot(e1, h)
    if abs(a) < eps:
        return False
    f = 1.0 / a
    s = p0 - t0
    u = f * np.dot(s, h)
    if u < -eps or u > 1.0 + eps:
        return False
    q = np.cross(s, e1)
    v = f * np.dot(d, q)
    if v < -eps or u + v > 1.0 + eps:
        return False
    t = f * np.dot(e2, q)
    return -eps <= t <= 1.0 + eps


def _segment_intersects_triangle_strict(p0: np.ndarray, p1: np.ndarray, t0: np.ndarray, t1: np.ndarray, t2: np.ndarray, eps: float) -> bool:
    d = p1 - p0
    e1 = t1 - t0
    e2 = t2 - t0
    h = np.cross(d, e2)
    a = np.dot(e1, h)
    if abs(a) < eps:
        return False
    f = 1.0 / a
    s = p0 - t0
    u = f * np.dot(s, h)
    if u <= eps or u >= 1.0 - eps:
        return False
    q = np.cross(s, e1)
    v = f * np.dot(d, q)
    if v <= eps or u + v >= 1.0 - eps:
        return False
    t = f * np.dot(e2, q)
    return eps < t < 1.0 - eps


def _orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, eps: float) -> bool:
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps and
            abs(_orient2d(a, b, p)) <= eps)


def _segments_intersect_2d(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray, eps: float) -> bool:
    o1 = _orient2d(a1, a2, b1)
    o2 = _orient2d(a1, a2, b2)
    o3 = _orient2d(b1, b2, a1)
    o4 = _orient2d(b1, b2, a2)
    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True
    if abs(o1) <= eps and _on_segment(a1, a2, b1, eps):
        return True
    if abs(o2) <= eps and _on_segment(a1, a2, b2, eps):
        return True
    if abs(o3) <= eps and _on_segment(b1, b2, a1, eps):
        return True
    if abs(o4) <= eps and _on_segment(b1, b2, a2, eps):
        return True
    return False


def _segments_intersect_2d_strict(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray, eps: float) -> bool:
    o1 = _orient2d(a1, a2, b1)
    o2 = _orient2d(a1, a2, b2)
    o3 = _orient2d(b1, b2, a1)
    o4 = _orient2d(b1, b2, a2)
    return (o1 * o2 < -eps) and (o3 * o4 < -eps)


def _point_in_tri_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float) -> bool:
    o1 = _orient2d(a, b, p)
    o2 = _orient2d(b, c, p)
    o3 = _orient2d(c, a, p)
    has_neg = (o1 < -eps) or (o2 < -eps) or (o3 < -eps)
    has_pos = (o1 > eps) or (o2 > eps) or (o3 > eps)
    return not (has_neg and has_pos)


def _point_in_tri_2d_strict(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float) -> bool:
    o1 = _orient2d(a, b, p)
    o2 = _orient2d(b, c, p)
    o3 = _orient2d(c, a, p)
    if (abs(o1) <= eps) or (abs(o2) <= eps) or (abs(o3) <= eps):
        return False
    return (o1 > 0 and o2 > 0 and o3 > 0) or (o1 < 0 and o2 < 0 and o3 < 0)


def _tri_tri_intersect(a0: np.ndarray, a1: np.ndarray, a2: np.ndarray,
                       b0: np.ndarray, b1: np.ndarray, b2: np.ndarray,
                       eps: float) -> bool:
    n1 = np.cross(a1 - a0, a2 - a0)
    n2 = np.cross(b1 - b0, b2 - b0)
    if np.linalg.norm(n1) < eps or np.linalg.norm(n2) < eps:
        return False
    if (abs(np.dot(n1, b0 - a0)) < eps and
        abs(np.dot(n1, b1 - a0)) < eps and
        abs(np.dot(n1, b2 - a0)) < eps):
        axis = int(np.argmax(np.abs(n1)))
        def proj(p):
            if axis == 0:
                return np.array([p[1], p[2]])
            if axis == 1:
                return np.array([p[0], p[2]])
            return np.array([p[0], p[1]])
        a0p, a1p, a2p = proj(a0), proj(a1), proj(a2)
        b0p, b1p, b2p = proj(b0), proj(b1), proj(b2)
        a_edges = [(a0p, a1p), (a1p, a2p), (a2p, a0p)]
        b_edges = [(b0p, b1p), (b1p, b2p), (b2p, b0p)]
        for e1 in a_edges:
            for e2 in b_edges:
                if _segments_intersect_2d_strict(e1[0], e1[1], e2[0], e2[1], eps):
                    return True
        if _point_in_tri_2d_strict(a0p, b0p, b1p, b2p, eps):
            return True
        if _point_in_tri_2d_strict(b0p, a0p, a1p, a2p, eps):
            return True
        return False
    a_edges3d = [(a0, a1), (a1, a2), (a2, a0)]
    b_edges3d = [(b0, b1), (b1, b2), (b2, b0)]
    for e0, e1 in a_edges3d:
        if _segment_intersects_triangle_strict(e0, e1, b0, b1, b2, eps):
            return True
    for e0, e1 in b_edges3d:
        if _segment_intersects_triangle_strict(e0, e1, a0, a1, a2, eps):
            return True
    return False


def count_self_intersections(vertices: np.ndarray, faces: np.ndarray, eps: float = 1e-9) -> dict:
    n = faces.shape[0]
    if n == 0:
        return {"pairs": 0, "pairs_sample": []}
    face_vertices = faces
    vertex_sets = [set(face_vertices[i]) for i in range(n)]
    pairs = []
    for i in range(n):
        a0, a1, a2 = vertices[face_vertices[i]]
        for j in range(i + 1, n):
            if vertex_sets[i] & vertex_sets[j]:
                continue
            b0, b1, b2 = vertices[face_vertices[j]]
            if _tri_tri_intersect(a0, a1, a2, b0, b1, b2, eps):
                pairs.append((i, j))
    sample = pairs[:10]
    return {"pairs": len(pairs), "pairs_sample": sample}


def detect_crease_edges(vertices, triangles, tri_vertex_normals, gap_threshold=1e-4):
    """
    检测裂隙边（两侧 Nagata 边界曲线不一致的边）
    
    Returns:
        crease_edges: dict mapping edge_key -> {
            'A': 端点A坐标, 'B': 端点B坐标,
            'n_A_L': A点左侧法向, 'n_A_R': A点右侧法向,
            'n_B_L': B点左侧法向, 'n_B_R': B点右侧法向,
            'tri_L': 左侧三角形索引, 'tri_R': 右侧三角形索引
        }
    """
    from collections import defaultdict
    
    # 构建边到三角形的映射
    edge_to_tris = defaultdict(list)
    
    for tri_idx, tri in enumerate(triangles):
        # 三条边: (0,1), (1,2), (0,2)
        edges = [
            (tri[0], tri[1], 0, 1),  # 边1: v00->v10
            (tri[1], tri[2], 1, 2),  # 边2: v10->v11
            (tri[0], tri[2], 0, 2),  # 边3: v00->v11
        ]
        for v0, v1, local0, local1 in edges:
            edge_key = tuple(sorted([v0, v1]))
            edge_to_tris[edge_key].append((tri_idx, v0, v1, local0, local1))
    
    crease_edges = {}
    
    for edge_key, tris_info in edge_to_tris.items():
        if len(tris_info) != 2:
            continue  # 边界边或非流形
        
        tri_L_info, tri_R_info = tris_info
        tri_L, v0_L, v1_L, local0_L, local1_L = tri_L_info
        tri_R, v0_R, v1_R, local0_R, local1_R = tri_R_info
        
        # 统一边方向 (A, B)
        A_idx, B_idx = edge_key
        A = vertices[A_idx]
        B = vertices[B_idx]
        
        # 获取两侧在端点处的法向
        def get_normal_at_vertex(tri_idx, global_v_idx, local_indices):
            tri = triangles[tri_idx]
            for local_idx in range(3):
                if tri[local_idx] == global_v_idx:
                    return tri_vertex_normals[tri_idx, local_idx]
            return None
        
        n_A_L = get_normal_at_vertex(tri_L, A_idx, local0_L)
        n_B_L = get_normal_at_vertex(tri_L, B_idx, local1_L)
        n_A_R = get_normal_at_vertex(tri_R, A_idx, local0_R)
        n_B_R = get_normal_at_vertex(tri_R, B_idx, local1_R)
        
        if n_A_L is None or n_B_L is None or n_A_R is None or n_B_R is None:
            continue
        
        # 计算两侧的 Nagata 边界曲线系数
        from nagata_patch import compute_curvature
        e = B - A
        c_L = compute_curvature(e, n_A_L, n_B_L)
        c_R = compute_curvature(e, n_A_R, n_B_R)
        
        # 采样比较
        max_gap = 0.0
        for t in np.linspace(0, 1, 11):
            p_L = (1-t)*A + t*B - c_L * t * (1-t)
            p_R = (1-t)*A + t*B - c_R * t * (1-t)
            gap = np.linalg.norm(p_L - p_R)
            max_gap = max(max_gap, gap)
        
        if max_gap > gap_threshold:
            crease_edges[edge_key] = {
                'A': A, 'B': B,
                'n_A_L': n_A_L, 'n_A_R': n_A_R,
                'n_B_L': n_B_L, 'n_B_R': n_B_R,
                'tri_L': tri_L, 'tri_R': tri_R,
                'max_gap': max_gap
            }
    
    return crease_edges


def _edge_index_for_triangle(tri: np.ndarray, edge_key: tuple) -> int:
    if tuple(sorted([tri[0], tri[1]])) == edge_key:
        return 0
    if tuple(sorted([tri[1], tri[2]])) == edge_key:
        return 1
    if tuple(sorted([tri[0], tri[2]])) == edge_key:
        return 2
    return -1


def _sample_uv_for_edge(edge_idx: int, steps: int = 5, eps: float = 0.05) -> list:
    ts = np.linspace(eps, 1.0 - eps, steps)
    uvs = []
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


def _check_edge_constraints_for_triangle(
    x00: np.ndarray,
    x10: np.ndarray,
    x11: np.ndarray,
    c1_orig: np.ndarray,
    c2_orig: np.ndarray,
    c3_orig: np.ndarray,
    edge_idx: int,
    c_sharp_edge: np.ndarray,
    k_factor: float,
    eps: float = 1e-10
) -> bool:
    n_ref = np.cross(x10 - x00, x11 - x00)
    n_len = np.linalg.norm(n_ref)
    if n_len < 1e-12:
        return True
    n_ref = n_ref / n_len

    is_crease = [False, False, False]
    c_sharps = [c1_orig, c2_orig, c3_orig]
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
    side_len = np.linalg.norm(side_dir)
    if side_len < 1e-12:
        return True
    side_dir = side_dir / side_len
    if np.dot(side_dir, opp - (a + b) * 0.5) < 0:
        side_dir = -side_dir

    for uu, vv in _sample_uv_for_edge(edge_idx):
        p = evaluate_nagata_patch_with_crease(
            x00, x10, x11,
            c1_orig, c2_orig, c3_orig,
            c_sharps[0], c_sharps[1], c_sharps[2],
            tuple(is_crease),
            np.array([uu]), np.array([vv]),
            k_factor,
            enforce_constraints=False
        ).flatten()
        dXdu, dXdv = evaluate_nagata_derivatives(
            x00, x10, x11,
            c1_orig, c2_orig, c3_orig,
            uu, vv,
            is_crease=tuple(is_crease),
            c_sharps=tuple(c_sharps),
            k_factor=k_factor
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


def compute_c_sharps_for_edges(crease_edges, vertices, triangles, tri_vertex_normals, k_factor: float = 0.0):
    """
    为所有裂隙边计算共享边界系数
    
    Returns:
        c_sharps: dict mapping edge_key -> c_sharp vector
    """
    c_sharps = {}
    
    for edge_key, info in crease_edges.items():
        A, B = info['A'], info['B']
        e = B - A
        
        # 计算两端点的折纹方向
        d_A = compute_crease_direction(info['n_A_L'], info['n_A_R'], e)
        d_B = compute_crease_direction(info['n_B_L'], info['n_B_R'], e)
        
        # 确保方向一致
        if np.dot(d_A, d_B) < 0:
            d_B = -d_B
        
        # 计算共享系数
        c_sharp = compute_c_sharp(A, B, d_A, d_B)

        tri_L = info['tri_L']
        tri_R = info['tri_R']
        tri_L_idx = triangles[tri_L]
        tri_R_idx = triangles[tri_R]
        edge_idx_L = _edge_index_for_triangle(tri_L_idx, edge_key)
        edge_idx_R = _edge_index_for_triangle(tri_R_idx, edge_key)

        if edge_idx_L < 0 or edge_idx_R < 0:
            c_sharps[edge_key] = c_sharp
            continue

        x00_L, x10_L, x11_L = vertices[tri_L_idx[0]], vertices[tri_L_idx[1]], vertices[tri_L_idx[2]]
        n00_L, n10_L, n11_L = tri_vertex_normals[tri_L]
        c1_L, c2_L, c3_L = nagata_patch(x00_L, x10_L, x11_L, n00_L, n10_L, n11_L)
        c_orig_L = [c1_L, c2_L, c3_L][edge_idx_L]

        x00_R, x10_R, x11_R = vertices[tri_R_idx[0]], vertices[tri_R_idx[1]], vertices[tri_R_idx[2]]
        n00_R, n10_R, n11_R = tri_vertex_normals[tri_R]
        c1_R, c2_R, c3_R = nagata_patch(x00_R, x10_R, x11_R, n00_R, n10_R, n11_R)
        c_orig_R = [c1_R, c2_R, c3_R][edge_idx_R]

        baseline = 0.5 * (c_orig_L + c_orig_R)
        low, high = 0.0, 1.0
        best = baseline
        for _ in range(12):
            mid = (low + high) * 0.5
            candidate = baseline + mid * (c_sharp - baseline)
            ok_L = _check_edge_constraints_for_triangle(
                x00_L, x10_L, x11_L,
                c1_L, c2_L, c3_L,
                edge_idx_L,
                candidate,
                k_factor
            )
            ok_R = _check_edge_constraints_for_triangle(
                x00_R, x10_R, x11_R,
                c1_R, c2_R, c3_R,
                edge_idx_R,
                candidate,
                k_factor
            )
            if ok_L and ok_R:
                best = candidate
                low = mid
            else:
                high = mid
        c_sharps[edge_key] = best
    
    return c_sharps


def create_nagata_mesh_enhanced(
    vertices: np.ndarray,
    triangles: np.ndarray,
    tri_vertex_normals: np.ndarray,
    tri_face_ids: np.ndarray,
    resolution: int = 10,
    d0: float = 0.1,
    cached_c_sharps: dict = None
) -> tuple:
    """
    带折纹修复的 Nagata 曲面生成
    
    Returns:
        tuple: (mesh, c_sharps) - 网格和计算/加载的c_sharps字典
    """
    c_sharps = None
    
    # 如果有缓存数据，直接使用
    if cached_c_sharps is not None:
        print(f"使用缓存的增强数据 ({len(cached_c_sharps)} 条裂隙边)")
        c_sharps = cached_c_sharps
    else:
        # 否则检测并计算
        print("检测裂隙边...")
        crease_edges = detect_crease_edges(vertices, triangles, tri_vertex_normals)
        print(f"发现 {len(crease_edges)} 条裂隙边")
        
        if len(crease_edges) == 0:
            print("无裂隙边，使用原始 Nagata 采样")
            mesh = create_nagata_mesh(vertices, triangles, tri_vertex_normals, tri_face_ids, resolution)
            return mesh, {}
        
        print("计算共享边界系数...")
        c_sharps = compute_c_sharps_for_edges(
            crease_edges, vertices, triangles, tri_vertex_normals, d0
        )
    
    print("采样带折纹修复的 Nagata 曲面...")
    all_vertices = []
    all_faces = []
    face_to_original = []
    vertex_offset = 0
    
    for tri_idx in range(triangles.shape[0]):
        i00, i10, i11 = triangles[tri_idx]
        x00, x10, x11 = vertices[i00], vertices[i10], vertices[i11]
        n00, n10, n11 = tri_vertex_normals[tri_idx]
        
        # 获取三条边的全局键
        edge_keys = (
            tuple(sorted([i00, i10])),  # 边1
            tuple(sorted([i10, i11])),  # 边2
            tuple(sorted([i00, i11])),  # 边3
        )
        
        tri_verts, tri_faces = sample_nagata_triangle_with_crease(
            x00, x10, x11, n00, n10, n11,
            c_sharps, edge_keys, resolution, d0
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
        return pv.PolyData(), c_sharps
    
    pv_faces = np.hstack([
        np.full((all_faces.shape[0], 1), 3, dtype=np.int32),
        all_faces.astype(np.int32)
    ]).flatten()
    
    mesh = pv.PolyData(all_vertices, pv_faces)
    
    if len(face_to_original) > 0:
        mesh.cell_data['face_id'] = tri_face_ids[face_to_original]
        mesh.cell_data['original_tri'] = face_to_original
    
    return mesh, c_sharps



def hierarchical_normal_fusion(normals: np.ndarray) -> np.ndarray:
    """
    分层法向量融合逻辑
    
    1. 第一层融合: 夹角 < 30度
    2. 第二层融合: 夹角 < 60度
    3. 第三层融合: 全局平均
    """
    if len(normals) <= 1:
        return np.mean(normals, axis=0)
    
    # 辅助函数: 聚类并融合
    def fuse_step(current_normals, threshold_deg):
        # 如果只有一个，直接返回
        if len(current_normals) <= 1:
            return current_normals
            
        threshold_rad = np.deg2rad(threshold_deg)
        # Cosine similarity threshold: cos(theta) > cos(threshold)
        # 注意: 夹角越小，cos越大。所以是 > cos_threshold
        cos_threshold = np.cos(threshold_rad)
        
        fused = []
        # 简单的贪婪聚类: 取第一个作为中心，找所有匹配的，平均并在列表中移除
        # 为了避免索引问题，使用mask
        active_mask = np.ones(len(current_normals), dtype=bool)
        
        for i in range(len(current_normals)):
            if not active_mask[i]:
                continue
                
            base = current_normals[i]
            cluster = [base]
            active_mask[i] = False
            
            # 找所有相似的
            for j in range(i + 1, len(current_normals)):
                if not active_mask[j]:
                    continue
                
                target = current_normals[j]
                # Dot product (normalized vectors)
                dot = np.dot(base, target)
                # Clip for numerical stability
                dot = np.clip(dot, -1.0, 1.0)
                
                if dot > cos_threshold:
                    cluster.append(target)
                    active_mask[j] = False
            
            # 融合当前簇
            cluster_mean = np.mean(cluster, axis=0)
            norm = np.linalg.norm(cluster_mean)
            if norm > 1e-12:
                cluster_mean /= norm
            fused.append(cluster_mean)
            
        return np.array(fused)

    # Stage 0: 原始去重 (Tolerance 1e-5 ~ 0.5度)
    # 已经是去重过的输入，但为了保险再做一次
    normals_rounded = np.round(normals, decimals=5)
    _, u_idx = np.unique(normals_rounded, axis=0, return_index=True)
    stage0 = normals[u_idx]
    
    # Stage 1: < 30度融合
    stage1 = fuse_step(stage0, 30.0)
    
    # Stage 2: < 60度融合
    stage2 = fuse_step(stage1, 60.0)
    
    # Stage 3: 全局平均
    final_mean = np.mean(stage2, axis=0)
    norm = np.linalg.norm(final_mean)
    if norm > 1e-12:
        final_mean /= norm
        
    return final_mean


def compute_average_normals(vertices: np.ndarray, triangles: np.ndarray, tri_vertex_normals: np.ndarray) -> np.ndarray:
    """
    计算顶点平均法向量（分层融合策略）
    """
    print("应用 'average' 策略: 分层融合法向量 (<30deg -> <60deg -> Global)...")
    
    # 1. 识别唯一顶点
    vertices_rounded = np.round(vertices, decimals=6)
    _, unique_indices, unique_inverse = np.unique(
        vertices_rounded, axis=0, return_index=True, return_inverse=True
    )
    num_unique = len(unique_indices)
    
    # 2. 收集每个唯一顶点的所有关联法向量
    flat_tri = triangles.flatten() # (3M,)
    flat_normals = tri_vertex_normals.reshape(-1, 3) # (3M, 3)
    
    corner_unique_ids = unique_inverse[flat_tri]
    
    # Sort
    sort_idx = np.argsort(corner_unique_ids)
    sorted_ids = corner_unique_ids[sort_idx]
    sorted_normals = flat_normals[sort_idx]
    
    unique_ids_in_tri, split_indices = np.unique(sorted_ids, return_index=True)
    
    avg_normals = np.zeros((num_unique, 3), dtype=np.float64)
    
    # 3. 遍历唯一顶点，执行分层融合
    for i in range(len(unique_ids_in_tri)):
        uid = unique_ids_in_tri[i]
        start = split_indices[i]
        end = split_indices[i+1] if i+1 < len(split_indices) else len(sorted_ids)
        
        group_normals = sorted_normals[start:end]
        
        # 调用分层融合
        avg_normals[uid] = hierarchical_normal_fusion(group_normals)
    
    # 4. 分配回每个三角形
    new_tri_vertex_normals = avg_normals[unique_inverse[triangles]] # (M, 3, 3)
    
    return new_tri_vertex_normals


def visualize_nagata(
    filepath: str,
    resolution: int = 10,
    scheme: str = 'original',
    show_comparison: bool = True,
    show_edges: bool = False,
    color_by_face_id: bool = False,
    enhance: bool = False,
    bake: bool = False,
    check_self_intersection: bool = False
):
    """
    可视化NSM文件的Nagata曲面
    
    Args:
        filepath: NSM文件路径
        resolution: Nagata采样密度
        show_comparison: 是否显示原始网格对比
        show_edges: 是否显示网格边
        color_by_face_id: 是否按面片ID着色
        enhance: 是否启用折纹裂隙修复
    """
    # 加载NSM数据
    print(f"加载文件: {filepath}")
    mesh_data = load_nsm(filepath)
    
    tri_vertex_normals = mesh_data.tri_vertex_normals
    
    # 如果指定了average策略，重新计算法向量
    if scheme == 'average':
        tri_vertex_normals = compute_average_normals(mesh_data.vertices, mesh_data.triangles, mesh_data.tri_vertex_normals)
    
    print(f"计算Nagata曲面 (分辨率={resolution})...")
    
    original_mesh = create_pyvista_mesh(mesh_data)
    
    if enhance:
        print("启用折纹裂隙修复模式...")
        
        eng_path = get_eng_filepath(filepath)
        cached_c_sharps = None
        if has_cached_data(filepath):
            cached_c_sharps = load_enhanced_data(eng_path)
        
        nagata_mesh = create_nagata_mesh(
            mesh_data.vertices,
            mesh_data.triangles,
            tri_vertex_normals,
            mesh_data.tri_face_ids,
            resolution
        )
        
        enhanced_mesh, c_sharps = create_nagata_mesh_enhanced(
            mesh_data.vertices,
            mesh_data.triangles,
            tri_vertex_normals,
            mesh_data.tri_face_ids,
            resolution,
            cached_c_sharps=cached_c_sharps
        )
        
        if bake and cached_c_sharps is None and c_sharps:
            save_enhanced_data(eng_path, c_sharps)
        
        print(f"Nagata曲面: {nagata_mesh.n_points} 个顶点, {nagata_mesh.n_cells} 个三角形")
        print(f"Enhanced曲面: {enhanced_mesh.n_points} 个顶点, {enhanced_mesh.n_cells} 个三角形")

        if check_self_intersection:
            enhanced_vertices, enhanced_faces = _polydata_to_triangles(enhanced_mesh)
            stats = count_self_intersections(enhanced_vertices, enhanced_faces)
            print(f"Enhanced自交对数: {stats['pairs']}")
            if stats["pairs_sample"]:
                print(f"示例对: {stats['pairs_sample']}")
        
        if color_by_face_id:
            scalars = 'face_id'
            cmap = 'tab20'
        else:
            scalars = None
            cmap = None
        
        plotter = pv.Plotter(shape=(1, 3))
        
        times_font = dict(font='times', font_size=14)
        
        plotter.subplot(0, 0)
        plotter.add_text("Mesh", font='times', font_size=14, position='upper_edge')
        plotter.set_background('white')
        if color_by_face_id:
            plotter.add_mesh(
                original_mesh,
                scalars=scalars,
                cmap=cmap,
                show_edges=show_edges,
                opacity=1.0
            )
        else:
            plotter.add_mesh(
                original_mesh,
                color='lightblue',
                show_edges=show_edges,
                opacity=1.0
            )
        plotter.add_axes()
        
        plotter.subplot(0, 1)
        plotter.add_text("Nagata", font='times', font_size=14, position='upper_edge')
        plotter.set_background('white')
        if color_by_face_id and 'face_id' in nagata_mesh.cell_data:
            plotter.add_mesh(
                nagata_mesh,
                scalars='face_id',
                cmap=cmap,
                show_edges=show_edges,
                opacity=0.5
            )
        else:
            plotter.add_mesh(
                nagata_mesh,
                color='lightblue',
                show_edges=show_edges,
                opacity=0.5
            )
        plotter.add_axes()
        
        plotter.subplot(0, 2)
        plotter.add_text("Enhance", font='times', font_size=14, position='upper_edge')
        plotter.set_background('white')
        if color_by_face_id and 'face_id' in enhanced_mesh.cell_data:
            plotter.add_mesh(
                enhanced_mesh,
                scalars='face_id',
                cmap=cmap,
                show_edges=show_edges,
                opacity=0.5
            )
        else:
            plotter.add_mesh(
                enhanced_mesh,
                color='lightblue',
                show_edges=show_edges,
                opacity=0.5
            )
        plotter.add_axes()
        
        plotter.link_views()
        
    elif show_comparison:
        nagata_mesh = create_nagata_mesh(
            mesh_data.vertices,
            mesh_data.triangles,
            tri_vertex_normals,
            mesh_data.tri_face_ids,
            resolution
        )
        
        print(f"Nagata曲面: {nagata_mesh.n_points} 个顶点, {nagata_mesh.n_cells} 个三角形")
        
        if color_by_face_id:
            scalars = 'face_id'
            cmap = 'tab20'
        else:
            scalars = None
            cmap = None
        
        plotter = pv.Plotter(shape=(1, 2))
        
        plotter.subplot(0, 0)
        plotter.add_text("原始网格", font_size=12, position='upper_edge')
        plotter.set_background('white')
        
        if color_by_face_id:
            plotter.add_mesh(
                original_mesh,
                scalars=scalars,
                cmap=cmap,
                show_edges=show_edges,
                opacity=1.0
            )
        else:
            plotter.add_mesh(
                original_mesh,
                color='lightblue',
                show_edges=show_edges,
                opacity=1.0
            )
        plotter.add_axes()
        
        plotter.subplot(0, 1)
        plotter.add_text("Nagata曲面", font_size=12, position='upper_edge')
        plotter.set_background('white')
        
        if color_by_face_id and 'face_id' in nagata_mesh.cell_data:
            plotter.add_mesh(
                nagata_mesh,
                scalars='face_id',
                cmap=cmap,
                show_edges=show_edges,
                opacity=0.5
            )
        else:
            plotter.add_mesh(
                nagata_mesh,
                color='lightblue',
                show_edges=show_edges,
                opacity=0.5
            )
        plotter.add_axes()
        
        plotter.link_views()
        
    else:
        nagata_mesh = create_nagata_mesh(
            mesh_data.vertices,
            mesh_data.triangles,
            tri_vertex_normals,
            mesh_data.tri_face_ids,
            resolution
        )
        
        print(f"Nagata曲面: {nagata_mesh.n_points} 个顶点, {nagata_mesh.n_cells} 个三角形")
        
        if color_by_face_id:
            scalars = 'face_id'
            cmap = 'tab20'
        else:
            scalars = None
            cmap = None
        
        plotter = pv.Plotter()
        plotter.set_background('white')
        
        if color_by_face_id and 'face_id' in nagata_mesh.cell_data:
            plotter.add_mesh(
                nagata_mesh,
                scalars='face_id',
                cmap=cmap,
                show_edges=show_edges,
                opacity=1.0
            )
            plotter.add_scalar_bar(title='Face ID')
        else:
            plotter.add_mesh(
                nagata_mesh,
                color='lightblue',
                show_edges=show_edges,
                opacity=1.0
            )
        
        plotter.add_axes()
        plotter.add_title(
            f"Nagata曲面 (分辨率={resolution})",
            font_size=12
        )
    
    print("\n交互式可视化已启动:")
    print("  - 左键拖动: 旋转")
    print("  - 右键拖动: 缩放")
    print("  - 中键拖动: 平移")
    print("  - 滚轮: 缩放")
    print("  - 'q': 退出")
    
    plotter.show()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Nagata Patch可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python visualize_nagata.py ../models/nsm/Gear_I.nsm
  python visualize_nagata.py ../models/nsm/Gear_I.nsm --scheme average
  python visualize_nagata.py ../models/nsm/Gear_I.nsm -r 20 --edges
  python visualize_nagata.py ../models/nsm/Gear_I.nsm --no-compare --color-by-id
        """
    )
    
    parser.add_argument('filepath', help='NSM文件路径')
    parser.add_argument('-r', '--resolution', type=int, default=10,
                        help='Nagata采样密度 (默认: 10)')
    parser.add_argument('--scheme', choices=['original', 'average'], default='original',
                        help='法向量策略: original (使用文件值), average (计算平均法向量)')
    parser.add_argument('--no-compare', action='store_true',
                        help='不显示原始网格对比')
    parser.add_argument('--edges', action='store_true',
                        help='显示网格边')
    parser.add_argument('--color-by-id', action='store_true',
                        help='按面片ID着色')
    parser.add_argument('--enhance', action='store_true',
                        help='启用折纹裂隙修复（修复法向不一致造成的裂隙）')
    parser.add_argument('--bake', action='store_true',
                        help='保存增强数据到 .eng 文件（下次可直接加载）')
    parser.add_argument('--check-self-intersection', action='store_true',
                        help='统计增强曲面自交对数')
    
    args = parser.parse_args()
    
    visualize_nagata(
        args.filepath,
        resolution=args.resolution,
        scheme=args.scheme,
        show_comparison=not args.no_compare,
        show_edges=args.edges,
        color_by_face_id=args.color_by_id,
        enhance=args.enhance,
        bake=args.bake,
        check_self_intersection=args.check_self_intersection
    )


if __name__ == '__main__':
    main()
