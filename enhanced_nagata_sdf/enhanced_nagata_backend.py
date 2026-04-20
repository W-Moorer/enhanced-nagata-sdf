from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
for _p in (_THIS_DIR, _PARENT_DIR):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from nagata_patch import (  # type: ignore
    NagataModelQuery,
    compute_crease_direction,
    compute_c_sharp,
    compute_curvature,
    evaluate_nagata_derivatives,
    evaluate_nagata_patch,
    evaluate_nagata_patch_with_crease,
    nagata_patch,
)
from nagata_storage import get_eng_filepath, has_cached_data, load_enhanced_data, save_enhanced_data  # type: ignore


@dataclass
class NSMMeshDataLite:
    vertices: np.ndarray
    triangles: np.ndarray
    tri_face_ids: np.ndarray
    tri_vertex_normals: np.ndarray


@dataclass
class QueryResult:
    point: np.ndarray
    nearest_point: np.ndarray
    distance: float
    signed_distance: float
    normal: np.ndarray
    triangle_index: int
    uv: Tuple[float, float]
    feature_type: str


@dataclass
class BackendBuildInfo:
    nsm_path: str
    num_vertices: int
    num_triangles: int
    num_crease_edges: int
    used_cache: bool
    eng_path: str


def load_nsm_lightweight(filepath: str) -> NSMMeshDataLite:
    with open(filepath, 'rb') as f:
        header = f.read(64)
        if len(header) != 64:
            raise ValueError('NSM header 不完整')
        magic = header[0:4].decode('ascii', errors='ignore')
        version = struct.unpack('<I', header[4:8])[0]
        nv = struct.unpack('<I', header[8:12])[0]
        nt = struct.unpack('<I', header[12:16])[0]
        if magic != 'NSM\x00' or version != 1:
            raise ValueError('无效 NSM 文件')
        vertices = np.fromfile(f, dtype=np.float64, count=nv * 3).reshape(nv, 3)
        triangles = np.fromfile(f, dtype=np.uint32, count=nt * 3).reshape(nt, 3)
        tri_face_ids = np.fromfile(f, dtype=np.uint32, count=nt)
        tri_vertex_normals = np.fromfile(f, dtype=np.float64, count=nt * 9).reshape(nt, 3, 3)
    return NSMMeshDataLite(vertices, triangles, tri_face_ids, tri_vertex_normals)


def detect_crease_edges(vertices, triangles, tri_vertex_normals, gap_threshold=1e-4):
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

    crease_edges = {}

    def get_normal_at_vertex(tri_idx, global_v_idx):
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
        c_L = compute_curvature(e, n_A_L, n_B_L)
        c_R = compute_curvature(e, n_A_R, n_B_R)
        max_gap = 0.0
        for t in np.linspace(0.0, 1.0, 11):
            p_L = (1 - t) * A + t * B - c_L * t * (1 - t)
            p_R = (1 - t) * A + t * B - c_R * t * (1 - t)
            max_gap = max(max_gap, float(np.linalg.norm(p_L - p_R)))
        if max_gap > gap_threshold:
            crease_edges[edge_key] = {
                'A': A, 'B': B,
                'n_A_L': n_A_L, 'n_A_R': n_A_R,
                'n_B_L': n_B_L, 'n_B_R': n_B_R,
                'tri_L': int(tri_L), 'tri_R': int(tri_R),
                'max_gap': max_gap,
            }
    return crease_edges


def _edge_index_for_triangle(tri: np.ndarray, edge_key: tuple) -> int:
    if tuple(sorted([int(tri[0]), int(tri[1])])) == edge_key:
        return 0
    if tuple(sorted([int(tri[1]), int(tri[2])])) == edge_key:
        return 1
    if tuple(sorted([int(tri[0]), int(tri[2])])) == edge_key:
        return 2
    return -1


def _sample_uv_for_edge(edge_idx: int, steps: int = 5, eps: float = 0.05):
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


def _check_edge_constraints_for_triangle(x00, x10, x11, c1_orig, c2_orig, c3_orig, edge_idx, c_sharp_edge, k_factor, eps=1e-10):
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
    if np.dot(side_dir, opp - 0.5 * (a + b)) < 0:
        side_dir = -side_dir

    for uu, vv in _sample_uv_for_edge(edge_idx):
        p = evaluate_nagata_patch_with_crease(
            x00, x10, x11, c1_orig, c2_orig, c3_orig,
            c_sharps[0], c_sharps[1], c_sharps[2],
            tuple(is_crease), np.array([uu]), np.array([vv]), k_factor,
            enforce_constraints=False,
        ).flatten()
        dXdu, dXdv = evaluate_nagata_derivatives(
            x00, x10, x11, c1_orig, c2_orig, c3_orig, uu, vv,
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


def compute_c_sharps_for_edges(crease_edges, vertices, triangles, tri_vertex_normals, k_factor=0.0):
    c_sharps = {}
    for edge_key, info in crease_edges.items():
        A, B = info['A'], info['B']
        e = B - A
        d_A = compute_crease_direction(info['n_A_L'], info['n_A_R'], e)
        d_B = compute_crease_direction(info['n_B_L'], info['n_B_R'], e)
        if float(np.dot(d_A, d_B)) < 0.0:
            d_B = -d_B
        c_sharp = compute_c_sharp(A, B, d_A, d_B)

        tri_L = int(info['tri_L'])
        tri_R = int(info['tri_R'])
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
            mid = 0.5 * (low + high)
            candidate = baseline + mid * (c_sharp - baseline)
            ok_L = _check_edge_constraints_for_triangle(x00_L, x10_L, x11_L, c1_L, c2_L, c3_L, edge_idx_L, candidate, k_factor)
            ok_R = _check_edge_constraints_for_triangle(x00_R, x10_R, x11_R, c1_R, c2_R, c3_R, edge_idx_R, candidate, k_factor)
            if ok_L and ok_R:
                best = candidate
                low = mid
            else:
                high = mid
        c_sharps[edge_key] = best
    return c_sharps


class EnhancedNagataBackend:
    def __init__(self, nsm_path: str, *, use_cache: bool = True, force_recompute: bool = False, bake_cache: bool = False, gap_threshold: float = 1e-4, k_factor: float = 0.0) -> None:
        self.nsm_path = str(nsm_path)
        self.mesh = load_nsm_lightweight(self.nsm_path)
        self.gap_threshold = float(gap_threshold)
        self.k_factor = float(k_factor)
        self.eng_path = get_eng_filepath(self.nsm_path)
        self.used_cache = False
        self.c_sharps = self._load_or_build_c_sharps(use_cache=use_cache, force_recompute=force_recompute, bake_cache=bake_cache)

        self.query_model = NagataModelQuery(
            vertices=self.mesh.vertices,
            triangles=self.mesh.triangles,
            tri_vertex_normals=self.mesh.tri_vertex_normals,
            nsm_filepath=None,
        )
        self.query_model.crease_map = dict(self.c_sharps)
        self.query_model._build_feature_bvh()
        self.patch_coeffs = self.query_model.patch_coeffs
        self.build_info = BackendBuildInfo(
            nsm_path=self.nsm_path,
            num_vertices=int(self.mesh.vertices.shape[0]),
            num_triangles=int(self.mesh.triangles.shape[0]),
            num_crease_edges=int(len(self.c_sharps)),
            used_cache=bool(self.used_cache),
            eng_path=self.eng_path,
        )

    def _load_or_build_c_sharps(self, *, use_cache: bool, force_recompute: bool, bake_cache: bool):
        if use_cache and (not force_recompute) and has_cached_data(self.nsm_path):
            cached = load_enhanced_data(self.eng_path)
            if cached is not None:
                self.used_cache = True
                return cached
        crease_edges = detect_crease_edges(self.mesh.vertices, self.mesh.triangles, self.mesh.tri_vertex_normals, gap_threshold=self.gap_threshold)
        c_sharps = compute_c_sharps_for_edges(crease_edges, self.mesh.vertices, self.mesh.triangles, self.mesh.tri_vertex_normals, k_factor=self.k_factor)
        if bake_cache and c_sharps:
            save_enhanced_data(self.eng_path, c_sharps)
        return c_sharps

    def estimate_patch_aabb(self, tri_idx: int, pad: float = 0.0):
        tri = self.mesh.triangles[int(tri_idx)]
        pts = [self.mesh.vertices[int(tri[0])], self.mesh.vertices[int(tri[1])], self.mesh.vertices[int(tri[2])]]
        sample_uvs = [(0.666, 0.333), (0.5, 0.0), (1.0, 0.5), (0.5, 0.5), (0.25, 0.05), (0.95, 0.25), (0.25, 0.22)]
        for u, v in sample_uvs:
            pts.append(self.evaluate_patch(tri_idx, u, v)['point'])
        pts_arr = np.stack(pts, axis=0)
        return np.min(pts_arr, axis=0) - float(pad), np.max(pts_arr, axis=0) + float(pad)

    def enumerate_active_blocks(self, tau: float, block_size: float):
        active = set()
        for tri_idx in range(self.mesh.triangles.shape[0]):
            aabb_min, aabb_max = self.estimate_patch_aabb(tri_idx, pad=tau)
            lo = np.floor(aabb_min / block_size).astype(int)
            hi = np.floor(aabb_max / block_size).astype(int)
            for i in range(int(lo[0]), int(hi[0]) + 1):
                for j in range(int(lo[1]), int(hi[1]) + 1):
                    for k in range(int(lo[2]), int(hi[2]) + 1):
                        active.add((i, j, k))
        return sorted(active)

    def get_patch_crease_info(self, tri_idx: int):
        return self.query_model._get_patch_crease_info(int(tri_idx))

    def evaluate_patch(self, tri_idx: int, u: float, v: float):
        tri = self.mesh.triangles[int(tri_idx)]
        x00 = self.mesh.vertices[int(tri[0])]
        x10 = self.mesh.vertices[int(tri[1])]
        x11 = self.mesh.vertices[int(tri[2])]
        c1, c2, c3 = self.patch_coeffs[int(tri_idx)]
        is_crease, c_sharps = self.get_patch_crease_info(int(tri_idx))
        if any(is_crease):
            point = evaluate_nagata_patch_with_crease(
                x00, x10, x11, c1, c2, c3,
                c_sharps[0], c_sharps[1], c_sharps[2],
                is_crease, np.array([u]), np.array([v]), self.k_factor,
            ).reshape(3)
            du, dv = evaluate_nagata_derivatives(x00, x10, x11, c1, c2, c3, float(u), float(v), is_crease=is_crease, c_sharps=c_sharps, k_factor=self.k_factor)
        else:
            point = evaluate_nagata_patch(x00, x10, x11, c1, c2, c3, np.array([u]), np.array([v])).reshape(3)
            du, dv = evaluate_nagata_derivatives(x00, x10, x11, c1, c2, c3, float(u), float(v))
        n = np.cross(du, dv)
        n_norm = float(np.linalg.norm(n))
        if n_norm > 1e-12:
            n = n / n_norm
        else:
            n = np.array([0.0, 0.0, 1.0], dtype=float)
        jacobian_signed = float(np.dot(np.cross(du, dv), n))
        return {'point': point, 'du': du, 'dv': dv, 'normal': n, 'jacobian_signed': jacobian_signed, 'triangle_index': int(tri_idx), 'uv': (float(u), float(v))}

    def query_point(self, point: np.ndarray, k_nearest: int = 16) -> QueryResult:
        point = np.asarray(point, dtype=float).reshape(3)
        raw = self.query_model.query(point, k_nearest=int(k_nearest))
        nearest_point = np.asarray(raw['nearest_point'], dtype=float).reshape(3)
        normal = np.asarray(raw['normal'], dtype=float).reshape(3)
        distance = float(raw['distance'])
        sign = float(np.sign(np.dot(point - nearest_point, normal)))
        if sign == 0.0:
            sign = 1.0
        return QueryResult(
            point=point,
            nearest_point=nearest_point,
            distance=distance,
            signed_distance=sign * distance,
            normal=normal,
            triangle_index=int(raw.get('triangle_index', -1)),
            uv=tuple(raw.get('uv', (0.0, 0.0))),
            feature_type=str(raw.get('feature_type', 'FACE')),
        )
