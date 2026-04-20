from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from .enhanced_nagata_backend import EnhancedNagataBackend


@dataclass
class SparseNarrowbandBuildConfig:
    tau: float = 0.02
    block_size: float = 0.02
    block_resolution: int = 8
    k_nearest: int = 16
    max_blocks: int = -1
    clip_to_tau: bool = True


class SparseNarrowbandSDF:
    def __init__(self, *, tau: float, block_size: float, block_resolution: int, block_coords: np.ndarray, sdf_values: np.ndarray) -> None:
        self.tau = float(tau)
        self.block_size = float(block_size)
        self.block_resolution = int(block_resolution)
        self.block_coords = np.asarray(block_coords, dtype=np.int32)
        self.sdf_values = np.asarray(sdf_values, dtype=np.float32)
        self._block_map: Dict[Tuple[int, int, int], int] = {tuple(map(int, bc)): i for i, bc in enumerate(self.block_coords)}

    @property
    def R(self) -> int:
        return self.block_resolution

    def world_to_block(self, x: np.ndarray) -> Tuple[int, int, int]:
        return tuple(np.floor(np.asarray(x, dtype=np.float64) / self.block_size).astype(np.int32).tolist())

    def block_origin(self, block: Tuple[int, int, int]) -> np.ndarray:
        return np.asarray(block, dtype=np.float64) * self.block_size

    def _query_single(self, x: np.ndarray, *, return_none_outside: bool = True) -> Optional[float]:
        x = np.asarray(x, dtype=np.float64).reshape(3)
        block = self.world_to_block(x)
        idx = self._block_map.get(block)
        if idx is None:
            return None if return_none_outside else float('nan')

        origin = self.block_origin(block)
        local = np.clip((x - origin) / self.block_size, 0.0, 1.0)
        R = self.block_resolution
        grid = self.sdf_values[idx]
        gx, gy, gz = local * R
        i0, j0, k0 = int(np.floor(gx)), int(np.floor(gy)), int(np.floor(gz))
        i1, j1, k1 = min(i0 + 1, R), min(j0 + 1, R), min(k0 + 1, R)
        tx, ty, tz = gx - i0, gy - j0, gz - k0

        c000 = float(grid[i0, j0, k0]); c100 = float(grid[i1, j0, k0])
        c010 = float(grid[i0, j1, k0]); c110 = float(grid[i1, j1, k0])
        c001 = float(grid[i0, j0, k1]); c101 = float(grid[i1, j0, k1])
        c011 = float(grid[i0, j1, k1]); c111 = float(grid[i1, j1, k1])

        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        return c0 * (1 - tz) + c1 * tz

    def query(self, points: Union[np.ndarray, List[float], List[List[float]]], *, return_none_outside: bool = True):
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim == 1:
            return self._query_single(arr, return_none_outside=return_none_outside)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError('points must have shape (3,) or (N,3)')
        vals = [self._query_single(p, return_none_outside=return_none_outside) for p in arr]
        if return_none_outside:
            return vals
        return np.asarray(vals, dtype=np.float64)

    def save_npz(self, out_path: str | Path, metadata: Optional[dict] = None) -> None:
        metadata = dict(metadata or {})
        metadata.update({
            'tau': self.tau,
            'block_size': self.block_size,
            'block_resolution': self.block_resolution,
            'num_blocks': int(self.block_coords.shape[0]),
            'format': 'sparse_narrowband_sdf_v1',
        })
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, block_coords=self.block_coords, sdf_values=self.sdf_values, metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))

    @classmethod
    def load_npz(cls, path: str | Path) -> 'SparseNarrowbandSDF':
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data['metadata_json']))
        return cls(
            tau=float(metadata['tau']),
            block_size=float(metadata['block_size']),
            block_resolution=int(metadata['block_resolution']),
            block_coords=data['block_coords'],
            sdf_values=data['sdf_values'],
        )


def _sample_block_nodes(block: Tuple[int, int, int], block_size: float, block_resolution: int) -> np.ndarray:
    origin = np.asarray(block, dtype=np.float64) * block_size
    axis = np.linspace(0.0, 1.0, block_resolution + 1, dtype=np.float64)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing='ij')
    local = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    return origin[None, :] + local * block_size


def build_sparse_narrowband_sdf(backend: EnhancedNagataBackend, config: SparseNarrowbandBuildConfig):
    active_blocks = backend.enumerate_active_blocks(tau=config.tau, block_size=config.block_size)
    if config.max_blocks > 0:
        active_blocks = active_blocks[: int(config.max_blocks)]
    R = int(config.block_resolution)
    block_coords = np.asarray(active_blocks, dtype=np.int32) if active_blocks else np.zeros((0, 3), dtype=np.int32)
    sdf_values = np.zeros((len(active_blocks), R + 1, R + 1, R + 1), dtype=np.float32)

    print(f'活跃块数: {len(active_blocks)}')
    print(f'每块节点数: {(R + 1) ** 3}')

    queried_nodes = 0
    clipped_nodes = 0
    feature_hist = {}
    progress_step = max(1, len(active_blocks) // 10) if active_blocks else 1
    for bi, block in enumerate(active_blocks):
        pts = _sample_block_nodes(block, config.block_size, R)
        vals = np.zeros((pts.shape[0],), dtype=np.float32)
        for pi, p in enumerate(pts):
            q = backend.query_point(p, k_nearest=config.k_nearest)
            val = float(q.signed_distance)
            if config.clip_to_tau:
                new_val = float(np.clip(val, -config.tau, config.tau))
                if new_val != val:
                    clipped_nodes += 1
                val = new_val
            vals[pi] = val
            queried_nodes += 1
            feature_hist[q.feature_type] = feature_hist.get(q.feature_type, 0) + 1
        sdf_values[bi] = vals.reshape(R + 1, R + 1, R + 1)
        if (bi + 1) % progress_step == 0 or (bi + 1) == len(active_blocks):
            print(f'已处理块 {bi + 1}/{len(active_blocks)}')

    sdf = SparseNarrowbandSDF(
        tau=config.tau,
        block_size=config.block_size,
        block_resolution=config.block_resolution,
        block_coords=block_coords,
        sdf_values=sdf_values,
    )
    metadata = {
        'num_active_blocks': int(len(active_blocks)),
        'queried_nodes': int(queried_nodes),
        'clipped_nodes': int(clipped_nodes),
        'feature_type_hist': feature_hist,
        'backend': {
            'nsm_path': backend.build_info.nsm_path,
            'eng_path': backend.build_info.eng_path,
            'num_vertices': backend.build_info.num_vertices,
            'num_triangles': backend.build_info.num_triangles,
            'num_crease_edges': backend.build_info.num_crease_edges,
            'used_cache': backend.build_info.used_cache,
        },
    }
    return sdf, metadata
