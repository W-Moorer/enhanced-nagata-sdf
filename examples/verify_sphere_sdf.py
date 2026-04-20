from pathlib import Path
import sys, json
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from enhanced_nagata_sdf import EnhancedNagataBackend, SparseNarrowbandBuildConfig, build_sparse_narrowband_sdf, load_nsm_lightweight

def main():
    mesh = load_nsm_lightweight('models/sphere.nsm')
    center = mesh.vertices.mean(axis=0)
    radius = np.linalg.norm(mesh.vertices - center[None, :], axis=1).mean()
    backend = EnhancedNagataBackend('models/sphere.nsm', use_cache=False, force_recompute=True, bake_cache=True)
    cfg = SparseNarrowbandBuildConfig(tau=0.05*radius, block_size=0.05*radius, block_resolution=8, k_nearest=16)
    sdf, metadata = build_sparse_narrowband_sdf(backend, cfg)
    R = sdf.R
    points = []
    for block_coord in sdf.block_coords:
        for i in range(R+1):
            for j in range(R+1):
                for k in range(R+1):
                    points.append((block_coord + np.array([i,j,k], dtype=float)/R) * sdf.block_size)
    points = np.asarray(points)
    values = np.asarray(sdf.query(points, return_none_outside=False), dtype=np.float64)
    ref = np.linalg.norm(points - center[None,:], axis=1) - radius
    ref_used = -ref if np.mean(np.abs(values + ref)) < np.mean(np.abs(values - ref)) else ref
    abs_err = np.abs(values - ref_used)
    report = {'num_points': int(points.shape[0]), 'mean_abs_err': float(np.mean(abs_err)), 'rmse': float(np.sqrt(np.mean((values-ref_used)**2))), 'p95_abs_err': float(np.percentile(abs_err,95)), 'max_abs_err': float(np.max(abs_err)), 'radius': float(radius), 'center': center.tolist()}
    Path('outputs').mkdir(exist_ok=True)
    out = Path('outputs/sphere_verification_report.json')
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f'报告已写出: {out}')

if __name__ == '__main__':
    main()
