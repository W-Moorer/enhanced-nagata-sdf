from pathlib import Path
import sys
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from enhanced_nagata_sdf import EnhancedNagataBackend, SparseNarrowbandBuildConfig, build_sparse_narrowband_sdf

def main():
    nsm_path = 'models/sphere.nsm'
    output_path = 'outputs/example_sdf.npz'
    if not Path(nsm_path).exists():
        raise FileNotFoundError(nsm_path)
    backend = EnhancedNagataBackend(nsm_path, use_cache=True, bake_cache=True)
    cfg = SparseNarrowbandBuildConfig(tau=0.02, block_size=0.02, block_resolution=8, k_nearest=16)
    sdf, metadata = build_sparse_narrowband_sdf(backend, cfg)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sdf.save_npz(output_path, metadata=metadata)
    sample_points = np.array([[0.0,0.0,0.0],[0.3,0.0,0.0],[0.5,0.0,0.0],[0.7,0.0,0.0]])
    values = sdf.query(sample_points)
    for p, v in zip(sample_points, values):
        print(f'{p.tolist()} -> {v}')

if __name__ == '__main__':
    main()
