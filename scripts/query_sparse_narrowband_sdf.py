from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from enhanced_nagata_sdf import SparseNarrowbandSDF

def main() -> None:
    parser = argparse.ArgumentParser(description='查询稀疏窄带 SDF')
    parser.add_argument('sdf_npz', type=str)
    parser.add_argument('--point', type=float, nargs=3)
    parser.add_argument('--points-file', type=str)
    args = parser.parse_args()
    sdf = SparseNarrowbandSDF.load_npz(args.sdf_npz)
    if args.point is not None:
        p = np.asarray(args.point, dtype=np.float64)
        v = sdf.query(p)
        print({'point': p.tolist(), 'sdf': None if v is None else float(v)})
        return
    if args.points_file:
        pts = np.loadtxt(args.points_file, dtype=np.float64).reshape(-1, 3)
        vals = sdf.query(pts)
        out = [{'point': p.tolist(), 'sdf': None if v is None else float(v)} for p, v in zip(pts, vals)]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    print('请提供 --point 或 --points-file')

if __name__ == '__main__':
    main()
