from __future__ import annotations
import argparse, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from enhanced_nagata_sdf import EnhancedNagataBackend, SparseNarrowbandBuildConfig, build_sparse_narrowband_sdf

def main() -> None:
    parser = argparse.ArgumentParser(description='构建可直接查询的稀疏窄带 SDF')
    parser.add_argument('nsm', type=str, help='输入 .nsm 文件路径')
    parser.add_argument('output', type=str, help='输出 .npz 文件路径')
    parser.add_argument('--tau', type=float, default=0.02)
    parser.add_argument('--block-size', type=float, default=0.02)
    parser.add_argument('--block-resolution', type=int, default=8)
    parser.add_argument('--k-nearest', type=int, default=16)
    parser.add_argument('--gap-threshold', type=float, default=1e-4)
    parser.add_argument('--k-factor', type=float, default=0.0)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--force-recompute', action='store_true')
    parser.add_argument('--bake-cache', action='store_true')
    parser.add_argument('--max-blocks', type=int, default=-1)
    parser.add_argument('--no-clip', action='store_true')
    args = parser.parse_args()

    backend = EnhancedNagataBackend(
        args.nsm,
        use_cache=not args.no_cache,
        force_recompute=args.force_recompute,
        bake_cache=args.bake_cache,
        gap_threshold=args.gap_threshold,
        k_factor=args.k_factor,
    )
    cfg = SparseNarrowbandBuildConfig(
        tau=args.tau,
        block_size=args.block_size,
        block_resolution=args.block_resolution,
        k_nearest=args.k_nearest,
        max_blocks=args.max_blocks,
        clip_to_tau=not args.no_clip,
    )
    sdf, metadata = build_sparse_narrowband_sdf(backend, cfg)
    out = Path(args.output)
    sdf.save_npz(out, metadata=metadata)
    print(f'写出完成: {out}')
    print(f'活跃块: {sdf.block_coords.shape[0]}')

if __name__ == '__main__':
    main()
