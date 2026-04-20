import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhanced_nagata_sdf.sparse_narrowband_sdf import SparseNarrowbandSDF


def test_sdf_construction():
    block_coords = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int32)
    R = 4
    sdf_values = np.random.randn(2, R+1, R+1, R+1).astype(np.float64)
    sdf = SparseNarrowbandSDF(block_coords=block_coords, sdf_values=sdf_values, block_size=1.0, block_resolution=R, tau=0.5)
    assert len(sdf.block_coords) == 2
    assert sdf.sdf_values.shape == (2, R+1, R+1, R+1)
    assert sdf.block_size == 1.0
    assert sdf.R == R


def test_sdf_query_single_point():
    block_coords = np.array([[0, 0, 0]], dtype=np.int32)
    R = 4
    block_size = 1.0
    sdf_values = np.zeros((1, R+1, R+1, R+1), dtype=np.float64)
    for i in range(R+1):
        z = i / R * block_size
        sdf_values[0, :, :, i] = z - 0.5
    sdf = SparseNarrowbandSDF(block_coords=block_coords, sdf_values=sdf_values, block_size=block_size, block_resolution=R, tau=1.0)
    result = sdf.query(np.array([0.5, 0.5, 0.5]))
    assert abs(result) < 0.1


def test_sdf_query_batch():
    block_coords = np.array([[0, 0, 0]], dtype=np.int32)
    R = 4
    sdf_values = np.random.randn(1, R+1, R+1, R+1).astype(np.float64)
    sdf = SparseNarrowbandSDF(block_coords=block_coords, sdf_values=sdf_values, block_size=1.0, block_resolution=R, tau=1.0)
    results = sdf.query(np.array([[0.5,0.5,0.5],[0.2,0.3,0.4],[0.8,0.7,0.6]]), return_none_outside=False)
    assert results.shape == (3,)


def test_sdf_out_of_bounds():
    block_coords = np.array([[0, 0, 0]], dtype=np.int32)
    R = 4
    sdf_values = np.random.randn(1, R+1, R+1, R+1).astype(np.float64)
    sdf = SparseNarrowbandSDF(block_coords=block_coords, sdf_values=sdf_values, block_size=1.0, block_resolution=R, tau=1.0)
    result = sdf.query(np.array([10.0, 10.0, 10.0]))
    assert result is None


def test_sdf_save_and_load(tmp_path):
    block_coords = np.array([[0, 0, 0]], dtype=np.int32)
    R = 4
    sdf_values = np.random.randn(1, R+1, R+1, R+1).astype(np.float64)
    sdf = SparseNarrowbandSDF(block_coords=block_coords, sdf_values=sdf_values, block_size=1.0, block_resolution=R, tau=1.0)
    path = tmp_path / 'test.npz'
    sdf.save_npz(path)
    loaded = SparseNarrowbandSDF.load_npz(path)
    np.testing.assert_array_equal(loaded.block_coords, sdf.block_coords)
    np.testing.assert_array_almost_equal(loaded.sdf_values, sdf.sdf_values)
    assert loaded.R == sdf.R
