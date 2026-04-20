import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import_sparse_narrowband_sdf():
    from enhanced_nagata_sdf import SparseNarrowbandSDF
    assert SparseNarrowbandSDF is not None


def test_import_enhanced_nagata_backend():
    from enhanced_nagata_sdf import EnhancedNagataBackend
    assert EnhancedNagataBackend is not None


def test_import_nagata_patch():
    from enhanced_nagata_sdf import nagata_patch
    assert nagata_patch is not None


def test_import_nsm_reader():
    from enhanced_nagata_sdf import nsm_reader
    assert nsm_reader is not None
