"""Parity reports must detect numerical and discrete-data discrepancies."""
import numpy as np

from slice_probe.parity import compare_arrays


def test_parity_distinguishes_roundoff_drift_and_discrete_mismatches():
    original = np.array([1., 2.], dtype=np.float32)
    roundoff = original.copy()
    roundoff[0] += 1e-7
    report = compare_arrays(original, roundoff)
    assert report['matches'] and not report['bitwise_equal']
    assert report['max_absolute_error'] > 0
    drift = original.copy()
    drift[0] += .01
    assert not compare_arrays(original, drift)['matches']
    assert not compare_arrays(original, original.astype('float64'))['matches']
    assert not compare_arrays(original, original[:1])['matches']
    assert not compare_arrays(np.array([1, 2]), np.array([1, 3]))['matches']
