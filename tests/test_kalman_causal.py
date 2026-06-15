"""
The most important test in the project: prove the Kalman filter is CAUSAL.

If perturbing a future observation changes a past filtered value, the filter is
leaking look-ahead — the most likely explanation for the paper's implausible
Sharpe. These tests guarantee our replication cannot make that mistake.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kdrl.kalman import kalman_filter_1d  # noqa: E402


def _series(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(scale=0.5, size=n)) + 100.0


def test_filter_is_causal():
    z = _series()
    cut = 400
    f1 = kalman_filter_1d(z)
    z2 = z.copy()
    z2[cut:] += 25.0                      # perturb only the FUTURE
    f2 = kalman_filter_1d(z2)
    # everything strictly before the perturbation must be identical
    assert np.allclose(f1[:cut], f2[:cut], atol=0, rtol=0)
    # and the future must actually have changed (sanity)
    assert not np.allclose(f1[cut:], f2[cut:])


def test_seed_and_shape():
    z = _series()
    f = kalman_filter_1d(z)
    assert f.shape == z.shape
    assert np.isfinite(f).all()
    assert f[0] == z[0]                   # first value seeds the filter


def test_denoises():
    # Filtered series should be smoother than the noisy input.
    z = _series()
    f = kalman_filter_1d(z, q=1e-4, r=1e-2)
    assert np.diff(f).var() < np.diff(z).var()
