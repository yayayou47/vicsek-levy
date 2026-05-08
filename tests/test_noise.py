"""Sanity checks for the alpha-stable generator."""
import numpy as np
import pytest

from noise import stable_rvs


def test_alpha_2_is_gaussian():
    rng = np.random.default_rng(0)
    s = stable_rvs(2.0, 1.0, 200_000, rng)
    # std should be sqrt(2)*1 ; tolerate finite-sample noise
    assert abs(np.std(s) - np.sqrt(2.0)) < 0.05
    # finite kurtosis
    k = ((s - s.mean()) ** 4).mean() / ((s - s.mean()) ** 2).mean() ** 2
    assert 2.5 < k < 3.5


def test_cauchy_has_heavy_tails():
    rng = np.random.default_rng(1)
    s = stable_rvs(1.0, 1.0, 200_000, rng)
    # Cauchy has no finite variance ; sample variance is huge & unstable
    # Robust check: fraction of |s| > 10 is > 5%
    frac = np.mean(np.abs(s) > 10)
    assert frac > 0.04


def test_scale_invariance():
    rng = np.random.default_rng(2)
    s1 = stable_rvs(1.5, 1.0, 50_000, rng)
    s2 = stable_rvs(1.5, 2.0, 50_000, rng)
    # quantiles scale linearly with c
    q1 = np.quantile(np.abs(s1), 0.9)
    q2 = np.quantile(np.abs(s2), 0.9)
    assert 1.7 < q2 / q1 < 2.3


def test_alpha_out_of_range():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        stable_rvs(0.0, 1.0, 10, rng)
    with pytest.raises(ValueError):
        stable_rvs(2.5, 1.0, 10, rng)
