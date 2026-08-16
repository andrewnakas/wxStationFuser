"""The closed-form CRPS formulas are the foundation everything else is fitted against.

If these are wrong, every tier optimises the wrong objective and every published skill
number is meaningless — and the failure would be silent, because a wrong-but-smooth
objective still converges. So they are checked against numerical integration of the
definition, CRPS = int (F(x) - 1{x >= y})^2 dx, rather than against remembered algebra.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, truncnorm

from wxfuser.models.distributions import (
    crps_ensemble,
    crps_normal,
    crps_point,
    crps_truncated_normal,
    quantiles_normal,
    quantiles_truncated_normal,
)


def numeric_crps(cdf, y, lo, hi):
    val, _ = quad(lambda x: (cdf(x) - (1.0 if x >= y else 0.0)) ** 2, lo, hi, limit=400)
    return val


@pytest.mark.parametrize(
    ("mu", "sigma", "y"),
    [(0.0, 1.0, 0.5), (5.0, 2.0, 3.0), (-3.0, 0.7, -2.0), (12.0, 4.0, 20.0)],
)
def test_crps_normal_matches_numeric(mu, sigma, y):
    expected = numeric_crps(lambda x: norm.cdf(x, mu, sigma), y, mu - 15 * sigma, mu + 15 * sigma)
    got = float(crps_normal(np.array([y]), np.array([mu]), np.array([sigma]))[0])
    assert got == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    ("mu", "sigma", "y"),
    [(3.0, 2.0, 4.0), (1.0, 2.0, 0.5), (-1.0, 2.0, 1.0), (8.0, 3.0, 2.0), (0.5, 1.5, 0.0)],
)
def test_crps_truncated_normal_matches_numeric(mu, sigma, y):
    a = (0.0 - mu) / sigma
    expected = numeric_crps(
        lambda x: truncnorm.cdf(x, a, np.inf, loc=mu, scale=sigma), y, 0.0, mu + 20 * sigma
    )
    got = float(crps_truncated_normal(np.array([y]), np.array([mu]), np.array([sigma]))[0])
    assert got == pytest.approx(expected, abs=1e-6)


def test_crps_is_minimised_by_the_true_distribution():
    """A proper score cannot be gamed: claiming the wrong spread must cost you.

    This is the property that makes minimum-CRPS fitting meaningful, so it is worth
    asserting directly rather than trusting the formula.
    """
    rng = np.random.default_rng(0)
    y = rng.normal(2.0, 1.5, size=40000)
    mu = np.full_like(y, 2.0)
    honest = crps_normal(y, mu, np.full_like(y, 1.5)).mean()
    overconfident = crps_normal(y, mu, np.full_like(y, 0.6)).mean()
    underconfident = crps_normal(y, mu, np.full_like(y, 4.0)).mean()
    assert honest < overconfident
    assert honest < underconfident


def test_crps_point_is_absolute_error():
    """How a deterministic model enters the comparison; the baseline depends on it."""
    y = np.array([1.0, -2.0, 3.5])
    x = np.array([0.0, 0.0, 3.0])
    assert crps_point(y, x) == pytest.approx(np.array([1.0, 2.0, 0.5]))


def test_crps_ensemble_converges_to_closed_form():
    rng = np.random.default_rng(3)
    samples = rng.normal(1.0, 2.0, size=(1, 20000))
    got = float(crps_ensemble(np.array([1.5]), samples)[0])
    expected = float(crps_normal(np.array([1.5]), np.array([1.0]), np.array([2.0]))[0])
    assert got == pytest.approx(expected, rel=0.02)


def test_normal_quantiles_are_ordered_and_centred():
    qs = [0.05, 0.25, 0.5, 0.75, 0.95]
    out = quantiles_normal(np.array([10.0]), np.array([2.0]), qs)
    vals = [out[f"q{int(q * 100):02d}"][0] for q in qs]
    assert vals == sorted(vals)
    assert out["q50"][0] == pytest.approx(10.0)


def test_truncated_normal_quantiles_never_go_negative():
    """Wind speed below zero is the failure this distribution exists to prevent."""
    qs = [0.05, 0.5, 0.95]
    out = quantiles_truncated_normal(np.array([0.4]), np.array([3.0]), qs)
    for q in qs:
        assert out[f"q{int(q * 100):02d}"][0] >= 0.0
