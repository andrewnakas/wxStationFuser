"""Predictive distributions with closed-form CRPS.

The continuous ranked probability score is the objective every tier is fitted against
and the metric every forecast is judged by (Gneiting & Raftery 2007). Closed forms
matter for two reasons: they make the fit a smooth optimisation instead of a Monte
Carlo one, and they make training cheap enough to refit hundreds of stations inside a
GitHub Actions run.

Distributions:
  * ``normal`` — temperature, and anything else roughly symmetric (Gneiting et al. 2005).
  * ``truncated_normal`` — wind speed and gusts, which cannot go below zero; CRPS from
    Thorarinsdottir & Gneiting (2010).
  * precipitation is handled as a two-part model in ``tier1_emos`` rather than a single
    parametric family, so its pieces (occurrence probability, positive-amount quantiles)
    live there.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

SQRT_PI = np.sqrt(np.pi)


def crps_normal(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """CRPS of N(mu, sigma^2) against observation y.

    CRPS = sigma * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],  z = (y - mu)/sigma
    """
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / SQRT_PI)


def crps_truncated_normal(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """CRPS of a normal truncated to [0, inf), per Thorarinsdottir & Gneiting (2010).

    Used for wind speed and gusts: a Gaussian would put mass below zero and be rewarded
    for it by the score, which shows up as systematically over-wide low-wind intervals.
    """
    sigma = np.maximum(sigma, 1e-6)
    y = np.maximum(y, 0.0)
    m = mu / sigma
    Fm = np.maximum(norm.cdf(m), 1e-10)  # normalising mass above the truncation point
    w = (y - mu) / sigma

    # CRPS = sigma/Fm^2 * { w*Fm*[2*Phi(w) + Fm - 2] + 2*phi(w)*Fm - Phi(sqrt(2)*m)/sqrt(pi) }
    term1 = (y - mu) * (2.0 * norm.cdf(w) + Fm - 2.0) / Fm
    term2 = 2.0 * sigma * norm.pdf(w) / Fm
    term3 = -sigma * norm.cdf(np.sqrt(2.0) * m) / (Fm**2 * SQRT_PI)
    return term1 + term2 + term3


def quantiles_normal(mu: np.ndarray, sigma: np.ndarray, qs: list[float]) -> dict[str, np.ndarray]:
    sigma = np.maximum(sigma, 1e-6)
    return {f"q{int(q * 100):02d}": mu + sigma * norm.ppf(q) for q in qs}


def quantiles_truncated_normal(
    mu: np.ndarray, sigma: np.ndarray, qs: list[float]
) -> dict[str, np.ndarray]:
    """Quantiles of N(mu, sigma^2) truncated to [0, inf).

    F^-1(q) = mu + sigma * Phi^-1( Phi(-mu/sigma) + q * (1 - Phi(-mu/sigma)) )
    """
    sigma = np.maximum(sigma, 1e-6)
    alpha = norm.cdf(-mu / sigma)
    out = {}
    for q in qs:
        p = alpha + q * (1.0 - alpha)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        out[f"q{int(q * 100):02d}"] = np.maximum(mu + sigma * norm.ppf(p), 0.0)
    return out


def crps_ensemble(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """CRPS of an empirical distribution given by ``samples`` (n_obs, n_samples).

    This is the standard (NRG) estimator, dividing by n^2 — not the "fair" estimator,
    which divides the spread term by n(n-1) to correct for finite ensemble size. For the
    large sample counts used here the difference is negligible, but the distinction
    matters if this is ever pointed at a small ensemble, where NRG rewards under-dispersion.
    """
    samples = np.sort(np.asarray(samples, dtype=float), axis=1)
    n = samples.shape[1]
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    term1 = np.abs(samples - y).mean(axis=1)
    # E|X - X'| via the sorted-sample identity, avoiding an O(n^2) pairwise matrix.
    idx = np.arange(1, n + 1)
    weights = (2.0 * idx - n - 1.0)
    term2 = (samples * weights).sum(axis=1) / (n * n)
    return term1 - term2


def crps_point(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """CRPS of a deterministic forecast, which reduces to absolute error.

    This is how a raw model enters the comparison: it offers no distribution, so it is
    scored as a point mass. Stated explicitly because it is the baseline every published
    skill number is measured against.
    """
    return np.abs(np.asarray(y, dtype=float) - np.asarray(x, dtype=float))
