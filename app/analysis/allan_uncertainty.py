"""Uncertainty estimates for overlapping Allan deviation curves."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
from scipy.stats import chi2


ONE_SIGMA_CONFIDENCE = 0.6826894921370859


def white_noise_edf(valid_starts: Iterable[int], order: int) -> Optional[float]:
    """Return moment-matched EDF for valid overlapping windows under white noise.

    Each window is the difference of two adjacent ``order``-sample means divided
    by sqrt(2).  The quadratic-form moments include correlations caused by
    overlapping windows; gaps are represented by omitted start indices.
    """
    n = int(order)
    starts = np.asarray(sorted({int(value) for value in valid_starts}), dtype=int)
    if n < 1 or starts.size == 0:
        return None

    diagonal = 1.0 / n
    gram_square_sum = starts.size * diagonal * diagonal
    span = int(starts[-1]) + 1
    indicator = np.zeros(span, dtype=float)
    indicator[starts] = 1.0
    fft_size = 1 << max(1, (2 * span - 1).bit_length())
    spectrum = np.fft.rfft(indicator, fft_size)
    pair_counts = np.fft.irfft(spectrum * np.conjugate(spectrum), fft_size)
    offsets = np.arange(1, min(2 * n, span), dtype=int)
    if offsets.size:
        overlap = np.maximum(0, n - np.abs(offsets))
        overlap_plus = np.maximum(0, n - np.abs(offsets + n))
        overlap_minus = np.maximum(0, n - np.abs(offsets - n))
        dots = (2 * overlap - overlap_plus - overlap_minus) / (2.0 * n * n)
        gram_square_sum += 2.0 * float(np.sum(np.rint(pair_counts[offsets]) * dots * dots))

    trace = starts.size * diagonal
    edf = trace * trace / gram_square_sum if gram_square_sum > 0 else None
    return float(edf) if edf is not None and math.isfinite(edf) and edf > 0 else None


def chi_square_errors(
    deviation: Optional[float], edf: Optional[float],
    confidence: float = ONE_SIGMA_CONFIDENCE,
) -> dict[str, Optional[float]]:
    """Return asymmetric central chi-square errors for an Allan deviation."""
    if deviation is None or edf is None:
        return {"edf": edf, "ci_lower": None, "ci_upper": None,
                "error_minus": None, "error_plus": None}
    sigma = float(deviation)
    nu = float(edf)
    if not math.isfinite(sigma) or sigma < 0 or not math.isfinite(nu) or nu <= 0:
        return {"edf": edf, "ci_lower": None, "ci_upper": None,
                "error_minus": None, "error_plus": None}
    tail = (1.0 - float(confidence)) / 2.0
    lower_quantile = float(chi2.ppf(tail, nu))
    upper_quantile = float(chi2.ppf(1.0 - tail, nu))
    if lower_quantile <= 0 or not math.isfinite(upper_quantile):
        return {"edf": nu, "ci_lower": None, "ci_upper": None,
                "error_minus": None, "error_plus": None}
    ci_lower = sigma * math.sqrt(nu / upper_quantile)
    ci_upper = sigma * math.sqrt(nu / lower_quantile)
    return {
        "edf": nu,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "error_minus": sigma - ci_lower,
        "error_plus": ci_upper - sigma,
    }
