#!/usr/bin/env python3
"""
Risk-Aware Objective Functions (Section III-D through III-F).

Provides:
  - Rτ: percentile and CVaR tail-risk operators.
  - Risk-shaped latency with dispersion penalty (Eq. 17).
  - Statistical diagnostics: confidence intervals, sample-size adequacy warnings.
  - Objective vector computation from raw profiling samples.

Design note on CVaR sample size (addresses reviewer concern):
  With Nt=300 and τ=0.95, CVaR uses only ~15 tail samples.
  We log a warning when effective tail count < 20 and recommend
  using percentile mode as default, with CVaR for ablation.
"""
from __future__ import annotations

import logging
import math
import statistics
from typing import Dict, List, Sequence, Tuple, Any

import numpy as np

logger = logging.getLogger("tinas.risk")

# =============================================================================
# Core Risk Operators
# =============================================================================

def percentile(x: Sequence[float], tau: float) -> float:
    """τ-th percentile (e.g., τ=0.95 → p95)."""
    if not x:
        return float("inf")
    return float(np.percentile(np.asarray(x, dtype=np.float64),
                               tau * 100.0, method="linear"))


def cvar(x: Sequence[float], tau: float) -> float:
    """
    Conditional Value at Risk: E[X | X ≥ p_τ(X)].
    This is the mean of samples above the τ-th percentile.
    """
    if not x:
        return float("inf")
    p = percentile(x, tau)
    tail = [v for v in x if v >= p]
    n_tail = len(tail)

    # Statistical adequacy warning (Section III-D, reviewer concern)
    if n_tail < 20:
        logger.warning(
            f"CVaR_{tau}: only {n_tail} tail samples from {len(x)} total. "
            f"Estimate is statistically fragile. Consider using percentile mode "
            f"or increasing Nt (currently {len(x)})."
        )

    return float(sum(tail) / n_tail) if tail else p


def R_tau(x: Sequence[float], tau: float, mode: str) -> float:
    """
    Tail-risk operator Rτ (Eq. 6-7).

    Args:
        x: sequence of measurements.
        tau: risk level (e.g., 0.95).
        mode: "percentile" or "cvar".
    """
    mode = mode.lower()
    if mode == "percentile":
        return percentile(x, tau)
    elif mode == "cvar":
        return cvar(x, tau)
    else:
        raise ValueError(f"Unknown risk mode: {mode}. Use 'percentile' or 'cvar'.")


def shaped_latency(x: Sequence[float], tau: float, mode: str,
                   lambda_sigma: float) -> float:
    """
    Risk-shaped latency objective (Eq. 17):
      f̃_lat(α) = Rτ(L) + λ_σ · σ(L)

    The dispersion penalty biases search toward consistent real-time behaviour.
    """
    r = R_tau(x, tau, mode)
    if lambda_sigma > 0 and len(x) > 1:
        r += lambda_sigma * statistics.pstdev(x)
    return float(r)


# =============================================================================
# Objective Vector Computation
# =============================================================================

def compute_objectives(
    mAP: float,
    samples: List[Dict[str, float]],
    tau: float = 0.95,
    risk_mode: str = "percentile",
    lambda_sigma: float = 0.5,
    mem_mode: str = "gpu_alloc",
    energy_mode: str = "risk",
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute the 4-objective vector f(α) from profiling samples.

    Args:
        mAP: detection accuracy (COCO mAP[0.50:0.95]).
        samples: list of per-run dicts from profiler (t_e2e_ms, gpu_alloc_mb, etc.)
        tau: risk level.
        risk_mode: "percentile" or "cvar".
        lambda_sigma: dispersion penalty weight.
        mem_mode: "gpu_alloc" or "max_gpu_cpu".
        energy_mode: "risk" or "mean".

    Returns:
        (objectives, aux) where:
          objectives = {ferr, flat, fmem, feng}
          aux = {p50_ms, p95_ms, p99_ms, cvar95_ms, mean_ms, std_ms, ...}
    """
    # Detection error (Eq. 4)
    ferr = 1.0 - float(mAP)

    # Latency (Eq. 8 + Eq. 17)
    te2e = [s["t_e2e_ms"] for s in samples]
    flat = shaped_latency(te2e, tau, risk_mode, lambda_sigma)

    # Memory (Eq. 9)
    if mem_mode == "gpu_alloc":
        mem = [s["gpu_alloc_mb"] for s in samples]
    else:
        mem = [max(s["gpu_alloc_mb"], s["cpu_rss_mb"]) for s in samples]
    fmem = R_tau(mem, tau, risk_mode)

    # Energy (Eq. 11)
    eng = [s["energy_j"] for s in samples if s.get("energy_j", -1) >= 0]
    if not eng:
        feng = -1.0
    elif energy_mode == "mean":
        feng = float(sum(eng) / len(eng))
    else:
        feng = R_tau(eng, tau, risk_mode)

    objectives = {"ferr": ferr, "flat": flat, "fmem": fmem, "feng": feng}

    # Auxiliary statistics for reporting (Section III-H)
    aux = {
        "mAP": float(mAP),
        "mean_ms": float(np.mean(te2e)) if te2e else -1.0,
        "std_ms": float(np.std(te2e)) if te2e else -1.0,
        "p50_ms": percentile(te2e, 0.50) if te2e else -1.0,
        "p95_ms": percentile(te2e, 0.95) if te2e else -1.0,
        "p99_ms": percentile(te2e, 0.99) if te2e else -1.0,
        "cvar95_ms": cvar(te2e, 0.95) if te2e else -1.0,
        "cv": float(np.std(te2e) / np.mean(te2e)) if te2e and np.mean(te2e) > 0 else -1.0,
        "p99_p50_ratio": percentile(te2e, 0.99) / percentile(te2e, 0.50)
            if te2e and percentile(te2e, 0.50) > 0 else -1.0,
        "mean_pre_ms": float(np.mean([s["t_pre_ms"] for s in samples])),
        "mean_inf_ms": float(np.mean([s["t_inf_ms"] for s in samples])),
        "mean_post_ms": float(np.mean([s["t_post_ms"] for s in samples])),
        "peak_gpu_mb": float(max(s["gpu_alloc_mb"] for s in samples)),
        "peak_cpu_mb": float(max(s["cpu_rss_mb"] for s in samples)),
        "n_samples": len(samples),
        "n_energy_samples": len(eng),
        "mean_energy_j": float(np.mean(eng)) if eng else -1.0,
        "mean_power_w": float(np.mean([s["mean_power_w"] for s in samples
                                        if s.get("mean_power_w", -1) > 0])) or -1.0,
    }

    return objectives, aux


# =============================================================================
# Latency Distribution Diagnostics (for paper Section VI: "let the data speak")
# =============================================================================

def latency_diagnostics(te2e: Sequence[float], label: str = "") -> Dict[str, Any]:
    """
    Compute diagnostics to determine if tail-risk objectives are empirically
    justified. This directly addresses the question: "is distributional
    treatment overkill on this hardware?"

    Returns dict with:
      - cv: coefficient of variation (σ/μ)
      - p99_p50_ratio: how heavy the tail is
      - iqr_ratio: (p75-p25)/p50
      - is_tail_significant: True if p99/p50 > 1.05 (i.e., tails matter)
    """
    if len(te2e) < 30:
        return {"error": "need ≥30 samples for diagnostics"}

    arr = np.asarray(te2e, dtype=np.float64)
    mu = float(np.mean(arr))
    std = float(np.std(arr))
    p50 = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))
    p25 = float(np.percentile(arr, 25))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))

    diag = {
        "label": label,
        "n": len(te2e),
        "mean_ms": mu,
        "std_ms": std,
        "cv": std / mu if mu > 0 else 0.0,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "p99_p50_ratio": p99 / p50 if p50 > 0 else 0.0,
        "p95_p50_ratio": p95 / p50 if p50 > 0 else 0.0,
        "iqr_ratio": (p75 - p25) / p50 if p50 > 0 else 0.0,
        "is_tail_significant": (p99 / p50 > 1.05) if p50 > 0 else False,
    }
    return diag


def bootstrap_ci(x: Sequence[float], statistic: str = "p95",
                 n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a tail-risk statistic.

    Returns (point_estimate, ci_lower, ci_upper).
    Useful for assessing whether CVaR/p95 differences between candidates
    are statistically significant.
    """
    rng = np.random.RandomState(seed)
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)

    stat_fn = {
        "p95": lambda a: float(np.percentile(a, 95)),
        "p99": lambda a: float(np.percentile(a, 99)),
        "mean": lambda a: float(np.mean(a)),
        "cvar95": lambda a: float(np.mean(a[a >= np.percentile(a, 95)])),
    }[statistic]

    boot_stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_stats.append(stat_fn(arr[idx]))

    boot_stats = sorted(boot_stats)
    lo = boot_stats[int(n_boot * alpha / 2)]
    hi = boot_stats[int(n_boot * (1 - alpha / 2))]
    point = stat_fn(arr)

    return point, lo, hi
