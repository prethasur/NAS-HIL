#!/usr/bin/env python3
"""
Probabilistic Dominance & Diversity Mechanisms (Section IV-E).

Implements:
  - Bootstrap-based probabilistic dominance P(α ≺ β) (Eq. 15-16)
  - Reference-vector entropy niching (Eq. 18)
  - Budget-adaptive resampling decisions (Eq. 19)
  - Crowding distance (fallback for within-niche selection)
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .risk import R_tau, shaped_latency


# =============================================================================
# Probabilistic Dominance (Section IV-E.2)
# =============================================================================

def dominates_point(a: List[float], b: List[float]) -> bool:
    """Standard Pareto dominance on point vectors: a ≺ b."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def bootstrap_dominance_prob(
    cand_a: Dict[str, Any],
    cand_b: Dict[str, Any],
    objectives: List[str],
    tau: float,
    risk_mode: str,
    lambda_sigma: float,
    n_draws: int = 200,
    rng: random.Random | None = None,
) -> float:
    """
    Estimate P(α ≺ β) via bootstrap resampling of hardware measurements (Eq. 15).

    For each bootstrap draw:
      1. Resample latency/memory/energy measurements with replacement
      2. Recompute risk-shaped objectives
      3. Check if a dominates b under this realization

    Returns: estimated probability P(α ≺ β) ∈ [0, 1].
    """
    r = rng or random.Random()

    def extract(cand, key):
        """Extract measurement samples for a given objective."""
        samples = cand.get("profiling_samples", [])
        if key == "flat":
            return [s["t_e2e_ms"] for s in samples]
        elif key == "fmem":
            return [max(s["gpu_alloc_mb"], s.get("cpu_rss_mb", 0)) for s in samples]
        elif key == "feng":
            return [s["energy_j"] for s in samples if s.get("energy_j", -1) >= 0]
        return []

    def boot(xs: List[float]) -> List[float]:
        if not xs:
            return []
        return [xs[r.randrange(len(xs))] for _ in range(len(xs))]

    count = 0
    for _ in range(n_draws):
        vec_a, vec_b = [], []

        for obj in objectives:
            if obj == "ferr":
                # Accuracy is deterministic (single mAP value), no resampling
                vec_a.append(cand_a["objectives"]["ferr"])
                vec_b.append(cand_b["objectives"]["ferr"])
            elif obj == "flat":
                sa, sb = boot(extract(cand_a, "flat")), boot(extract(cand_b, "flat"))
                va = shaped_latency(sa, tau, risk_mode, lambda_sigma) if sa else float("inf")
                vb = shaped_latency(sb, tau, risk_mode, lambda_sigma) if sb else float("inf")
                vec_a.append(va)
                vec_b.append(vb)
            elif obj == "fmem":
                sa, sb = boot(extract(cand_a, "fmem")), boot(extract(cand_b, "fmem"))
                va = R_tau(sa, tau, risk_mode) if sa else float("inf")
                vb = R_tau(sb, tau, risk_mode) if sb else float("inf")
                vec_a.append(va)
                vec_b.append(vb)
            elif obj == "feng":
                sa, sb = boot(extract(cand_a, "feng")), boot(extract(cand_b, "feng"))
                if sa and sb:
                    va = R_tau(sa, tau, risk_mode)
                    vb = R_tau(sb, tau, risk_mode)
                else:
                    va = cand_a["objectives"].get("feng", 0)
                    vb = cand_b["objectives"].get("feng", 0)
                vec_a.append(va)
                vec_b.append(vb)

        if dominates_point(vec_a, vec_b):
            count += 1

    return count / n_draws


# =============================================================================
# PRB Non-Dominated Sorting
# =============================================================================

def prb_nondominated_sort(
    population: List[Dict[str, Any]],
    objectives: List[str],
    tau: float,
    risk_mode: str,
    lambda_sigma: float,
    eta: float = 0.70,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> Tuple[List[List[int]], Dict[Tuple[int, int], float]]:
    """
    Non-dominated sorting with probabilistic dominance (Section IV-E).

    Instead of deterministic a≺b, we use:
      α ≺_p β  iff  P(α ≺ β) ≥ η     (Eq. 16)

    Returns:
      fronts: list of lists of population indices, front 0 = Pareto front.
      p_cache: dict mapping (i,j) → P(i ≺ j) for reuse.
    """
    n = len(population)
    rng = random.Random(seed)
    p_cache: Dict[Tuple[int, int], float] = {}

    def get_pdom(i: int, j: int) -> float:
        key = (i, j)
        if key not in p_cache:
            p_cache[key] = bootstrap_dominance_prob(
                population[i], population[j],
                objectives, tau, risk_mode, lambda_sigma,
                n_draws=n_bootstrap, rng=rng,
            )
        return p_cache[key]

    # NSGA-II-style fast non-dominated sort with probabilistic dominance
    S = [[] for _ in range(n)]  # solutions dominated by i
    n_dom = [0] * n              # number of solutions dominating i
    fronts: List[List[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p_ij = get_pdom(i, j)
            p_ji = get_pdom(j, i)
            if p_ij >= eta and p_ji < eta:
                S[i].append(j)
            elif p_ji >= eta and p_ij < eta:
                n_dom[i] += 1

        if n_dom[i] == 0:
            fronts[0].append(i)

    # Build subsequent fronts
    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    next_front.append(j)
        k += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return fronts, p_cache


# =============================================================================
# Reference-Vector Entropy Niching (Section IV-E.4, Eq. 18)
# =============================================================================

def generate_reference_vectors(n_obj: int, n_divisions: int = 12) -> np.ndarray:
    """
    Generate uniformly distributed reference vectors for niching.
    Uses Das-Dennis method for n_obj dimensions.
    """
    from itertools import combinations_with_replacement

    # Simplex-lattice design
    points = []
    for combo in combinations_with_replacement(range(n_divisions + 1), n_obj - 1):
        point = []
        prev = 0
        for c in combo:
            point.append(c - prev)
            prev = c
        point.append(n_divisions - prev)
        points.append([p / n_divisions for p in point])

    W = np.array(points, dtype=np.float64)
    # Normalize
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return W / norms


def assign_niches(
    population: List[Dict[str, Any]],
    objectives: List[str],
    ref_vectors: np.ndarray,
) -> List[int]:
    """
    Assign each candidate to its nearest reference vector (niche).
    Returns list of niche indices.
    """
    n = len(population)
    n_obj = len(objectives)

    # Build objective matrix (normalised)
    obj_matrix = np.zeros((n, n_obj))
    for i, cand in enumerate(population):
        for j, obj in enumerate(objectives):
            obj_matrix[i, j] = cand["objectives"].get(obj, float("inf"))

    # Normalise to [0, 1] per objective
    mins = obj_matrix.min(axis=0)
    maxs = obj_matrix.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    norm = (obj_matrix - mins) / ranges

    # Assign to nearest reference vector via cosine similarity
    niche_ids = []
    for i in range(n):
        v = norm[i]
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            niche_ids.append(0)
            continue
        cos_sim = ref_vectors @ v / (np.linalg.norm(ref_vectors, axis=1) * v_norm + 1e-12)
        niche_ids.append(int(np.argmax(cos_sim)))

    return niche_ids


def architecture_entropy(genome_tokens: List[str]) -> float:
    """
    Eq. 18: H(α) = -Σ p_q log(p_q + ε)
    Measures diversity of operator types in a genome.
    """
    if not genome_tokens:
        return 0.0
    counts = Counter(genome_tokens)
    total = sum(counts.values())
    eps = 1e-10
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log(p + eps)
    return entropy


# =============================================================================
# Budget-Adaptive Resampling Decision (Section IV-E.5, Eq. 19)
# =============================================================================

def identify_uncertain_candidates(
    front_indices: List[int],
    p_cache: Dict[Tuple[int, int], float],
    delta: float = 0.08,
) -> List[int]:
    """
    Identify candidates in a front whose dominance relationships are uncertain.
    A candidate is uncertain if:
      max_β |P(α ≺ β) - 0.5| < δ

    These need more measurements to confidently determine their rank.
    """
    uncertain = []
    for i in front_indices:
        best_margin = 1.0
        for j in front_indices:
            if i == j:
                continue
            for key in [(i, j), (j, i)]:
                if key in p_cache:
                    margin = abs(p_cache[key] - 0.5)
                    best_margin = min(best_margin, margin)
        if best_margin < delta:
            uncertain.append(i)
    return uncertain


# =============================================================================
# Crowding Distance (fallback within-niche selection)
# =============================================================================

def crowding_distance(
    population: List[Dict[str, Any]],
    indices: List[int],
    objectives: List[str],
) -> Dict[int, float]:
    """NSGA-II crowding distance for within-niche tie-breaking."""
    if len(indices) <= 2:
        return {i: float("inf") for i in indices}

    cd = {i: 0.0 for i in indices}

    for obj in objectives:
        vals = [(i, population[i]["objectives"].get(obj, float("inf"))) for i in indices]
        vals.sort(key=lambda x: x[1])

        cd[vals[0][0]] = float("inf")
        cd[vals[-1][0]] = float("inf")

        obj_range = vals[-1][1] - vals[0][1]
        if obj_range == 0:
            continue

        for k in range(1, len(vals) - 1):
            cd[vals[k][0]] += (vals[k + 1][1] - vals[k - 1][1]) / obj_range

    return cd
