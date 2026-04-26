"""
scoring.py – Candidate scoring and ranking.

Given a query spectrum and a set of candidate molecules (from retrieval and/or
generation), score each candidate by:

  1. Spectrum similarity     – vector distance in feature space
  2. Forward-model error     – predict the candidate's spectrum and compare
  3. Structural plausibility – SELFIES validity, molecular weight match, etc.

The final score is a weighted combination.  Candidates are returned ranked.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import selfies as sf
from rdkit import Chem

try:
    from nmrsolver import config as C
except Exception:
    try:
        from . import config as C
    except Exception:
        import config as C

try:
    from nmrsolver import retrieval as _retrieval_mod
except Exception:
    try:
        from . import retrieval as _retrieval_mod
    except Exception:
        import retrieval as _retrieval_mod

spectrum_to_vector = _retrieval_mod.spectrum_to_vector


# ──────────────────────────────────────────────────────────────────────────────
#  Score components
# ──────────────────────────────────────────────────────────────────────────────


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def peak_set_distance(query_ppm: List[float], cand_ppm: List[float]) -> float:
    """
    Wasserstein-like distance between two sets of ppm values.

    Uses the greedy best-match approach: for each query peak, find the nearest
    unmatched candidate peak.  Unmatched peaks add a fixed penalty.
    """
    if not query_ppm or not cand_ppm:
        return 1.0

    q = sorted(query_ppm)
    c = sorted(cand_ppm)
    used = [False] * len(c)
    total_error = 0.0

    for qv in q:
        best_dist = float("inf")
        best_j = -1
        for j, cv in enumerate(c):
            if not used[j]:
                d = abs(qv - cv)
                if d < best_dist:
                    best_dist = d
                    best_j = j
        if best_j >= 0 and best_dist < 20.0:  # 20 ppm tolerance
            used[best_j] = True
            total_error += best_dist
        else:
            total_error += 20.0  # penalty for unmatched

    # Penalty for extra candidate peaks
    unmatched_cand = sum(1 for u in used if not u)
    total_error += unmatched_cand * 10.0

    # Normalise by total peaks
    return total_error / (len(q) + len(c) + 1e-6)


def bin_distance(query_bins: List[int], cand_bins: List[int]) -> float:
    """L1 distance between normalised bin histograms."""
    qsum = sum(query_bins) or 1
    csum = sum(cand_bins) or 1
    qn = [v / qsum for v in query_bins]
    cn = [v / csum for v in cand_bins]
    return sum(abs(a - b) for a, b in zip(qn, cn))


def peak_count_error(query_count: int, cand_count: int) -> float:
    """Relative peak count error."""
    return abs(query_count - cand_count) / max(query_count, 1)


def mw_error(query_daltons: float, cand_daltons: float) -> float:
    """Relative molecular weight error."""
    return abs(query_daltons - cand_daltons) / max(query_daltons, 1.0)


def selfies_valid(selfies_str: str) -> bool:
    """Check if a SELFIES string decodes to a valid molecule."""
    try:
        smiles = sf.decoder(selfies_str)
        if smiles is None:
            return False
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
#  Candidate scoring
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ScoredCandidate:
    """A scored candidate molecule."""

    rank: int
    db_id: Optional[int]  # database ID (if from retrieval)
    smiles: str
    selfies: str
    source: str  # "retrieval" | "generated"

    # ── Score components ──────────────────────────────────────────────────────
    spectrum_sim: float  # cosine similarity of spectrum vectors
    peak_dist: float  # peak-set distance
    bin_dist: float  # histogram distance
    count_err: float  # peak count error
    mw_err: float  # molecular weight error
    valid: bool  # SELFIES → valid molecule?

    total_score: float = 0.0  # weighted combination


def score_candidate(
    query_row: Dict[str, Any],
    cand_row: Dict[str, Any],
    source: str = "retrieval",
    weights: Optional[Dict[str, float]] = None,
) -> ScoredCandidate:
    """
    Score a single candidate against the query.

    Args:
        query_row  – the query spectrum (same schema as DB row)
        cand_row   – the candidate molecule
        source     – "retrieval" or "generated"
        weights    – optional weight dict for score components

    Returns:
        ScoredCandidate with computed scores
    """
    if weights is None:
        weights = {
            "spectrum_sim": 0.30,
            "peak_dist": 0.25,
            "bin_dist": 0.20,
            "count_err": 0.10,
            "mw_err": 0.10,
            "valid": 0.05,
        }

    # Spectrum vectors
    q_vec = spectrum_to_vector(query_row)
    c_vec = spectrum_to_vector(cand_row)
    spec_sim = cosine_similarity(q_vec, c_vec)

    # Peak positions
    q_peaks = json.loads(query_row.get("peaks", "[]"))
    c_peaks = json.loads(cand_row.get("peaks", "[]"))
    q_ppm = [p["ppm"] for p in q_peaks if p.get("ppm") is not None]
    c_ppm = [p["ppm"] for p in c_peaks if p.get("ppm") is not None]
    p_dist = peak_set_distance(q_ppm, c_ppm)

    # Bin histogram
    bin_names = [
        "bin_0_50",
        "bin_50_90",
        "bin_90_110",
        "bin_110_165",
        "bin_165_195",
        "bin_195_220",
        "bin_out",
    ]
    q_bins = [query_row.get(b, 0) for b in bin_names]
    c_bins = [cand_row.get(b, 0) for b in bin_names]
    b_dist = bin_distance(q_bins, c_bins)

    # Count error
    cnt_err = peak_count_error(
        query_row.get("peak_count", 0), cand_row.get("peak_count", 0)
    )

    # MW error
    mw_e = mw_error(query_row.get("daltons", 0), cand_row.get("daltons", 0))

    # Validity
    cand_selfies = cand_row.get("selfies", "")
    valid = selfies_valid(cand_selfies)

    # Weighted total (higher = better)
    total = (
        weights["spectrum_sim"] * spec_sim
        + weights["peak_dist"] * max(0, 1 - p_dist)  # invert: lower dist = higher score
        + weights["bin_dist"] * max(0, 1 - b_dist)
        + weights["count_err"] * max(0, 1 - cnt_err)
        + weights["mw_err"] * max(0, 1 - mw_e)
        + weights["valid"] * (1.0 if valid else 0.0)
    )

    return ScoredCandidate(
        rank=0,
        db_id=cand_row.get("id"),
        smiles=cand_row.get("smiles", ""),
        selfies=cand_selfies,
        source=source,
        spectrum_sim=spec_sim,
        peak_dist=p_dist,
        bin_dist=b_dist,
        count_err=cnt_err,
        mw_err=mw_e,
        valid=valid,
        total_score=total,
    )


def rank_candidates(
    query_row: Dict[str, Any],
    candidate_rows: List[Dict[str, Any]],
    source: str = "retrieval",
    weights: Optional[Dict[str, float]] = None,
    top_k: int = 10,
) -> List[ScoredCandidate]:
    """
    Score and rank a list of candidates against the query.

    Returns the top-k candidates sorted by total_score (descending).
    """
    scored = [
        score_candidate(query_row, c, source=source, weights=weights)
        for c in candidate_rows
    ]
    scored.sort(key=lambda s: -s.total_score)
    for i, s in enumerate(scored[:top_k]):
        s.rank = i + 1
    return scored[:top_k]
