"""
predict.py – End-to-end inference pipeline for the NMR Solver.

Usage:
    # By database row ID
    python -m nmrsolver.predict --id 42

    # From a JSON peaks file
    python -m nmrsolver.predict --spectrum spectrum.json

    # With retrieval candidates
    python -m nmrsolver.predict --id 42 --retrieval --retrieval-index indexes/index.faiss

    # Quick test (no forward model re-scoring)
    python -m nmrsolver.predict --id 42 --no-rescore

Pipeline:
    1. Encode the query spectrum
    2. (optional) Retrieve nearest neighbours from FAISS index
    3. Generate candidates via inverse model beam search
    4. (optional) Re-score all candidates with the forward model
    5. Rank and display results
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import selfies as sf
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

# Robust package-relative imports
try:
    from nmrsolver import config as C
except Exception:
    try:
        from . import config as C
    except Exception:
        import config as C

try:
    from nmrsolver import data as _data_mod
except Exception:
    try:
        from . import data as _data_mod
    except Exception:
        import data as _data_mod

SelfiesVocab = _data_mod.SelfiesVocab
collate_fn = _data_mod.collate_fn
encode_formula_vector = _data_mod.encode_formula_vector
encode_global_features = _data_mod.encode_global_features
encode_spectrum = _data_mod.encode_spectrum
load_rows = _data_mod.load_rows

try:
    from nmrsolver import models as _models_mod
except Exception:
    try:
        from . import models as _models_mod
    except Exception:
        import models as _models_mod

ForwardModel = _models_mod.ForwardModel
InverseModel = _models_mod.InverseModel

try:
    from nmrsolver import retrieval as _retrieval_mod
except Exception:
    try:
        from . import retrieval as _retrieval_mod
    except Exception:
        import retrieval as _retrieval_mod

RetrievalIndex = _retrieval_mod.RetrievalIndex
spectrum_to_vector = _retrieval_mod.spectrum_to_vector

try:
    from nmrsolver import scoring as _scoring_mod
except Exception:
    try:
        from . import scoring as _scoring_mod
    except Exception:
        import scoring as _scoring_mod

ScoredCandidate = _scoring_mod.ScoredCandidate
rank_candidates = _scoring_mod.rank_candidates
score_candidate = _scoring_mod.score_candidate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
#  Load models
# ──────────────────────────────────────────────────────────────────────────────


def load_inverse_model(ckpt_path: str, vocab: SelfiesVocab) -> InverseModel:
    """Load a trained inverse model from checkpoint."""
    model = InverseModel(vocab_size=len(vocab))
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def load_forward_model(ckpt_path: str, vocab: SelfiesVocab) -> ForwardModel:
    """Load a trained forward model from checkpoint."""
    model = ForwardModel(vocab_size=len(vocab))
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
#  Query construction
# ──────────────────────────────────────────────────────────────────────────────


def query_from_db_id(
    row_id: int, db_path: str = C.DB_PATH, table: str = C.TABLE_NAME
) -> Dict[str, Any]:
    """Fetch a single row from the database by ID."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    row = con.execute(f'SELECT * FROM "{table}" WHERE id = ?', (row_id,)).fetchone()
    con.close()
    if row is None:
        raise ValueError(f"Row {row_id} not found in {table}")
    return dict(row)


def query_from_json(
    spectrum_path: str,
    solvent: str = "not_known",
    daltons: float = 0.0,
    carbon_count: int = 0,
    heavy_atom_count: int = 0,
    formula: str = "",
) -> Dict[str, Any]:
    """
    Construct a query row from a JSON spectrum file.

    The file should be a JSON array of peaks, e.g.:
        [{"ppm": 25.3, "multiplicity": "q", "coupling_hz": 7.1}, ...]

    Additional metadata can be provided via arguments.
    """
    with open(spectrum_path) as f:
        peaks = json.load(f)

    # Compute bin counts from peaks
    bin_ranges = [
        ("bin_0_50", 0, 50),
        ("bin_50_90", 50, 90),
        ("bin_90_110", 90, 110),
        ("bin_110_165", 110, 165),
        ("bin_165_195", 165, 195),
        ("bin_195_220", 195, 220),
    ]
    bins = {name: 0 for name, _, _ in bin_ranges}
    bins["bin_out"] = 0

    for p in peaks:
        ppm = p.get("ppm", 0)
        placed = False
        for name, lo, hi in bin_ranges:
            if lo <= ppm < hi:
                bins[name] += 1
                placed = True
                break
        if not placed:
            bins["bin_out"] += 1

    row = {
        "id": -1,
        "peaks": json.dumps(peaks),
        "peak_count": len(peaks),
        "carbon_count": carbon_count or len(peaks),  # heuristic fallback
        "heavy_atom_count": heavy_atom_count,
        "daltons": daltons,
        "sp3_c": 0,
        "sp2_c": 0,
        "sp1_c": 0,
        "solvent": solvent,
        "formula": formula,
        "smiles": "",
        "selfies": "",
        "inchi": "",
        **bins,
    }
    return row


# ──────────────────────────────────────────────────────────────────────────────
#  Prepare tensors from a query row
# ──────────────────────────────────────────────────────────────────────────────


def prepare_query_tensors(row: Dict[str, Any]):
    """
    Convert a query row to model input tensors.

    Returns (spec_tensor, spec_mask, global_feats, formula_vector) all on DEVICE with batch dim.
    """
    spec = encode_spectrum(row["peaks"])
    spec = spec[: C.MAX_PEAKS]
    spec_t = torch.tensor(spec, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # Pad to a consistent shape
    n_peaks = spec_t.size(1)
    if n_peaks == 0:
        spec_t = torch.zeros(1, 1, 4, dtype=torch.long, device=DEVICE)
        spec_mask = torch.ones(1, 1, dtype=torch.bool, device=DEVICE)
    else:
        spec_mask = torch.zeros(1, n_peaks, dtype=torch.bool, device=DEVICE)

    gf = (
        torch.tensor(encode_global_features(row), dtype=torch.float32)
        .unsqueeze(0)
        .to(DEVICE)
    )
    formula_vec = (
        torch.tensor(encode_formula_vector(row), dtype=torch.float32)
        .unsqueeze(0)
        .to(DEVICE)
    )

    return spec_t, spec_mask, gf, formula_vec


# ──────────────────────────────────────────────────────────────────────────────
#  Candidate generation
# ──────────────────────────────────────────────────────────────────────────────


def generate_candidates(
    inverse_model: InverseModel,
    vocab: SelfiesVocab,
    spec_t: torch.Tensor,
    spec_mask: torch.Tensor,
    gf: torch.Tensor,
    formula_vec: Optional[torch.Tensor] = None,
    beam_size: int = 10,
    length_penalty: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Generate candidate molecules via beam search.

    Returns a list of dicts with keys: selfies, smiles, log_prob, valid.
    """
    beams = inverse_model.generate_beam(
        spec_t,
        spec_mask,
        gf,
        formula_vec,
        beam_size=beam_size,
        bos_idx=vocab.bos_idx,
        eos_idx=vocab.eos_idx,
        length_penalty=length_penalty,
        num_return_sequences=beam_size,
    )

    candidates = []
    seen = set()

    for beam_list in beams:  # one per batch item
        for token_ids, log_prob in beam_list:
            selfies_str = vocab.decode(token_ids)
            try:
                smiles = sf.decoder(selfies_str) or ""
            except Exception:
                smiles = ""

            # Canonicalise to dedup
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is not None:
                canon = Chem.MolToSmiles(mol)
            else:
                canon = smiles

            if canon in seen:
                continue
            seen.add(canon)

            # Build a pseudo-row for scoring
            cand_row = _smiles_to_row(canon, selfies_str)
            cand_row["_log_prob"] = log_prob
            cand_row["_source"] = "generated"
            candidates.append(cand_row)

    return candidates


def _smiles_to_row(smiles: str, selfies_str: str = "") -> Dict[str, Any]:
    """Build a minimal row dict from a SMILES for scoring purposes."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None

    if mol is not None:
        daltons = Descriptors.MolWt(mol)
        heavy = mol.GetNumHeavyAtoms()
        # Count carbons
        carbons = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    else:
        daltons = 0.0
        heavy = 0
        carbons = 0

    return {
        "id": None,
        "smiles": smiles,
        "selfies": selfies_str or (sf.encoder(smiles) if smiles else ""),
        "formula": "",
        "peaks": "[]",
        "peak_count": 0,
        "carbon_count": carbons,
        "heavy_atom_count": heavy,
        "daltons": daltons,
        "sp3_c": 0,
        "sp2_c": 0,
        "sp1_c": 0,
        "solvent": "not_known",
        "bin_0_50": 0,
        "bin_50_90": 0,
        "bin_90_110": 0,
        "bin_110_165": 0,
        "bin_165_195": 0,
        "bin_195_220": 0,
        "bin_out": 0,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Re-scoring with forward model
# ──────────────────────────────────────────────────────────────────────────────


def rescore_with_forward(
    forward_model: ForwardModel,
    vocab: SelfiesVocab,
    query_row: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Re-score candidates by predicting their spectra with the forward model
    and comparing to the query spectrum.

    Adds '_fwd_score' key to each candidate dict.
    """
    if not candidates:
        return candidates

    # Encode query spectrum as a target for comparison
    query_peaks = json.loads(query_row.get("peaks", "[]"))
    query_ppm = sorted([p["ppm"] for p in query_peaks if p.get("ppm") is not None])

    bin_names = [
        "bin_0_50",
        "bin_50_90",
        "bin_90_110",
        "bin_110_165",
        "bin_165_195",
        "bin_195_220",
        "bin_out",
    ]
    query_bins = [query_row.get(b, 0) for b in bin_names]
    query_bin_sum = sum(query_bins) or 1
    query_bins_norm = [b / query_bin_sum for b in query_bins]

    for cand in candidates:
        selfies_str = cand.get("selfies", "")
        if not selfies_str:
            cand["_fwd_score"] = 0.0
            continue

        # Encode and predict
        sf_ids = vocab.encode(selfies_str)
        sf_ids = sf_ids[: C.MAX_SELFIES_LEN]
        sf_t = torch.tensor([sf_ids], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            preds = forward_model(sf_t)

        # Compare bins
        pred_bins = preds["bins"][0].cpu().numpy()
        pred_bins_sum = max(pred_bins.sum(), 1e-6)
        pred_bins_norm = pred_bins / pred_bins_sum
        bin_sim = 1.0 - np.sum(np.abs(pred_bins_norm - np.array(query_bins_norm))) / 2

        # Compare peak count
        pred_count = max(preds["count"][0, 0].item(), 0)
        true_count = query_row.get("peak_count", 0)
        count_sim = 1.0 - abs(pred_count - true_count) / max(true_count, 1)
        count_sim = max(count_sim, 0)

        # Compare peak positions
        pred_peaks_norm = preds["peaks"][0].cpu().numpy()
        pred_ppm = sorted(pred_peaks_norm[: int(pred_count + 0.5)] * C.PPM_MAX)
        if query_ppm and len(pred_ppm) > 0:
            # Simple average pairwise distance
            min_len = min(len(query_ppm), len(pred_ppm))
            ppm_err = (
                sum(abs(query_ppm[i] - pred_ppm[i]) for i in range(min_len)) / min_len
            )
            ppm_sim = max(0, 1.0 - ppm_err / 50.0)  # normalise
        else:
            ppm_sim = 0.0

        cand["_fwd_score"] = 0.4 * bin_sim + 0.3 * count_sim + 0.3 * ppm_sim

    return candidates


# ──────────────────────────────────────────────────────────────────────────────
#  Full prediction pipeline
# ──────────────────────────────────────────────────────────────────────────────


def predict(
    query_row: Dict[str, Any],
    inverse_ckpt: str,
    vocab_path: str,
    forward_ckpt: Optional[str] = None,
    retrieval_index_path: Optional[str] = None,
    beam_size: int = 10,
    top_k: int = 10,
    length_penalty: float = 0.7,
    use_retrieval: bool = False,
    use_rescore: bool = True,
    db_path: str = C.DB_PATH,
) -> List[ScoredCandidate]:
    """
    Full prediction pipeline.

    Args:
        query_row           – the query spectrum (dict with DB schema)
        inverse_ckpt        – path to inverse model checkpoint
        vocab_path          – path to vocabulary file
        forward_ckpt        – path to forward model checkpoint (for re-scoring)
        retrieval_index_path – path to FAISS index (for retrieval candidates)
        beam_size           – beam search width
        top_k               – number of results to return
        use_retrieval       – whether to use FAISS retrieval
        use_rescore         – whether to use forward model re-scoring
        db_path             – database path (for retrieval candidate lookup)

    Returns:
        List of ScoredCandidate, ranked by total_score
    """
    # ── Load vocab ────────────────────────────────────────────────────────
    vocab = SelfiesVocab.load(vocab_path)
    print(f"[Predict] Vocab: {len(vocab)} tokens")

    # ── Load inverse model ────────────────────────────────────────────────
    inv_model = load_inverse_model(inverse_ckpt, vocab)
    print("[Predict] Inverse model loaded")

    # ── Prepare query ─────────────────────────────────────────────────────
    spec_t, spec_mask, gf, formula_vec = prepare_query_tensors(query_row)

    # ── Generate candidates (beam search) ─────────────────────────────────
    generation_beam_size = max(beam_size, top_k)
    gen_cands = generate_candidates(
        inv_model,
        vocab,
        spec_t,
        spec_mask,
        gf,
        formula_vec=formula_vec,
        beam_size=generation_beam_size,
        length_penalty=length_penalty,
    )
    print(f"[Predict] Generated {len(gen_cands)} unique candidates via beam search")

    # ── Retrieval candidates ──────────────────────────────────────────────
    ret_cands: List[Dict[str, Any]] = []
    if use_retrieval and retrieval_index_path:
        index = RetrievalIndex.load(retrieval_index_path)
        q_vec = spectrum_to_vector(query_row)
        results = index.search(q_vec, top_k=top_k * 2)

        # Fetch full rows for retrieved IDs
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        for row_id, score in results:
            r = con.execute(
                f'SELECT * FROM "{C.TABLE_NAME}" WHERE id = ?', (row_id,)
            ).fetchone()
            if r is not None:
                d = dict(r)
                d["_source"] = "retrieval"
                d["_retrieval_score"] = score
                ret_cands.append(d)
        con.close()
        print(f"[Predict] Retrieved {len(ret_cands)} candidates from index")

    # ── Merge candidates ──────────────────────────────────────────────────
    all_cands = gen_cands + ret_cands

    # ── Forward model re-scoring ──────────────────────────────────────────
    if use_rescore and forward_ckpt and os.path.exists(forward_ckpt):
        fwd_model = load_forward_model(forward_ckpt, vocab)
        print("[Predict] Forward model loaded, re-scoring…")
        all_cands = rescore_with_forward(fwd_model, vocab, query_row, all_cands)
    else:
        for c in all_cands:
            c["_fwd_score"] = 0.5  # neutral score

    # ── Score & rank ──────────────────────────────────────────────────────
    scored = rank_candidates(query_row, all_cands, source="mixed", top_k=top_k)

    # Blend in forward-model score and log-prob if available
    for sc in scored:
        # Find matching candidate to get extra scores
        for c in all_cands:
            if c.get("smiles") == sc.smiles:
                fwd = c.get("_fwd_score", 0.5)
                lp = c.get("_log_prob", 0.0)
                # Blend: 70% base score + 20% forward score + 10% normalised log-prob
                lp_norm = 1.0 / (1.0 + np.exp(-lp / 10.0))  # sigmoid normalise
                sc.total_score = 0.70 * sc.total_score + 0.20 * fwd + 0.10 * lp_norm
                break

    # Re-sort after blending
    scored.sort(key=lambda s: -s.total_score)
    for i, s in enumerate(scored):
        s.rank = i + 1

    return scored


# ──────────────────────────────────────────────────────────────────────────────
#  Pretty-print results
# ──────────────────────────────────────────────────────────────────────────────


def print_results(
    scored: List[ScoredCandidate],
    query_row: Optional[Dict[str, Any]] = None,
):
    """Pretty-print ranked prediction results."""
    if query_row:
        print(f"\n{'─'*72}")
        print(
            f"  QUERY: id={query_row.get('id')}  "
            f"peaks={query_row.get('peak_count')}  "
            f"carbons={query_row.get('carbon_count')}  "
            f"daltons={query_row.get('daltons')}"
        )
        if query_row.get("smiles"):
            print(f"  TRUE:  {query_row['smiles']}")
        print(f"{'─'*72}")

    print(
        f"\n  {'Rank':<5} {'Score':>6} {'SpecSim':>8} {'PkDist':>8} "
        f"{'BinDist':>8} {'Valid':>5}  SMILES"
    )
    print(f"  {'─'*5} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*5}  {'─'*40}")

    for s in scored:
        valid_str = "✓" if s.valid else "✗"
        smiles_short = s.smiles[:50] + ("…" if len(s.smiles) > 50 else "")
        print(
            f"  {s.rank:<5} {s.total_score:>6.3f} {s.spectrum_sim:>8.3f} "
            f"{s.peak_dist:>8.3f} {s.bin_dist:>8.3f} {valid_str:>5}  "
            f"{smiles_short}"
        )

    print()


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="NMR Solver – predict molecular structure from spectrum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--id", type=int, help="Database row ID to use as query")
    input_group.add_argument(
        "--spectrum", type=str, help="Path to JSON file with peak list"
    )

    parser.add_argument(
        "--inverse-ckpt",
        type=str,
        default=os.path.join(C.CHECKPOINT_DIR, "inverse", "best.pt"),
        help="Inverse model checkpoint",
    )
    parser.add_argument(
        "--forward-ckpt",
        type=str,
        default=os.path.join(C.CHECKPOINT_DIR, "forward", "best.pt"),
        help="Forward model checkpoint",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=os.path.join(C.CHECKPOINT_DIR, "vocab.pkl"),
        help="Vocabulary file",
    )
    parser.add_argument(
        "--retrieval",
        action="store_true",
        help="Use FAISS retrieval for additional candidates",
    )
    parser.add_argument(
        "--retrieval-index",
        type=str,
        default=os.path.join(C.INDEX_DIR, "retrieval.faiss"),
        help="Path to FAISS index",
    )
    parser.add_argument(
        "--no-rescore",
        action="store_true",
        help="Skip forward model re-scoring",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=10,
        help="Beam search width",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to show",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=0.7,
        help="Length penalty used during beam search ranking",
    )
    parser.add_argument(
        "--solvent",
        type=str,
        default="not_known",
        help="Solvent (if using --spectrum)",
    )
    parser.add_argument(
        "--daltons",
        type=float,
        default=0.0,
        help="Molecular weight (if using --spectrum)",
    )
    parser.add_argument(
        "--formula",
        type=str,
        default="",
        help="Exact molecular formula constraint (if using --spectrum)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=C.DB_PATH,
        help="Database path",
    )

    args = parser.parse_args()

    # Build query
    if args.id is not None:
        query_row = query_from_db_id(args.id, args.db)
    else:
        query_row = query_from_json(
            args.spectrum,
            solvent=args.solvent,
            daltons=args.daltons,
            formula=args.formula,
        )

    # Run prediction
    scored = predict(
        query_row,
        inverse_ckpt=args.inverse_ckpt,
        vocab_path=args.vocab,
        forward_ckpt=args.forward_ckpt if not args.no_rescore else None,
        retrieval_index_path=args.retrieval_index if args.retrieval else None,
        beam_size=args.beam_size,
        top_k=args.top_k,
        length_penalty=args.length_penalty,
        use_retrieval=args.retrieval,
        use_rescore=not args.no_rescore,
        db_path=args.db,
    )

    # Display
    print_results(scored, query_row)


if __name__ == "__main__":
    main()
