"""
retrieval.py – Fast nearest-neighbour candidate retrieval using FAISS.

Encodes each molecule's ¹³C NMR spectrum as a fixed-length vector and builds
an index for sub-second similarity search.

Spectrum vector (dim = retrieval_dim = 128 by default):
  - Histogram bins normalised         (7 dims)
  - Peak count / 50                   (1 dim)
  - Carbon / heavy_atom ratio         (1 dim)
  - Sorted ppm values padded to 50    (50 dims)
  - Multiplicity bag-of-words         (16 dims)
  - Coupling histogram (0–400 Hz)     (10 dims)
  → concatenated and L2-normalised to `retrieval_dim` via a small MLP
    OR used raw (simpler approach, no training needed).
"""

from __future__ import annotations

import json
import os
import pickle
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

try:
    from nmrsolver import config as C
except Exception:
    try:
        from . import config as C
    except Exception:
        import config as C

# ──────────────────────────────────────────────────────────────────────────────
#  Spectrum → fixed vector  (no learned parameters)
# ──────────────────────────────────────────────────────────────────────────────

_RAW_DIM = 95  # 7 + 1 + 1 + 50 + 16 + 10 + 6 + 4 = 95


def spectrum_to_vector(row: Dict[str, Any]) -> np.ndarray:
    """
    Convert one database row into a fixed-length float32 vector for retrieval.
    """
    pc = max(row.get("peak_count", 1), 1)
    cc = max(row.get("carbon_count", 1), 1)

    vec: List[float] = []

    # 1. Normalised bin histogram (7)
    for bname in [
        "bin_0_50",
        "bin_50_90",
        "bin_90_110",
        "bin_110_165",
        "bin_165_195",
        "bin_195_220",
        "bin_out",
    ]:
        vec.append(row.get(bname, 0) / pc)

    # 2. Peak count (1)
    vec.append(pc / 50.0)

    # 3. Carbon / heavy_atom ratio (1)
    ha = max(row.get("heavy_atom_count", 1), 1)
    vec.append(cc / ha)

    # 4. Sorted ppm values, normalised to [0, 1], padded to 50 (50)
    peaks_json = row.get("peaks", "[]")
    peaks = json.loads(peaks_json) if isinstance(peaks_json, str) else peaks_json
    ppm_vals = sorted(
        [float(p["ppm"]) for p in peaks if p.get("ppm") is not None],
        reverse=True,
    )
    ppm_norm = [(v - C.PPM_MIN) / (C.PPM_MAX - C.PPM_MIN) for v in ppm_vals]
    ppm_padded = (ppm_norm[:50] + [0.0] * 50)[:50]
    vec.extend(ppm_padded)

    # 5. Multiplicity bag-of-words (16)
    mult_bow = [0.0] * len(C.MULT_VOCAB)
    for p in peaks:
        m = p.get("multiplicity")
        idx = C.MULT_TO_IDX.get(m, C.MULT_TO_IDX.get("none", 0)) if m else 0
        mult_bow[idx] += 1
    # Normalise
    total = sum(mult_bow) or 1.0
    mult_bow = [v / total for v in mult_bow]
    vec.extend(mult_bow)

    # 6. Coupling histogram (10 bins: 0-40, 40-80, …, 360-400)
    j_hist = [0.0] * 10
    for p in peaks:
        j = p.get("coupling_hz")
        if j is not None:
            if isinstance(j, list):
                j = j[0] if j else 0.0
            b = min(int(j / 40.0), 9)
            j_hist[b] += 1
    j_total = sum(j_hist) or 1.0
    j_hist = [v / j_total for v in j_hist]
    vec.extend(j_hist)

    # 7. Solvent one-hot (6)
    solvent = row.get("solvent", "not_known")
    sol_idx = C.SOLVENT_TO_IDX.get(solvent, 5)
    sol_oh = [0.0] * 6
    sol_oh[min(sol_idx, 5)] = 1.0
    vec.extend(sol_oh)

    # 8. sp3/sp2/sp1 ratios + daltons/1000 (4)
    vec.append(row.get("sp3_c", 0) / cc)
    vec.append(row.get("sp2_c", 0) / cc)
    vec.append(row.get("sp1_c", 0) / cc)
    vec.append(row.get("daltons", 0) / 1000.0)

    return np.array(vec, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  FAISS index
# ──────────────────────────────────────────────────────────────────────────────


class RetrievalIndex:
    """
    Wraps a FAISS index for spectrum-based molecule retrieval.

    Build flow:
        idx = RetrievalIndex.build(rows)
        idx.save("index_dir/")

    Query flow:
        idx = RetrievalIndex.load("index_dir/")
        candidates = idx.search(query_row, top_k=64)
    """

    def __init__(self, index: faiss.Index, id_map: List[int]):
        self.index = index
        self.id_map = id_map  # position → database row id

    @classmethod
    def build(
        cls,
        rows: List[Dict[str, Any]],
        use_gpu: bool = False,
    ) -> "RetrievalIndex":
        """
        Build a FAISS index from database rows.
        """
        print(f"[Retrieval] Encoding {len(rows)} spectra …")
        vectors = np.stack([spectrum_to_vector(r) for r in rows])
        id_map = [r["id"] for r in rows]

        # L2-normalise for cosine similarity
        faiss.normalize_L2(vectors)

        dim = vectors.shape[1]
        # For < 1M vectors, flat index is fine; for more, use IVF
        if len(rows) > 500_000:
            nlist = min(int(len(rows) ** 0.5), 4096)
            quantiser = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(
                quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            print(f"[Retrieval] Training IVF index (nlist={nlist}) …")
            index.train(vectors)
            index.nprobe = min(nlist // 4, 64)
        else:
            index = faiss.IndexFlatIP(dim)

        print("[Retrieval] Adding vectors …")
        index.add(vectors)
        print(f"[Retrieval] Index built: {index.ntotal} vectors, dim={dim}")

        return cls(index, id_map)

    def search(
        self,
        query_row: Dict[str, Any],
        top_k: int = 64,
    ) -> List[Tuple[int, float]]:
        """
        Search for nearest neighbours.

        Returns list of (database_id, similarity_score).
        """
        vec = spectrum_to_vector(query_row).reshape(1, -1)
        faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.id_map[idx], float(score)))
        return results

    def search_vector(
        self,
        vec: np.ndarray,
        top_k: int = 64,
    ) -> List[Tuple[int, float]]:
        """Search from a pre-computed vector."""
        vec = vec.reshape(1, -1).copy()
        faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.id_map[idx], float(score)))
        return results

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dir_path, "faiss.index"))
        with open(os.path.join(dir_path, "id_map.pkl"), "wb") as f:
            pickle.dump(self.id_map, f)
        print(f"[Retrieval] Saved to {dir_path}")

    @classmethod
    def load(cls, dir_path: str) -> "RetrievalIndex":
        index = faiss.read_index(os.path.join(dir_path, "faiss.index"))
        with open(os.path.join(dir_path, "id_map.pkl"), "rb") as f:
            id_map = pickle.load(f)
        print(f"[Retrieval] Loaded: {index.ntotal} vectors")
        return cls(index, id_map)
