"""
data.py – Data loading, tokenisation, and PyTorch datasets.

Handles:
  1. SQLite → raw records
  2. Spectrum tokenisation  (ppm → bin, multiplicity → idx, J → bin)
  3. SELFIES tokenisation   (string → token IDs, with vocab building)
  4. Global feature vector   (carbon_count, sp3_c, bins, solvent, …)
  5. PyTorch Dataset + DataLoader factories
  6. Data augmentation       (ppm noise, mult dropout, peak dropout)
"""

from __future__ import annotations

import json
import math
import os
import pickle
import random
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import selfies as sf
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from . import config as C

# ──────────────────────────────────────────────────────────────────────────────
#  1. SQLite loading
# ──────────────────────────────────────────────────────────────────────────────


def load_rows(
    db_path: str = C.DB_PATH,
    table: str = C.TABLE_NAME,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load all rows from the database as a list of dicts."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    q = f'SELECT * FROM "{table}"'
    if limit:
        q += f" LIMIT {limit}"
    rows = [dict(r) for r in con.execute(q).fetchall()]
    con.close()
    return rows


# ──────────────────────────────────────────────────────────────────────────────
#  2. Spectrum tokenisation
# ──────────────────────────────────────────────────────────────────────────────


def ppm_to_bin(ppm: float) -> int:
    """Discretise a ppm value into a bin index."""
    b = int((ppm - C.PPM_MIN) / C.PPM_BIN_WIDTH)
    return max(0, min(b, C.NUM_PPM_BINS - 1))


def j_to_bin(j: Optional[float]) -> int:
    """Discretise a coupling constant (Hz) into a bin index.  0 = no coupling."""
    if j is None:
        return 0
    if isinstance(j, list):
        j = j[0] if j else 0.0
    return max(0, min(int(j / C.J_BIN_WIDTH) + 1, C.NUM_J_BINS - 1))


def mult_to_idx(mult: Optional[str]) -> int:
    """Map multiplicity string to index."""
    if mult is None:
        return C.MULT_TO_IDX["none"]
    return C.MULT_TO_IDX.get(mult, C.MULT_TO_IDX["other"])


def encode_peak(
    ppm: float, mult: Optional[str], j: Optional[float]
) -> Tuple[int, int, int]:
    """Encode a single peak as (ppm_bin, mult_idx, j_bin)."""
    return ppm_to_bin(ppm), mult_to_idx(mult), j_to_bin(j)


def encode_spectrum(
    peaks_json: str,
    sort: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """Parse JSON peak list → aggregated list of (ppm_bin, mult_idx, j_bin, intensity)."""
    if not peaks_json:
        return []
    peaks = json.loads(peaks_json)
    encoded_basic: List[Tuple[int, int, int]] = []
    for p in peaks:
        ppm = p.get("ppm")
        if ppm is None:
            continue
        encoded_basic.append(
            encode_peak(
                float(ppm),
                p.get("multiplicity"),
                p.get("coupling_hz"),
            )
        )
    counts = Counter(encoded_basic)
    encoded = [
        (ppm_bin, mult_idx, j_bin, min(int(count), C.MAX_PEAK_INTENSITY))
        for (ppm_bin, mult_idx, j_bin), count in counts.items()
    ]
    if sort:
        encoded.sort(key=lambda x: (x[0], x[1], x[2]))
    return encoded


def smiles_to_formula(smiles: str) -> str:
    """Derive a molecular formula string from SMILES if possible."""
    if not smiles:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return ""


def formula_to_vector(formula: str) -> List[float]:
    """Convert an exact molecular formula string to a fixed atom-count vector."""
    vec = [0.0] * len(C.FORMULA_ATOMS)
    if not formula:
        return vec
    for atom, count_str in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        idx = C.FORMULA_TO_IDX.get(atom)
        if idx is None:
            continue
        vec[idx] += float(int(count_str) if count_str else 1)
    return vec


def encode_formula_vector(row: Dict[str, Any]) -> List[float]:
    """Build an exact formula vector from row metadata or SMILES."""
    formula = (
        row.get("molecular_formula")
        or row.get("formula")
        or row.get("formula_string")
        or ""
    )
    if not formula:
        formula = smiles_to_formula(row.get("smiles", ""))
    return formula_to_vector(formula)


# ──────────────────────────────────────────────────────────────────────────────
#  3. SELFIES tokenisation
# ──────────────────────────────────────────────────────────────────────────────


class SelfiesVocab:
    """
    Build and manage a SELFIES token vocabulary.

    Tokens are the bracket-delimited units, e.g. "[C]", "[=Branch1]", "[Ring1]".
    """

    def __init__(self, tokens: Optional[List[str]] = None):
        if tokens is None:
            tokens = []
        all_tokens = list(C.SPECIAL_TOKENS) + tokens
        # Deduplicate while preserving order
        seen = set()
        unique: List[str] = []
        for t in all_tokens:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        self.tokens = unique
        self.tok2idx: Dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        self.idx2tok: Dict[int, str] = {i: t for i, t in enumerate(self.tokens)}

    @property
    def pad_idx(self) -> int:
        return self.tok2idx[C.SELFIES_PAD]

    @property
    def bos_idx(self) -> int:
        return self.tok2idx[C.SELFIES_BOS]

    @property
    def eos_idx(self) -> int:
        return self.tok2idx[C.SELFIES_EOS]

    @property
    def unk_idx(self) -> int:
        return self.tok2idx[C.SELFIES_UNK]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, selfies_str: str) -> List[int]:
        """Convert a SELFIES string to a list of token IDs (with BOS/EOS)."""
        toks = list(sf.split_selfies(selfies_str))
        ids = [self.bos_idx]
        for t in toks:
            ids.append(self.tok2idx.get(t, self.unk_idx))
        ids.append(self.eos_idx)
        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to a SELFIES string."""
        toks = []
        for i in ids:
            t = self.idx2tok.get(i, C.SELFIES_UNK)
            if t == C.SELFIES_EOS:
                break
            if t in (C.SELFIES_PAD, C.SELFIES_BOS):
                continue
            toks.append(t)
        return "".join(toks)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.tokens, f)

    @classmethod
    def load(cls, path: str) -> "SelfiesVocab":
        with open(path, "rb") as f:
            tokens = pickle.load(f)
        return cls(tokens)

    @classmethod
    def build_from_data(
        cls,
        selfies_list: List[str],
        min_freq: int = 2,  # TODO przeanalizować wpływ na dane i model
    ) -> "SelfiesVocab":
        """Build vocabulary from a list of SELFIES strings."""
        counter: Counter = Counter()
        for s in selfies_list:
            for tok in sf.split_selfies(s):
                counter[tok] += 1
        tokens = [tok for tok, cnt in counter.most_common() if cnt >= min_freq]
        print(f"[Vocab] {len(tokens)} tokens (min_freq={min_freq})")
        return cls(tokens)


# ──────────────────────────────────────────────────────────────────────────────
#  4. Global feature vector
# ──────────────────────────────────────────────────────────────────────────────


def encode_global_features(row: Dict[str, Any]) -> List[float]:
    """
    Build a fixed-length global feature vector from a database row.

    Features (20-dim):
      0   peak_count        (log-scaled)
      1   carbon_count      (log-scaled)
      2   heavy_atom_count  (log-scaled)
      3   daltons           (log-scaled / 1000)
      4   sp3_c / carbon_count  (ratio)
      5   sp2_c / carbon_count  (ratio)
      6   sp1_c / carbon_count  (ratio)
      7   bin_0_50 / peak_count
      8   bin_50_90 / peak_count
      9   bin_90_110 / peak_count
      10  bin_110_165 / peak_count
      11  bin_165_195 / peak_count
      12  bin_195_220 / peak_count
      13  bin_out / peak_count
      14–19  solvent one-hot (first 6 solvents; rest = index 5)
    """
    cc = max(row.get("carbon_count", 1), 1)
    pc = max(row.get("peak_count", 1), 1)

    feats = [
        math.log1p(pc),
        math.log1p(cc),
        math.log1p(row.get("heavy_atom_count", 0)),
        math.log1p(row.get("daltons", 0)) / 8.0,  # normalise
        row.get("sp3_c", 0) / cc,
        row.get("sp2_c", 0) / cc,
        row.get("sp1_c", 0) / cc,
    ]

    # Bin ratios
    for bname in [
        "bin_0_50",
        "bin_50_90",
        "bin_90_110",
        "bin_110_165",
        "bin_165_195",
        "bin_195_220",
        "bin_out",
    ]:
        feats.append(row.get(bname, 0) / pc)

    # Solvent one-hot (6 dims)
    solvent = row.get("solvent", "not_known")
    sol_idx = C.SOLVENT_TO_IDX.get(solvent, 5)  # "other" → 5
    sol_onehot = [0.0] * 6
    sol_onehot[min(sol_idx, 5)] = 1.0
    feats.extend(sol_onehot)

    return feats  # 7 + 7 + 6 = 20


# ──────────────────────────────────────────────────────────────────────────────
#  5. Data augmentation
# ──────────────────────────────────────────────────────────────────────────────


def augment_spectrum(
    encoded_peaks: List[Tuple[int, int, int, int]],
    cfg: C.TrainConfig,
) -> List[Tuple[int, int, int, int]]:
    """Apply stochastic augmentation to an encoded spectrum."""
    augmented = []
    for ppm_bin, mult_idx, j_bin, intensity in encoded_peaks:
        # Drop entire peak
        if random.random() < cfg.peak_drop_prob:
            continue

        # PPM noise (in bin-space)
        noise = random.gauss(0, cfg.ppm_noise_std / C.PPM_BIN_WIDTH)
        ppm_bin = max(0, min(int(ppm_bin + noise), C.NUM_PPM_BINS - 1))

        # Multiplicity dropout → "none"
        if random.random() < cfg.mult_drop_prob:
            mult_idx = C.MULT_TO_IDX["none"]

        # Coupling noise (in bin-space)
        if j_bin > 0:
            j_noise = random.gauss(0, cfg.j_noise_std / C.J_BIN_WIDTH)
            j_bin = max(1, min(int(j_bin + j_noise), C.NUM_J_BINS - 1))

        augmented.append((ppm_bin, mult_idx, j_bin, intensity))

    augmented.sort(key=lambda x: x[0])
    return augmented


# ──────────────────────────────────────────────────────────────────────────────
#  6. PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────


class NMRDataset(Dataset):
    """
    Each item returns:
        spectrum_tensor   – (num_peaks, 4)  int  [ppm_bin, mult_idx, j_bin, intensity]
        global_features   – (20,)           float
        formula_vector    – (num_formula_features,) float
        selfies_ids       – (seq_len,)      long  [bos … eos]
        row_id            – int
    """

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        vocab: SelfiesVocab,
        augment: bool = False,
        train_cfg: Optional[C.TrainConfig] = None,
    ):
        self.rows = rows
        self.vocab = vocab
        self.augment = augment
        self.train_cfg = train_cfg or C.TrainConfig()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]

        # Spectrum
        spec = encode_spectrum(row["peaks"])
        if self.augment:
            spec = augment_spectrum(spec, self.train_cfg)
        # Truncate
        spec = spec[: C.MAX_PEAKS]
        if spec:
            spec_t = torch.tensor(spec, dtype=torch.long)  # (n_peaks, 4)
        else:
            spec_t = torch.empty((0, 4), dtype=torch.long)

        # Global features
        gf = torch.tensor(encode_global_features(row), dtype=torch.float32)
        formula_vec = torch.tensor(encode_formula_vector(row), dtype=torch.float32)

        # SELFIES target
        sf_ids = self.vocab.encode(row["selfies"])
        sf_ids = sf_ids[: C.MAX_SELFIES_LEN]
        sf_t = torch.tensor(sf_ids, dtype=torch.long)

        return spec_t, gf, formula_vec, sf_t, row["id"]


def collate_fn(batch):
    """
    Custom collation: pad spectrum and selfies sequences independently.
    """
    specs, gfs, formula_vecs, sfs, ids = zip(*batch)

    # Pad spectra – (batch, max_peaks, 4)
    spec_lens = [s.size(0) for s in specs]
    max_plen = max(spec_lens)
    spec_padded = torch.zeros(len(specs), max_plen, 4, dtype=torch.long)
    spec_mask = torch.ones(len(specs), max_plen, dtype=torch.bool)  # True = masked
    for i, s in enumerate(specs):
        n = s.size(0)
        if n > 0:
            spec_padded[i, :n] = s
            spec_mask[i, :n] = False

    # Stack global features
    gf_tensor = torch.stack(gfs)  # (batch, 20)
    formula_tensor = torch.stack(formula_vecs)  # (batch, num_formula_features)

    # Pad SELFIES – (batch, max_sf_len)
    sf_padded = pad_sequence(sfs, batch_first=True, padding_value=0)
    sf_mask = sf_padded == 0  # True = masked

    ids_tensor = torch.tensor(ids, dtype=torch.long)

    return (
        spec_padded,
        spec_mask,
        gf_tensor,
        formula_tensor,
        sf_padded,
        sf_mask,
        ids_tensor,
    )


def _compute_split_sizes(
    total_rows: int, val_fraction: float, test_fraction: float
) -> tuple[int, int, int]:
    """Compute train/val/test sizes while keeping splits non-empty when possible."""
    if total_rows <= 0:
        return 0, 0, 0

    n_test = int(total_rows * test_fraction)
    n_val = int(total_rows * val_fraction)

    if total_rows >= 3:
        if test_fraction > 0 and n_test == 0:
            n_test = 1
        if val_fraction > 0 and n_val == 0:
            n_val = 1

        if n_test + n_val >= total_rows:
            overflow = n_test + n_val - (total_rows - 1)
            reduce_test = min(overflow, n_test)
            n_test -= reduce_test
            overflow -= reduce_test
            if overflow > 0:
                n_val = max(0, n_val - overflow)

    n_train = total_rows - n_val - n_test
    return n_train, n_val, n_test


def _save_split_ids(test_rows: List[Dict[str, Any]], output_path: str) -> None:
    """Persist held-out test row IDs to disk as JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "count": len(test_rows),
        "ids": [int(r["id"]) for r in test_rows if r.get("id") is not None],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def make_dataloaders(
    db_path: str = C.DB_PATH,
    table: str = C.TABLE_NAME,
    limit: Optional[int] = None,
    vocab_path: Optional[str] = None,
    train_cfg: Optional[C.TrainConfig] = None,
    test_ids_path: Optional[str] = None,
    restrict_carbons_to_peaks: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, SelfiesVocab]:
    """
    Build train + val + test dataloaders from the database.

    Returns (train_loader, val_loader, test_loader, vocab).
    """
    tcfg = train_cfg or C.TrainConfig()
    print(f"[Data] Loading rows from {db_path} / {table} …")
    rows = load_rows(db_path, table, limit)

    print(f"[Data] {len(rows)} rows loaded.")

    # Filter to only most common solvent
    if rows:
        solvent_counts = Counter(r.get("solvent", "not_known") for r in rows)
        most_common_solvent, _ = solvent_counts.most_common(1)[0]
        rows = [r for r in rows if r.get("solvent", "not_known") == most_common_solvent]
        print(f"[Data] Filtered to solvent: {most_common_solvent} ({len(rows)} rows)")

    if restrict_carbons_to_peaks:
        rows = [
            r
            for r in rows
            if r.get("carbon_count") is not None
            and r.get("peak_count") is not None
            and r.get("carbon_count", 0) <= r.get("peak_count", 0)
        ]
        print(
            "[Data] Filtered to rows with carbon_count <= peak_count "
            f"({len(rows)} rows)"
        )

    # Build or load vocab
    if vocab_path and os.path.exists(vocab_path):
        vocab = SelfiesVocab.load(vocab_path)
        print(f"[Data] Vocab loaded from {vocab_path} ({len(vocab)} tokens)")
    else:
        selfies_strs = [r["selfies"] for r in rows if r.get("selfies")]
        vocab = SelfiesVocab.build_from_data(selfies_strs, min_freq=2)
        if vocab_path:
            os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
            vocab.save(vocab_path)
            print(f"[Data] Vocab saved to {vocab_path}")

    # Split
    n_train, n_val, n_test = _compute_split_sizes(
        len(rows),
        tcfg.val_fraction,
        tcfg.test_fraction,
    )
    rng = random.Random(tcfg.seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    train_rows = [rows[i] for i in indices[:n_train]]
    val_rows = [rows[i] for i in indices[n_train : n_train + n_val]]
    test_rows = [rows[i] for i in indices[n_train + n_val : n_train + n_val + n_test]]

    if test_ids_path:
        _save_split_ids(test_rows, test_ids_path)
        print(f"[Data] Test IDs saved to {test_ids_path}")

    train_ds = NMRDataset(train_rows, vocab, augment=True, train_cfg=tcfg)
    val_ds = NMRDataset(val_rows, vocab, augment=False)
    test_ds = NMRDataset(test_rows, vocab, augment=False)

    train_dl = DataLoader(
        train_ds,
        batch_size=tcfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=tcfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=tcfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    print(f"[Data] Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    return train_dl, val_dl, test_dl, vocab
