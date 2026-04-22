"""
config.py – Central configuration for the NMR Solver.

All tuneable hyper-parameters, paths, and constants live here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════════

DB_PATH = "/home/vqire/Downloads/nmrexp_no_spectra.db"
TABLE_NAME = "nmr_data"

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
INDEX_DIR = os.path.join(PROJECT_ROOT, "indexes")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# ══════════════════════════════════════════════════════════════════════════════
#  SPECTRUM TOKENISATION
# ══════════════════════════════════════════════════════════════════════════════

# PPM discretisation – 0.5 ppm bins from -10 to 240
PPM_MIN = -10.0
PPM_MAX = 240.0
PPM_BIN_WIDTH = 0.5
NUM_PPM_BINS = int((PPM_MAX - PPM_MIN) / PPM_BIN_WIDTH)  # 500

# Multiplicity vocabulary
MULT_VOCAB = [
    "none",
    "s",
    "d",
    "t",
    "q",
    "dd",
    "dt",
    "td",
    "dq",
    "qd",
    "m",
    "dm",
    "quint",
    "quint.",
    "br",
    "other",
]
MULT_TO_IDX = {m: i for i, m in enumerate(MULT_VOCAB)}

# Coupling constant discretisation (Hz) – 1 Hz bins from 0 to 400
J_MAX = 400.0
J_BIN_WIDTH = 1.0
NUM_J_BINS = int(J_MAX / J_BIN_WIDTH) + 1  # 0 = no coupling
MAX_PEAK_INTENSITY = 16

# Solvent vocabulary
SOLVENT_VOCAB = [
    "CDCl3",
    "DMSO-d6",
    "CD3OD",
    "not_known",
    "CD3COCD3",
    "CD2Cl2",
    "C6D6",
    "CD3CN",
    "D2O",
    "mixed",
    "other",
]
SOLVENT_TO_IDX = {s: i for i, s in enumerate(SOLVENT_VOCAB)}

# Exact-formula atom vocabulary used for formula guidance.
FORMULA_ATOMS = [
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "B",
    "Si",
    "Se",
    "Ge",
    "Na",
    "K",
    "Li",
    "Mg",
    "Ca",
    "Al",
    "Zn",
    "Cu",
    "Fe",
]
FORMULA_TO_IDX = {atom: i for i, atom in enumerate(FORMULA_ATOMS)}

# ══════════════════════════════════════════════════════════════════════════════
#  SELFIES TOKENISATION
# ══════════════════════════════════════════════════════════════════════════════

# These are built dynamically from the training data (see data.py)
# but we define special tokens here.
SELFIES_PAD = "<pad>"
SELFIES_BOS = "<bos>"
SELFIES_EOS = "<eos>"
SELFIES_UNK = "<unk>"
SPECIAL_TOKENS = [SELFIES_PAD, SELFIES_BOS, SELFIES_EOS, SELFIES_UNK]

MAX_SELFIES_LEN = 256  # max output tokens (covers 99%+ of data)
MAX_PEAKS = 140  # max peaks per spectrum

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL HYPER-PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelConfig:
    """Shared config for forward and inverse Transformer models."""

    # ── Encoder (spectrum) ────────────────────────────────────────────────────
    # Encoder (spectrum)
    enc_d_model: int = (
        256  # Model dimension for encoder. Typical: 128–1024. Higher = more expressive, slower.
    )
    enc_nhead: int = (
        8  # Number of attention heads in encoder. Min: 1, Max: enc_d_model/8. More heads = better multi-feature learning.
    )
    enc_layers: int = (
        6  # Number of encoder layers. Min: 1, Max: 12+. More layers = deeper model, higher capacity.
    )
    enc_dim_ff: int = (
        1024  # Feedforward network dimension in encoder. Min: enc_d_model, Max: 4×enc_d_model.
    )
    enc_dropout: float = (
        0.1  # Dropout rate in encoder. Min: 0.0, Max: 0.5. Regularizes model.
    )
    enc_num_inds: int = (
        32  # Number of inducing points for Set Transformer. Min: 1, Max: 128+.
    )
    enc_num_seeds: int = 1  # Number of seed vectors for pooling. Min: 1, Max: 8.

    # Decoder (SELFIES)
    dec_d_model: int = 256  # Model dimension for decoder. Same as enc_d_model.
    dec_nhead: int = 8  # Number of attention heads in decoder. Same as enc_nhead.
    dec_layers: int = 6  # Number of decoder layers. Same as enc_layers.
    dec_dim_ff: int = (
        1024  # Feedforward network dimension in decoder. Same as enc_dim_ff.
    )
    dec_dropout: float = 0.1  # Dropout rate in decoder. Same as enc_dropout.

    # Global features injected via cross-attention or concatenation
    # (carbon_count, heavy_atom_count, daltons, sp3_c, sp2_c, sp1_c, bin_0_50 … bin_out, solvent (one-hot) → total ~20)
    num_global_features: int = 20  # Number of global features. Min: 1, Max: 32+.
    num_formula_features: int = len(
        FORMULA_ATOMS
    )  # Number of formula atom types. Set by len(FORMULA_ATOMS).

    # Retrieval (FAISS)
    retrieval_dim: int = (
        128  # Dimensionality of spectrum embedding for retrieval. Min: 32, Max: 512.
    )
    retrieval_top_k: int = 64  # Number of candidates retrieved. Min: 1, Max: 256+.


@dataclass
class TrainConfig:
    """Training hyper-parameters."""

    batch_size: int = 128  # Number of samples per training batch. Min: 1, Max: 1024+.
    lr: float = 3e-4  # Learning rate. Min: 1e-6, Max: 1e-2.
    weight_decay: float = 1e-2  # L2 regularization strength. Min: 0.0, Max: 1.0.
    warmup_steps: int = 4000  # Steps to linearly increase LR. Min: 0, Max: 10,000+.
    max_epochs: int = 10  # Maximum training epochs. Min: 1, Max: 1000+.
    grad_clip: float = 1.0  # Maximum gradient norm. Min: 0.0, Max: 10.0+.
    label_smoothing: float = 0.1  # Smoothing for target labels. Min: 0.0, Max: 0.2.
    val_fraction: float = 0.05  # Fraction of data for validation. Min: 0.0, Max: 0.5.
    test_fraction: float = 0.02  # Fraction of data for test set. Min: 0.0, Max: 0.5.
    seed: int = 42  # Random seed for reproducibility. Any integer.

    # Tanimoto RL loss (REINFORCE)
    tanimoto_interval: int = 5  # Steps between RL loss computation. Min: 1, Max: 100+.
    tanimoto_rl_samples: int = 4  # RL decode samples per batch. Min: 1, Max: 16+.
    rl_tanimoto_weight: float = (
        0.75  # Weight for Tanimoto in RL reward. Min: 0.0, Max: 1.0.
    )
    rl_token_weight: float = (
        0.25  # Weight for token accuracy in RL reward. Min: 0.0, Max: 1.0.
    )
    rl_scale_boost: float = 1.25  # Multiplier for RL reward. Min: 1.0, Max: 10.0.

    # Best inverse checkpoint selection
    best_ckpt_tanimoto_weight: float = (
        0.55  # Weight for Tanimoto in best checkpoint selection. Min: 0.0, Max: 1.0.
    )
    best_ckpt_token_acc_weight: float = (
        0.30  # Weight for token accuracy in best checkpoint selection. Min: 0.0, Max: 1.0.
    )
    best_ckpt_validity_weight: float = (
        0.15  # Weight for validity in best checkpoint selection. Min: 0.0, Max: 1.0.
    )
    best_ckpt_metric_tolerance: float = (
        0.002  # Minimum improvement required to save new best checkpoint. Min: 0.0, Max: 0.1.
    )

    # Beam search
    beam_size: int = 10  # Beam width for beam search decoding. Min: 1, Max: 100+.
    top_k_output: int = 10  # Number of top outputs to keep. Min: 1, Max: beam_size.

    # Data augmentation
    ppm_noise_std: float = (
        0.3  # Stddev of Gaussian noise on ppm values. Min: 0.0, Max: 5.0.
    )
    mult_drop_prob: float = (
        0.15  # Probability of dropping multiplicity info. Min: 0.0, Max: 1.0.
    )
    j_noise_std: float = (
        1.0  # Stddev of Gaussian noise on coupling constants. Min: 0.0, Max: 10.0.
    )
    peak_drop_prob: float = (
        0.05  # Probability of dropping a peak entirely. Min: 0.0, Max: 1.0.
    )


def suggest_model_config(trial: Any, base: ModelConfig | None = None) -> ModelConfig:
    """Build a `ModelConfig` from an Optuna trial."""
    base = base or ModelConfig()

    enc_d_model = int(
        trial.suggest_categorical(
            "enc_d_model",
            # [128, 192, 256, 320, 384, 512],
            [128, 192, 256],
        )
    )

    # Keep the search space static across trials to avoid Optuna's dynamic
    # categorical distribution error.
    # All choices below divide every enc_d_model option above.
    enc_nhead = int(trial.suggest_categorical("enc_nhead", [1, 2, 4, 8]))
    enc_layers = trial.suggest_int("enc_layers", 2, 8)
    enc_ff_mult = int(trial.suggest_categorical("enc_ff_mult", [1, 2, 4]))
    enc_dim_ff = enc_d_model * enc_ff_mult
    enc_dropout = trial.suggest_float("enc_dropout", 0.0, 0.5)
    # dec_d_model = int(
    #     trial.suggest_categorical(
    #         "dec_d_model",
    #         [128, 192, 256, 320, 384, 512],
    #     )
    # )

    return ModelConfig(
        enc_d_model=enc_d_model,
        enc_nhead=enc_nhead,
        enc_layers=enc_layers,
        enc_dim_ff=enc_dim_ff,
        enc_dropout=enc_dropout,
        enc_num_inds=trial.suggest_int("enc_num_inds", 32, 128, step=8),
        enc_num_seeds=trial.suggest_int("enc_num_seeds", 1, 8),
        dec_d_model=enc_d_model,
        dec_nhead=enc_nhead,
        dec_layers=enc_layers,
        dec_dim_ff=enc_dim_ff,
        dec_dropout=enc_dropout,
        num_global_features=int(
            trial.suggest_categorical(
                "num_global_features",
                [base.num_global_features],
            )
        ),
        num_formula_features=int(
            trial.suggest_categorical(
                "num_formula_features",
                [base.num_formula_features],
            )
        ),
        retrieval_dim=int(
            trial.suggest_categorical(
                "retrieval_dim",
                [32, 64, 96, 128, 192, 256, 384, 512],
            )
        ),
        retrieval_top_k=64,
    )


def suggest_train_config(trial: Any, base: TrainConfig | None = None) -> TrainConfig:
    """Build a `TrainConfig` from an Optuna trial.

    `batch_size`, `max_epochs`, and `top_k_output` are intentionally inherited
    from the caller and excluded from the search space.
    """
    base = base or TrainConfig()
    # Enforce that the three best_ckpt_* weights always sum to 1
    # Sample two, compute the third
    best_ckpt_tanimoto_weight = trial.suggest_float(
        "best_ckpt_tanimoto_weight", 0.4, 1.0
    )
    best_ckpt_token_acc_weight = trial.suggest_float(
        "best_ckpt_token_acc_weight", 0.0, 1.0
    )
    # Clamp so the sum does not exceed 1
    max_token_acc = min(best_ckpt_token_acc_weight, 1.0 - best_ckpt_tanimoto_weight)
    best_ckpt_token_acc_weight = max(0.0, max_token_acc)
    best_ckpt_validity_weight = (
        1.0 - best_ckpt_tanimoto_weight - best_ckpt_token_acc_weight
    )
    # Clamp validity weight to [0, 1]
    if best_ckpt_validity_weight < 0.0:
        best_ckpt_validity_weight = 0.0
    elif best_ckpt_validity_weight > 1.0:
        best_ckpt_validity_weight = 1.0

    return TrainConfig(
        batch_size=base.batch_size,
        lr=trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        warmup_steps=trial.suggest_int("warmup_steps", 500, 8000, step=250),
        max_epochs=base.max_epochs,
        grad_clip=trial.suggest_float("grad_clip", 0.5, 5.0, log=True),
        label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.1),
        val_fraction=trial.suggest_float("val_fraction", 0.01, 0.20),
        test_fraction=trial.suggest_float("test_fraction", 0.005, 0.10),
        seed=trial.suggest_int("seed", 1, 100000),
        tanimoto_interval=trial.suggest_int("tanimoto_interval", 5, 25),
        tanimoto_rl_samples=trial.suggest_int("tanimoto_rl_samples", 1, 8),
        rl_tanimoto_weight=trial.suggest_float("rl_tanimoto_weight", 0.2, 0.8),
        rl_token_weight=trial.suggest_float("rl_token_weight", 0.2, 0.8),
        rl_scale_boost=trial.suggest_float("rl_scale_boost", 0.5, 3.0),
        best_ckpt_tanimoto_weight=best_ckpt_tanimoto_weight,
        best_ckpt_token_acc_weight=best_ckpt_token_acc_weight,
        best_ckpt_validity_weight=best_ckpt_validity_weight,
        best_ckpt_metric_tolerance=trial.suggest_float(
            "best_ckpt_metric_tolerance", 1e-5, 1e-1, log=True
        ),
        beam_size=trial.suggest_int("beam_size", 1, 32),
        top_k_output=base.top_k_output,
        ppm_noise_std=trial.suggest_float("ppm_noise_std", 0.0, 1.0),
        mult_drop_prob=trial.suggest_float("mult_drop_prob", 0.0, 0.2),
        j_noise_std=trial.suggest_float("j_noise_std", 0.0, 2.0),
        peak_drop_prob=trial.suggest_float("peak_drop_prob", 0.0, 0.08),
    )
