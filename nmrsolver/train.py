"""
train.py – Training loops for the NMR Solver models.

Usage:
    python -m nmrsolver.train inverse          # train inverse model
    python -m nmrsolver.train forward          # train forward model
    python -m nmrsolver.train both             # train both sequentially
    python -m nmrsolver.train inverse --limit 50000  # quick dev run

Checkpoints, vocab, and logs are saved to  checkpoints/ .
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import DataStructs, RDKFingerprint
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from . import config as C
from .data import (
    NMRDataset,
    SelfiesVocab,
    collate_fn,
    encode_formula_vector,
    encode_spectrum,
    load_rows,
    make_dataloaders,
)
from .models import ForwardModel, InverseModel

# ──────────────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_IDS: list[int] = []
USE_MULTI_GPU = False
optuna: Any | None = None


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when wrapped for multi-GPU training."""
    return model.module if isinstance(model, nn.DataParallel) else model


def _configure_runtime(args: argparse.Namespace) -> None:
    """Configure the runtime device and optional multi-GPU execution."""
    global DEVICE, GPU_IDS, USE_MULTI_GPU

    if not torch.cuda.is_available():
        DEVICE = torch.device("cpu")
        GPU_IDS = []
        USE_MULTI_GPU = False
        return

    available_gpu_count = torch.cuda.device_count()
    requested_gpu_ids = args.gpu_ids
    if requested_gpu_ids:
        gpu_ids = [int(x.strip()) for x in requested_gpu_ids.split(",") if x.strip()]
    else:
        gpu_ids = list(range(available_gpu_count))

    invalid_gpu_ids = [
        gpu_id for gpu_id in gpu_ids if gpu_id >= available_gpu_count or gpu_id < 0
    ]
    if invalid_gpu_ids:
        raise ValueError(
            f"Invalid GPU ids {invalid_gpu_ids}; available GPU ids are 0..{available_gpu_count - 1}"
        )

    GPU_IDS = gpu_ids or [0]
    DEVICE = torch.device(f"cuda:{GPU_IDS[0]}")
    torch.cuda.set_device(GPU_IDS[0])
    USE_MULTI_GPU = bool(args.multi_gpu and len(GPU_IDS) > 1)


def _prepare_model_for_training(model: nn.Module) -> nn.Module:
    """Move a model to the active device and wrap it for multi-GPU if requested."""
    model = model.to(DEVICE)
    if USE_MULTI_GPU:
        model = nn.DataParallel(model, device_ids=GPU_IDS, output_device=GPU_IDS[0])
    return model


def get_lr_lambda(warmup: int):
    """Linear warmup then inverse-sqrt decay."""

    def lr_lambda(step: int) -> float:
        step = max(step, 1)
        if step < warmup:
            return step / warmup
        return (warmup / step) ** 0.5

    return lr_lambda


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tanimoto_similarity(smi_a: str, smi_b: str) -> float:
    """Tanimoto similarity between two SMILES strings (RDK fingerprint)."""
    try:
        mol_a = Chem.MolFromSmiles(smi_a)
        mol_b = Chem.MolFromSmiles(smi_b)
        if mol_a is None or mol_b is None:
            return 0.0
        fp_a = RDKFingerprint(mol_a)
        fp_b = RDKFingerprint(mol_b)
        return DataStructs.TanimotoSimilarity(fp_a, fp_b)
    except Exception:
        return 0.0


def is_valid_smiles(smiles: str) -> bool:
    """Return whether SMILES parses successfully."""
    if not smiles:
        return False
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def decode_selfies_for_reward(selfies_str: str) -> Optional[str]:
    """
    Decode SELFIES to SMILES for reward/eval purposes.

    Returns None when the SELFIES contains `<unk>` or cannot be decoded.
    This allows skipping base-data examples that are not faithfully representable
    by the current vocabulary.
    """
    if not selfies_str or C.SELFIES_UNK in selfies_str:
        return None
    try:
        decoded = sf.decoder(selfies_str)
    except Exception:
        return None
    return decoded if isinstance(decoded, str) else None


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    path: str,
    extra_state: Optional[dict] = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
) -> dict:
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    _unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def _append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """Append a JSON record to a JSONL file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _train_cfg_to_dict(cfg: C.TrainConfig) -> dict[str, Any]:
    return {
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "warmup_steps": cfg.warmup_steps,
        "max_epochs": cfg.max_epochs,
        "grad_clip": cfg.grad_clip,
        "label_smoothing": cfg.label_smoothing,
        "val_fraction": cfg.val_fraction,
        "test_fraction": cfg.test_fraction,
        "seed": cfg.seed,
        "tanimoto_interval": cfg.tanimoto_interval,
        "tanimoto_rl_samples": cfg.tanimoto_rl_samples,
        "rl_tanimoto_weight": cfg.rl_tanimoto_weight,
        "rl_token_weight": cfg.rl_token_weight,
        "rl_scale_boost": cfg.rl_scale_boost,
        "best_ckpt_tanimoto_weight": cfg.best_ckpt_tanimoto_weight,
        "best_ckpt_token_acc_weight": cfg.best_ckpt_token_acc_weight,
        "best_ckpt_validity_weight": cfg.best_ckpt_validity_weight,
        "best_ckpt_metric_tolerance": cfg.best_ckpt_metric_tolerance,
        "beam_size": cfg.beam_size,
        "top_k_output": cfg.top_k_output,
        "ppm_noise_std": cfg.ppm_noise_std,
        "mult_drop_prob": cfg.mult_drop_prob,
        "j_noise_std": cfg.j_noise_std,
        "peak_drop_prob": cfg.peak_drop_prob,
    }


def _model_cfg_to_dict(cfg: C.ModelConfig) -> dict[str, Any]:
    return {
        "enc_d_model": cfg.enc_d_model,
        "enc_nhead": cfg.enc_nhead,
        "enc_layers": cfg.enc_layers,
        "enc_dim_ff": cfg.enc_dim_ff,
        "enc_dropout": cfg.enc_dropout,
        "enc_num_inds": cfg.enc_num_inds,
        "enc_num_seeds": cfg.enc_num_seeds,
        "dec_d_model": cfg.dec_d_model,
        "dec_nhead": cfg.dec_nhead,
        "dec_layers": cfg.dec_layers,
        "dec_dim_ff": cfg.dec_dim_ff,
        "dec_dropout": cfg.dec_dropout,
        "num_global_features": cfg.num_global_features,
        "num_formula_features": cfg.num_formula_features,
        "retrieval_dim": cfg.retrieval_dim,
        "retrieval_top_k": cfg.retrieval_top_k,
    }


def _make_execution_dir(mode: str, study_name: Optional[str]) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    study_slug = (study_name or "study").replace(" ", "_")
    execution_dir = Path(C.LOG_DIR) / f"{timestamp}_{mode}_{study_slug}"
    execution_dir.mkdir(parents=True, exist_ok=True)
    return execution_dir


def _copy_if_exists(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    if not src_path.exists():
        return
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def _copytree_if_exists(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    if not src_path.exists():
        return
    dst_path = Path(dst)
    if dst_path.exists():
        shutil.rmtree(dst_path)
    shutil.copytree(src_path, dst_path)


def inverse_quality_score(
    token_acc: float,
    tanimoto: float,
    validity: float,
    tcfg: C.TrainConfig,
) -> float:
    """Weighted composite score for inverse-model checkpoint selection."""
    return (
        tcfg.best_ckpt_token_acc_weight * token_acc
        + tcfg.best_ckpt_tanimoto_weight * tanimoto
        + tcfg.best_ckpt_validity_weight * validity
    )


def should_save_best_inverse_checkpoint(
    current_token_acc: float,
    current_tanimoto: float,
    current_validity: float,
    best_token_acc: float,
    best_tanimoto: float,
    best_validity: float,
    best_score: float,
    tcfg: C.TrainConfig,
) -> tuple[bool, float]:
    """
    Decide whether current inverse metrics are collectively better.

    Criteria:
    - weighted quality score must improve
    - tanimoto must not regress past tolerance
    - token accuracy must not regress past tolerance
    - at least one of the three metrics must improve materially
    """
    tol = tcfg.best_ckpt_metric_tolerance
    score = inverse_quality_score(
        current_token_acc,
        current_tanimoto,
        current_validity,
        tcfg,
    )

    improves = [
        current_token_acc > best_token_acc + tol,
        current_tanimoto > best_tanimoto + tol,
        current_validity > best_validity + tol,
    ]
    no_material_regression = (
        current_tanimoto >= best_tanimoto - tol
        and current_token_acc >= best_token_acc - tol
        and current_validity >= best_validity - 2 * tol
    )
    should_save = score > best_score + tol and no_material_regression and any(improves)
    return should_save, score


def generated_token_match_ratio(
    pred_ids: torch.Tensor,
    true_ids: torch.Tensor,
    pad_idx: int,
) -> float:
    """Simple token overlap ratio between generated and target token ids."""
    pred_list = pred_ids.tolist()
    true_list = true_ids.tolist()

    def _truncate(ids: list[int]) -> list[int]:
        out = []
        for idx in ids:
            if idx == pad_idx:
                break
            out.append(idx)
        return out

    pred_list = _truncate(pred_list)
    true_list = _truncate(true_list)
    if not true_list:
        return 0.0
    min_len = min(len(pred_list), len(true_list))
    if min_len == 0:
        return 0.0
    matches = sum(1 for i in range(min_len) if pred_list[i] == true_list[i])
    length_penalty = min_len / max(len(true_list), 1)
    return float(matches / min_len) * float(length_penalty)


# ──────────────────────────────────────────────────────────────────────────────
#  Inverse model training
# ──────────────────────────────────────────────────────────────────────────────


def train_inverse(
    train_dl: DataLoader,
    val_dl: DataLoader,
    vocab: SelfiesVocab,
    tcfg: C.TrainConfig = C.TrainConfig(),
    mcfg: C.ModelConfig = C.ModelConfig(),
    resume_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    shared_log_path: Optional[str] = None,
    return_summary: bool = False,
):
    """Train the inverse model (spectrum → SELFIES)."""
    print(f"\n{'='*72}")
    print("  INVERSE MODEL TRAINING  (Spectrum → SELFIES)")
    print(f"{'='*72}")

    model = _prepare_model_for_training(InverseModel(vocab_size=len(vocab), cfg=mcfg))
    print(f"[Model] Parameters: {count_params(_unwrap_model(model)):,}")
    print(
        f"[Model] Device: {DEVICE}"
        + (
            f" | DataParallel GPUs={GPU_IDS}"
            if USE_MULTI_GPU
            else f" | GPUs={[GPU_IDS[0]]}" if GPU_IDS else ""
        )
    )

    optimizer = AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = LambdaLR(optimizer, get_lr_lambda(tcfg.warmup_steps))
    scaler = GradScaler(
        device="cuda" if DEVICE.type == "cuda" else "cpu", enabled=DEVICE.type == "cuda"
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx, label_smoothing=tcfg.label_smoothing
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_token_acc = 0.0
    best_tanimoto = 0.0
    best_validity = 0.0
    best_inverse_score = float("-inf")
    best_checkpoint_path: Optional[str] = None

    if resume_path and os.path.exists(resume_path):
        ckpt = load_checkpoint(model, optimizer, resume_path)
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        best_val_loss = ckpt.get("loss", float("inf"))
        best_token_acc = float(ckpt.get("best_token_acc", 0.0))
        best_tanimoto = float(ckpt.get("best_tanimoto", 0.0))
        best_validity = float(ckpt.get("best_validity", 0.0))
        best_inverse_score = float(
            ckpt.get(
                "best_inverse_score",
                inverse_quality_score(
                    best_token_acc,
                    best_tanimoto,
                    best_validity,
                    tcfg,
                ),
            )
        )
        print(f"[Resume] epoch={start_epoch}, step={global_step}")

    ckpt_dir = checkpoint_dir or os.path.join(C.CHECKPOINT_DIR, "inverse")
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path = os.path.join(ckpt_dir, "training_log.jsonl")
    log_file = open(log_path, "a")

    # Exponential moving averages for auto-scaling CE and RL losses
    ema_ce = 1.0
    ema_rl = 1.0
    ema_alpha = 0.01  # smoothing factor
    rl_start_step = max(500, tcfg.warmup_steps)
    rl_ce_threshold = 10.0

    for epoch in range(start_epoch, tcfg.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rl_loss = 0.0
        n_batches = 0
        n_rl_batches = 0
        t0 = time.time()

        for batch_idx, (
            spec,
            spec_mask,
            gf,
            formula_vec,
            sf_ids,
            sf_mask,
            ids,
        ) in enumerate(train_dl):
            spec = spec.to(DEVICE)
            spec_mask = spec_mask.to(DEVICE)
            gf = gf.to(DEVICE)
            formula_vec = formula_vec.to(DEVICE)
            sf_ids = sf_ids.to(DEVICE)

            # Teacher forcing: input = sf_ids[:, :-1], target = sf_ids[:, 1:]
            tgt_in = sf_ids[:, :-1]
            tgt_out = sf_ids[:, 1:]

            optimizer.zero_grad()
            with autocast(
                device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                enabled=(DEVICE.type == "cuda"),
            ):
                logits = model(spec, spec_mask, gf, formula_vec, tgt_in)
                ce_loss = criterion(
                    logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)
                )

            if not torch.isfinite(ce_loss):
                print(
                    f"  [Warn] Non-finite CE loss at step {global_step}; skipping batch"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            # ── Tanimoto REINFORCE loss (every N steps) ───────────────
            rl_loss_val = 0.0
            use_rl = (
                global_step >= rl_start_step
                and global_step % tcfg.tanimoto_interval == 0
                and ce_loss.item() < rl_ce_threshold
            )
            if use_rl:
                n_rl = min(tcfg.tanimoto_rl_samples, spec.size(0))
                try:
                    with torch.no_grad():
                        generated = _unwrap_model(model).generate_greedy(
                            spec[:n_rl],
                            spec_mask[:n_rl],
                            gf[:n_rl],
                            formula_vec[:n_rl],
                            bos_idx=vocab.bos_idx,
                            eos_idx=vocab.eos_idx,
                        )
                except Exception as exc:
                    print(
                        f"  [Warn] RL generation skipped at step {global_step}: {exc}"
                    )
                    loss = ce_loss
                    use_rl = False

            if use_rl:
                # Compute Tanimoto rewards
                rewards = []
                valid_rl_indices = []
                for i in range(n_rl):
                    pred_selfies = vocab.decode(generated[i].tolist())
                    true_selfies = vocab.decode(sf_ids[i].tolist())
                    true_smi = decode_selfies_for_reward(true_selfies)
                    if true_smi is None:
                        continue
                    pred_smi = decode_selfies_for_reward(pred_selfies) or ""
                    valid_rl_indices.append(i)
                    tanimoto_reward = tanimoto_similarity(pred_smi, true_smi)
                    token_reward = generated_token_match_ratio(
                        generated[i],
                        sf_ids[i],
                        vocab.pad_idx,
                    )
                    rewards.append(
                        tcfg.rl_tanimoto_weight * tanimoto_reward
                        + tcfg.rl_token_weight * token_reward
                    )

                # Build RL loss: log-prob of greedy tokens weighted by (1 - reward)
                # Use logits from teacher forcing as proxy (same forward pass)
                if valid_rl_indices:
                    valid_idx_t = torch.tensor(valid_rl_indices, device=DEVICE)
                    with autocast(
                        device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                        enabled=(DEVICE.type == "cuda"),
                    ):
                        log_probs = F.log_softmax(logits[valid_idx_t], dim=-1)
                        # Clamp generated to valid range for gathering
                        gen_ids = generated[valid_idx_t, : logits.size(1)].clamp(
                            0, logits.size(-1) - 1
                        )
                        # Pad or truncate to match logits time dim
                        if gen_ids.size(1) < logits.size(1):
                            pad = torch.zeros(
                                gen_ids.size(0),
                                logits.size(1) - gen_ids.size(1),
                                dtype=torch.long,
                                device=DEVICE,
                            )
                            gen_ids = torch.cat([gen_ids, pad], dim=1)
                        token_lp = log_probs.gather(2, gen_ids.unsqueeze(-1)).squeeze(
                            -1
                        )
                        mask = (tgt_out[valid_idx_t] != vocab.pad_idx).float()
                        seq_lp = (token_lp * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                            min=1
                        )

                        rewards_t = torch.tensor(
                            rewards, device=DEVICE, dtype=seq_lp.dtype
                        )
                        rl_loss = -((rewards_t - rewards_t.mean()) * seq_lp).mean()

                        # Auto-scale: make RL loss same magnitude as CE loss
                        scale = tcfg.rl_scale_boost * ema_ce / max(ema_rl, 1e-8)
                        loss = ce_loss + scale * rl_loss

                    rl_loss_val = rl_loss.item()
                    ema_rl = (1 - ema_alpha) * ema_rl + ema_alpha * abs(rl_loss_val)
                    epoch_rl_loss += rl_loss_val
                    n_rl_batches += 1
                else:
                    loss = ce_loss
            else:
                loss = ce_loss

            ema_ce = (1 - ema_alpha) * ema_ce + ema_alpha * ce_loss.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            # Only step the LR scheduler if the optimizer has performed a step
            # (optimizer.state becomes non-empty after the first successful step).
            if optimizer.state:
                scheduler.step()

            epoch_loss += ce_loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 200 == 0:
                lr = scheduler.get_last_lr()[0]
                avg_ce = epoch_loss / n_batches
                avg_rl = epoch_rl_loss / max(n_rl_batches, 1)
                print(
                    f"  [E{epoch:02d} | step {global_step:6d}] "
                    f"ce={avg_ce:.4f}  rl={avg_rl:.4f}  lr={lr:.2e}"
                )

        # ── Epoch summary ─────────────────────────────────────────────────
        epoch_time = time.time() - t0
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ────────────────────────────────────────────────────
        val_loss, val_acc, val_tani, val_validity = evaluate_inverse(
            model, val_dl, vocab, criterion
        )

        avg_rl_epoch = epoch_rl_loss / max(n_rl_batches, 1)
        current_inverse_score = inverse_quality_score(
            val_acc,
            val_tani,
            val_validity,
            tcfg,
        )
        record = {
            "record_type": "epoch_metrics",
            "model": "inverse",
            "epoch": epoch,
            "step": global_step,
            "train_ce_loss": round(avg_train_loss, 5),
            "train_rl_loss": round(avg_rl_epoch, 5),
            "val_loss": round(val_loss, 5),
            "val_token_acc": round(val_acc, 5),
            "val_tanimoto": round(val_tani, 5),
            "val_validity": round(val_validity, 5),
            "val_inverse_score": round(current_inverse_score, 5),
            "lr": scheduler.get_last_lr()[0],
            "time_s": round(epoch_time, 1),
        }
        print(
            f"\n  Epoch {epoch:02d} done in {epoch_time:.0f}s  │ "
            f"ce_loss={avg_train_loss:.4f}  rl_loss={avg_rl_epoch:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"token_acc={val_acc:.3f}  tanimoto={val_tani:.3f}  validity={val_validity:.3f}  "
            f"score={current_inverse_score:.3f}"
        )
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        if shared_log_path:
            _append_jsonl(shared_log_path, record)

        # ── Checkpoint ────────────────────────────────────────────────────
        save_best, candidate_score = should_save_best_inverse_checkpoint(
            val_acc,
            val_tani,
            val_validity,
            best_token_acc,
            best_tanimoto,
            best_validity,
            best_inverse_score,
            tcfg,
        )
        if save_best:
            best_val_loss = val_loss
            best_token_acc = val_acc
            best_tanimoto = val_tani
            best_validity = val_validity
            best_inverse_score = candidate_score
            save_checkpoint(
                model,
                optimizer,
                epoch,
                global_step,
                candidate_score,
                os.path.join(ckpt_dir, "best.pt"),
                extra_state={
                    "best_val_loss": val_loss,
                    "best_token_acc": best_token_acc,
                    "best_tanimoto": best_tanimoto,
                    "best_validity": best_validity,
                    "best_inverse_score": best_inverse_score,
                },
            )
            best_checkpoint_path = os.path.join(ckpt_dir, "best.pt")
            print(
                "  ✓ Saved best checkpoint "
                f"(score={best_inverse_score:.4f}, token_acc={best_token_acc:.4f}, "
                f"tanimoto={best_tanimoto:.4f}, validity={best_validity:.4f})"
            )

        # Save periodic checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                global_step,
                val_loss,
                os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"),
                extra_state={
                    "best_val_loss": best_val_loss,
                    "best_token_acc": best_token_acc,
                    "best_tanimoto": best_tanimoto,
                    "best_validity": best_validity,
                    "best_inverse_score": best_inverse_score,
                },
            )

    log_file.close()
    summary = {
        "best_val_loss": best_val_loss,
        "best_token_acc": best_token_acc,
        "best_tanimoto": best_tanimoto,
        "best_validity": best_validity,
        "best_inverse_score": best_inverse_score,
        "best_checkpoint_path": best_checkpoint_path,
        "checkpoint_dir": ckpt_dir,
    }
    print(
        f"\n[Done] Best inverse score: {best_inverse_score:.4f} "
        f"(token_acc={best_token_acc:.4f}, tanimoto={best_tanimoto:.4f}, validity={best_validity:.4f})"
    )
    if return_summary:
        return summary
    return model


@torch.no_grad()
def evaluate_inverse(
    model: nn.Module,
    val_dl: DataLoader,
    vocab: SelfiesVocab,
    criterion: nn.Module,
    max_eval_batches: int = 100,
) -> tuple[float, float, float]:
    """
    Evaluate inverse model.

    Returns (val_loss, token_accuracy, mean_tanimoto, validity_ratio).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    tanimoto_scores = []
    validity_scores = []
    n_batches = 0

    for batch_idx, (
        spec,
        spec_mask,
        gf,
        formula_vec,
        sf_ids,
        sf_mask,
        ids,
    ) in enumerate(val_dl):
        if batch_idx >= max_eval_batches:
            break

        spec = spec.to(DEVICE)
        spec_mask = spec_mask.to(DEVICE)
        gf = gf.to(DEVICE)
        formula_vec = formula_vec.to(DEVICE)
        sf_ids = sf_ids.to(DEVICE)

        tgt_in = sf_ids[:, :-1]
        tgt_out = sf_ids[:, 1:]

        with autocast(
            device_type="cuda" if DEVICE.type == "cuda" else "cpu",
            enabled=(DEVICE.type == "cuda"),
        ):
            logits = model(spec, spec_mask, gf, formula_vec, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        total_loss += loss.item()

        # Token-level accuracy (ignoring padding)
        preds = logits.argmax(dim=-1)
        mask = tgt_out != vocab.pad_idx
        total_correct += (preds == tgt_out).masked_select(mask).sum().item()
        total_tokens += mask.sum().item()

        # Tanimoto on a few samples per batch (greedy decode is slow)
        if batch_idx < 5:
            generated = _unwrap_model(model).generate_greedy(
                spec[:4],
                spec_mask[:4],
                gf[:4],
                formula_vec[:4],
                bos_idx=vocab.bos_idx,
                eos_idx=vocab.eos_idx,
            )
            for i in range(min(4, generated.size(0))):
                pred_selfies = vocab.decode(generated[i].tolist())
                true_selfies = vocab.decode(sf_ids[i].tolist())
                true_smi = decode_selfies_for_reward(true_selfies)
                if true_smi is None:
                    continue
                pred_smi = decode_selfies_for_reward(pred_selfies) or ""
                validity_scores.append(1.0 if is_valid_smiles(pred_smi) else 0.0)
                tanimoto_scores.append(tanimoto_similarity(pred_smi, true_smi))

        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    token_acc = total_correct / max(total_tokens, 1)
    avg_tani = float(np.mean(tanimoto_scores)) if tanimoto_scores else 0.0
    avg_validity = float(np.mean(validity_scores)) if validity_scores else 0.0

    model.train()
    return avg_loss, token_acc, avg_tani, avg_validity


# ──────────────────────────────────────────────────────────────────────────────
#  Forward model training
# ──────────────────────────────────────────────────────────────────────────────


def train_forward(
    train_dl: DataLoader,
    val_dl: DataLoader,
    vocab: SelfiesVocab,
    tcfg: C.TrainConfig = C.TrainConfig(),
    mcfg: C.ModelConfig = C.ModelConfig(),
    resume_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    shared_log_path: Optional[str] = None,
    return_summary: bool = False,
):
    """Train the forward model (SELFIES → spectrum)."""
    print(f"\n{'='*72}")
    print("  FORWARD MODEL TRAINING  (SELFIES → Spectrum)")
    print(f"{'='*72}")

    model = _prepare_model_for_training(ForwardModel(vocab_size=len(vocab), cfg=mcfg))
    print(f"[Model] Parameters: {count_params(_unwrap_model(model)):,}")
    print(
        f"[Model] Device: {DEVICE}"
        + (
            f" | DataParallel GPUs={GPU_IDS}"
            if USE_MULTI_GPU
            else f" | GPUs={[GPU_IDS[0]]}" if GPU_IDS else ""
        )
    )

    optimizer = AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    scheduler = LambdaLR(optimizer, get_lr_lambda(tcfg.warmup_steps))
    scaler = GradScaler(enabled=DEVICE.type == "cuda")

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    best_checkpoint_path: Optional[str] = None

    if resume_path and os.path.exists(resume_path):
        ckpt = load_checkpoint(model, optimizer, resume_path)
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        best_val_loss = ckpt.get("loss", float("inf"))
        print(f"[Resume] epoch={start_epoch}, step={global_step}")

    ckpt_dir = checkpoint_dir or os.path.join(C.CHECKPOINT_DIR, "forward")
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path = os.path.join(ckpt_dir, "training_log.jsonl")
    log_file = open(log_path, "a")

    for epoch in range(start_epoch, tcfg.max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (
            spec,
            spec_mask,
            gf,
            formula_vec,
            sf_ids,
            sf_mask,
            ids,
        ) in enumerate(train_dl):
            sf_ids = sf_ids.to(DEVICE)
            spec = spec.to(DEVICE)
            spec_mask = spec_mask.to(DEVICE)
            gf = gf.to(DEVICE)

            # Build targets from the spectrum data
            targets = _build_forward_targets(spec, spec_mask, gf)

            optimizer.zero_grad()
            with autocast(
                device_type="cuda" if DEVICE.type == "cuda" else "cpu",
                enabled=(DEVICE.type == "cuda"),
            ):
                preds = model(sf_ids)
                loss = _forward_loss(preds, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            # Only step the LR scheduler if the optimizer has performed a step
            # (optimizer.state becomes non-empty after the first successful step).
            if optimizer.state:
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 200 == 0:
                lr = scheduler.get_last_lr()[0]
                avg = epoch_loss / n_batches
                print(
                    f"  [E{epoch:02d} | step {global_step:6d}] "
                    f"loss={avg:.4f}  lr={lr:.2e}"
                )

        epoch_time = time.time() - t0
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        val_loss = evaluate_forward(model, val_dl, vocab)

        record = {
            "record_type": "epoch_metrics",
            "model": "forward",
            "epoch": epoch,
            "step": global_step,
            "train_loss": round(avg_train_loss, 5),
            "val_loss": round(val_loss, 5),
            "lr": scheduler.get_last_lr()[0],
            "time_s": round(epoch_time, 1),
        }
        print(
            f"\n  Epoch {epoch:02d} done in {epoch_time:.0f}s  │ "
            f"train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}"
        )
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        if shared_log_path:
            _append_jsonl(shared_log_path, record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                epoch,
                global_step,
                val_loss,
                os.path.join(ckpt_dir, "best.pt"),
            )
            best_checkpoint_path = os.path.join(ckpt_dir, "best.pt")
            print("  ✓ Saved best checkpoint")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                global_step,
                val_loss,
                os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"),
            )

    log_file.close()
    summary = {
        "best_val_loss": best_val_loss,
        "best_checkpoint_path": best_checkpoint_path,
        "checkpoint_dir": ckpt_dir,
    }
    print(f"\n[Done] Best validation loss: {best_val_loss:.4f}")
    if return_summary:
        return summary
    return model


def _ensure_optuna_available() -> None:
    global optuna
    if optuna is None:
        try:
            optuna = importlib.import_module("optuna")
        except ImportError as exc:
            raise ImportError(
                "Optuna is not installed. Install it with `pip install optuna` to use fine-tuning."
            ) from exc
    if optuna is None:
        raise ImportError(
            "Optuna is not installed. Install it with `pip install optuna` to use fine-tuning."
        )


def _build_base_configs(
    args: argparse.Namespace,
) -> tuple[C.TrainConfig, C.ModelConfig]:
    tcfg = C.TrainConfig()
    if args.epochs is not None:
        tcfg.max_epochs = args.epochs
    if args.batch_size is not None:
        tcfg.batch_size = args.batch_size
    if args.lr is not None:
        tcfg.lr = args.lr
    if args.test_fraction is not None:
        tcfg.test_fraction = args.test_fraction
    return tcfg, C.ModelConfig()


def run_optuna_finetuning(args: argparse.Namespace) -> None:
    """Run Optuna hyper-parameter tuning for the selected training mode."""
    _ensure_optuna_available()
    base_tcfg, base_mcfg = _build_base_configs(args)

    execution_dir = _make_execution_dir(args.mode, args.optuna_study_name)
    trials_root = execution_dir / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)
    trial_summary_log = execution_dir / "trial_best_metrics.jsonl"
    _append_jsonl(
        execution_dir / "execution_metadata.jsonl",
        {
            "record_type": "execution_start",
            "mode": args.mode,
            "study_name": args.optuna_study_name,
            "db": args.db,
            "limit": args.limit,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "test_fraction": args.test_fraction,
            "restrict_carbons_to_peaks": args.restrict_carbons_to_peaks,
            "optuna_trials": args.optuna_trials,
            "optuna_timeout": args.optuna_timeout,
            "optuna_storage": args.optuna_storage,
        },
    )

    if args.mode == "forward":
        direction = "minimize"
    else:
        direction = "maximize"

    study = optuna.create_study(
        study_name=args.optuna_study_name,
        storage=args.optuna_storage,
        load_if_exists=bool(args.optuna_storage and args.optuna_study_name),
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=base_tcfg.seed),
    )

    def objective(trial: Any) -> float:
        tcfg = C.suggest_train_config(trial, base=base_tcfg)
        mcfg = C.suggest_model_config(trial, base=base_mcfg)
        trial_dir = trials_root / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_log_path = trial_dir / "trial_log.jsonl"

        _append_jsonl(
            trial_log_path,
            {
                "record_type": "trial_start",
                "trial_number": trial.number,
                "mode": args.mode,
                "optuna_params": dict(trial.params),
                "train_config": _train_cfg_to_dict(tcfg),
                "model_config": _model_cfg_to_dict(mcfg),
            },
        )

        vocab_path = str(trial_dir / "vocab.pkl")
        test_ids_path = str(trial_dir / "test_ids.json")
        train_dl, val_dl, test_dl, vocab = make_dataloaders(
            db_path=args.db,
            limit=args.limit,
            vocab_path=vocab_path,
            train_cfg=tcfg,
            test_ids_path=test_ids_path,
            restrict_carbons_to_peaks=args.restrict_carbons_to_peaks,
        )

        if args.mode == "inverse":
            summary = train_inverse(
                train_dl,
                val_dl,
                vocab,
                tcfg,
                mcfg,
                checkpoint_dir=str(trial_dir / "inverse"),
                shared_log_path=str(trial_log_path),
                return_summary=True,
            )
            trial.set_user_attr("best_token_acc", summary["best_token_acc"])
            trial.set_user_attr("best_tanimoto", summary["best_tanimoto"])
            trial.set_user_attr("best_validity", summary["best_validity"])
            trial.set_user_attr("best_val_loss", summary["best_val_loss"])
            _append_jsonl(
                trial_summary_log,
                {
                    "trial_number": trial.number,
                    "mode": args.mode,
                    "score": float(summary["best_inverse_score"]),
                    "best_token_acc": summary["best_token_acc"],
                    "best_tanimoto": summary["best_tanimoto"],
                    "best_validity": summary["best_validity"],
                    "best_val_loss": summary["best_val_loss"],
                    "best_checkpoint_path": summary["best_checkpoint_path"],
                    "test_ids_path": test_ids_path,
                },
            )
            return float(summary["best_inverse_score"])

        if args.mode == "forward":
            summary = train_forward(
                train_dl,
                val_dl,
                vocab,
                tcfg,
                mcfg,
                checkpoint_dir=str(trial_dir / "forward"),
                shared_log_path=str(trial_log_path),
                return_summary=True,
            )
            trial.set_user_attr("best_val_loss", summary["best_val_loss"])
            _append_jsonl(
                trial_summary_log,
                {
                    "trial_number": trial.number,
                    "mode": args.mode,
                    "score": float(summary["best_val_loss"]),
                    "best_val_loss": summary["best_val_loss"],
                    "best_checkpoint_path": summary["best_checkpoint_path"],
                    "test_ids_path": test_ids_path,
                },
            )
            return float(summary["best_val_loss"])

        inverse_summary = train_inverse(
            train_dl,
            val_dl,
            vocab,
            tcfg,
            mcfg,
            checkpoint_dir=str(trial_dir / "inverse"),
            shared_log_path=str(trial_log_path),
            return_summary=True,
        )
        forward_summary = train_forward(
            train_dl,
            val_dl,
            vocab,
            tcfg,
            mcfg,
            checkpoint_dir=str(trial_dir / "forward"),
            shared_log_path=str(trial_log_path),
            return_summary=True,
        )
        combined_score = float(inverse_summary["best_inverse_score"]) - float(
            forward_summary["best_val_loss"]
        )
        trial.set_user_attr("best_token_acc", inverse_summary["best_token_acc"])
        trial.set_user_attr("best_tanimoto", inverse_summary["best_tanimoto"])
        trial.set_user_attr("best_validity", inverse_summary["best_validity"])
        trial.set_user_attr("inverse_score", inverse_summary["best_inverse_score"])
        trial.set_user_attr("forward_best_val_loss", forward_summary["best_val_loss"])
        _append_jsonl(
            trial_summary_log,
            {
                "trial_number": trial.number,
                "mode": args.mode,
                "score": combined_score,
                "best_token_acc": inverse_summary["best_token_acc"],
                "best_tanimoto": inverse_summary["best_tanimoto"],
                "best_validity": inverse_summary["best_validity"],
                "inverse_score": inverse_summary["best_inverse_score"],
                "forward_best_val_loss": forward_summary["best_val_loss"],
                "inverse_best_checkpoint_path": inverse_summary["best_checkpoint_path"],
                "forward_best_checkpoint_path": forward_summary["best_checkpoint_path"],
                "test_ids_path": test_ids_path,
            },
        )
        return combined_score

    study.optimize(objective, n_trials=args.optuna_trials, timeout=args.optuna_timeout)

    print(f"\n[Optuna] Study complete: {study.study_name or 'unnamed-study'}")
    print(f"[Optuna] Best value: {study.best_value:.6f}")
    print("[Optuna] Best params:")
    for key, value in study.best_trial.params.items():
        print(f"  - {key}: {value}")
    if study.best_trial.user_attrs:
        print("[Optuna] Best trial metrics:")
        for key, value in study.best_trial.user_attrs.items():
            print(f"  - {key}: {value}")

    best_trial_dir = trials_root / f"trial_{study.best_trial.number:04d}"
    best_artifacts_dir = execution_dir / "best_checkpoint"
    best_artifacts_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "inverse":
        _copy_if_exists(
            best_trial_dir / "inverse" / "best.pt",
            best_artifacts_dir / "best.pt",
        )
    elif args.mode == "forward":
        _copy_if_exists(
            best_trial_dir / "forward" / "best.pt",
            best_artifacts_dir / "best.pt",
        )
    else:
        _copy_if_exists(
            best_trial_dir / "inverse" / "best.pt",
            best_artifacts_dir / "inverse_best.pt",
        )
        _copy_if_exists(
            best_trial_dir / "forward" / "best.pt",
            best_artifacts_dir / "forward_best.pt",
        )

    _copy_if_exists(best_trial_dir / "vocab.pkl", execution_dir / "vocab.pkl")
    _copy_if_exists(best_trial_dir / "test_ids.json", execution_dir / "test_ids.json")
    _append_jsonl(
        execution_dir / "execution_metadata.jsonl",
        {
            "record_type": "execution_complete",
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": dict(study.best_trial.params),
            "best_user_attrs": dict(study.best_trial.user_attrs),
        },
    )


def _build_forward_targets(
    spec: torch.Tensor,  # (B, P, 4)
    spec_mask: torch.Tensor,  # (B, P)
    gf: torch.Tensor,  # (B, 20)
) -> dict[str, torch.Tensor]:
    """
    Build regression targets for the forward model from the batch data.

    Returns dict with:
        bins  – (B, 7)           bin counts extracted from global features
        count – (B, 1)           peak count (from global features)
        peaks – (B, MAX_PEAKS)   normalised ppm positions (0–1)
        peak_mask – (B, MAX_PEAKS) bool: True where valid
    """
    B = spec.size(0)
    device = spec.device

    # Bin counts: gf indices 7..13 are bin_ratios * peak_count
    # We store them as ratios, so multiply by peak_count to get counts
    peak_count = torch.exp(gf[:, 0]) - 1  # gf[0] = log1p(peak_count)
    bin_ratios = gf[:, 7:14]  # 7 bins
    bin_counts = bin_ratios * peak_count.unsqueeze(1)

    # Peak positions: expand ppm bins according to encoded peak intensities.
    ppm_bins = spec[:, :, 0].float()
    intensities = spec[:, :, 3].clamp(min=1).long()
    expanded_ppm_rows = []
    for b in range(B):
        ppm_values = []
        for p in range(spec.size(1)):
            if spec_mask[b, p]:
                continue
            repeat = int(intensities[b, p].item())
            ppm_values.extend([ppm_bins[b, p].item() / C.NUM_PPM_BINS] * repeat)
        expanded_ppm_rows.append(ppm_values[: C.MAX_PEAKS])

    # Pad to MAX_PEAKS
    ppm_padded = torch.zeros(B, C.MAX_PEAKS, device=device)
    peak_valid_mask = torch.zeros(B, C.MAX_PEAKS, dtype=torch.bool, device=device)
    for b, ppm_values in enumerate(expanded_ppm_rows):
        if not ppm_values:
            continue
        n = min(len(ppm_values), C.MAX_PEAKS)
        ppm_padded[b, :n] = torch.tensor(ppm_values[:n], device=device)
        peak_valid_mask[b, :n] = True

    return {
        "bins": bin_counts,
        "count": peak_count.unsqueeze(1),
        "peaks": ppm_padded,
        "peak_mask": peak_valid_mask,
    }


def _forward_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Combined loss for the forward model."""
    # Bin histogram loss (MSE)
    bin_loss = F.mse_loss(preds["bins"], targets["bins"])

    # Peak count loss (MSE on log-scale)
    count_loss = F.mse_loss(preds["count"], targets["count"])

    # Peak position loss (only on valid peaks)
    peak_mask = targets["peak_mask"]
    if peak_mask.any():
        pred_peaks = preds["peaks"]
        tgt_peaks = targets["peaks"]
        masked_pred = pred_peaks[peak_mask]
        masked_tgt = tgt_peaks[peak_mask]
        peak_loss = F.mse_loss(masked_pred, masked_tgt)
    else:
        peak_loss = torch.tensor(0.0, device=preds["bins"].device)

    return bin_loss + 0.5 * count_loss + 2.0 * peak_loss


@torch.no_grad()
def evaluate_forward(
    model: nn.Module,
    val_dl: DataLoader,
    vocab: SelfiesVocab,
    max_eval_batches: int = 100,
) -> float:
    """Evaluate forward model. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n = 0

    for batch_idx, (
        spec,
        spec_mask,
        gf,
        formula_vec,
        sf_ids,
        sf_mask,
        ids,
    ) in enumerate(val_dl):
        if batch_idx >= max_eval_batches:
            break

        sf_ids = sf_ids.to(DEVICE)
        spec = spec.to(DEVICE)
        spec_mask = spec_mask.to(DEVICE)
        gf = gf.to(DEVICE)

        targets = _build_forward_targets(spec, spec_mask, gf)
        preds = model(sf_ids)
        loss = _forward_loss(preds, targets)

        total_loss += loss.item()
        n += 1

    model.train()
    return total_loss / max(n, 1)


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train NMR Solver models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["inverse", "forward", "both"],
        help="Which model to train",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit rows loaded from DB (for quick experiments)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max_epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch_size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=C.DB_PATH,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=None,
        help="Fraction of rows to hold out as test data",
    )
    parser.add_argument(
        "--test-ids-out",
        type=str,
        default=None,
        help="Path to save held-out test row IDs as JSON",
    )
    parser.add_argument(
        "--restrict-carbons-to-peaks",
        action="store_true",
        help="Only use rows where carbon_count is less than or equal to peak_count",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=None,
        help="Run Optuna fine-tuning for the selected mode with this many trials",
    )
    parser.add_argument(
        "--optuna-timeout",
        type=int,
        default=None,
        help="Optional Optuna timeout in seconds",
    )
    parser.add_argument(
        "--optuna-study-name",
        type=str,
        default=None,
        help="Optional Optuna study name",
    )
    parser.add_argument(
        "--optuna-storage",
        type=str,
        default=None,
        help="Optional Optuna storage URL, e.g. sqlite:///optuna.db",
    )
    parser.add_argument(
        "--optuna-artifact-dir",
        type=str,
        default=None,
        help="Directory where Optuna trial checkpoints/logs should be stored",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use all selected CUDA devices with DataParallel when more than one GPU is available",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated CUDA device ids to use, e.g. '0,1,2,3'",
    )
    args = parser.parse_args()

    _configure_runtime(args)

    if DEVICE.type == "cuda":
        print(
            f"[Runtime] CUDA available: {torch.cuda.device_count()} GPU(s) | selected={GPU_IDS}"
            + (" | multi_gpu=enabled" if USE_MULTI_GPU else " | multi_gpu=disabled")
        )
    else:
        print("[Runtime] CUDA not available; training on CPU")

    if args.optuna_trials is not None:
        run_optuna_finetuning(args)
        return

    # Build configs
    tcfg, mcfg = _build_base_configs(args)

    vocab_path = os.path.join(C.CHECKPOINT_DIR, "vocab.pkl")
    test_ids_path = args.test_ids_out or os.path.join(
        C.CHECKPOINT_DIR,
        "test_ids.json",
    )
    train_dl, val_dl, test_dl, vocab = make_dataloaders(
        db_path=args.db,
        limit=args.limit,
        vocab_path=vocab_path,
        train_cfg=tcfg,
        test_ids_path=test_ids_path,
        restrict_carbons_to_peaks=args.restrict_carbons_to_peaks,
    )

    if len(test_dl.dataset) > 0:
        print(f"[Data] Held out {len(test_dl.dataset)} rows for final testing")

    if args.mode in ("inverse", "both"):
        train_inverse(train_dl, val_dl, vocab, tcfg, mcfg, resume_path=args.resume)

    if args.mode in ("forward", "both"):
        train_forward(train_dl, val_dl, vocab, tcfg, mcfg, resume_path=args.resume)


if __name__ == "__main__":
    main()
