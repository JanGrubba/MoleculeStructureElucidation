"""
models.py – Transformer models for forward and inverse NMR prediction.

ForwardModel   : SELFIES → ¹³C NMR spectrum  (predict peaks from structure)
InverseModel   : ¹³C NMR spectrum → SELFIES  (recover structure from peaks)

Both share the same architecture: encoder-decoder Transformer with global
feature injection.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from nmrsolver import config as C
except Exception:
    try:
        from . import config as C
    except Exception:
        import config as C


def _causal_bool_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """Boolean causal mask for Transformer decoder attention."""
    return torch.triu(
        torch.ones(size, size, dtype=torch.bool, device=device),
        diagonal=1,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Positional encoding
# ──────────────────────────────────────────────────────────────────────────────


class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MAB(nn.Module):
    """Multihead attention block for set encoders."""

    def __init__(
        self, dim_q: int, dim_k: int, dim_v: int, num_heads: int, dropout: float = 0.1
    ):
        super().__init__()
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.fc_q = nn.Linear(dim_q, dim_v)
        self.fc_k = nn.Linear(dim_k, dim_v)
        self.fc_v = nn.Linear(dim_k, dim_v)
        self.fc_o = nn.Linear(dim_v, dim_v)
        self.ln0 = nn.LayerNorm(dim_v)
        self.ln1 = nn.LayerNorm(dim_v)
        self.ff = nn.Sequential(
            nn.Linear(dim_v, 4 * dim_v),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim_v, dim_v),
        )

    def forward(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = q_in.size(0)
        q = self.fc_q(q_in).view(
            bsz, q_in.size(1), self.num_heads, self.dim_v // self.num_heads
        )
        k = self.fc_k(k_in).view(
            bsz, k_in.size(1), self.num_heads, self.dim_v // self.num_heads
        )
        v = self.fc_v(k_in).view(
            bsz, k_in.size(1), self.num_heads, self.dim_v // self.num_heads
        )

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 1, 3)

        attn = torch.matmul(q, k) / math.sqrt(self.dim_v // self.num_heads)
        if key_padding_mask is not None:
            mask_value = torch.finfo(attn.dtype).min
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), mask_value
            )
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(bsz, q_in.size(1), self.dim_v)
        out = self.ln0(self.fc_o(out) + q_in)
        return self.ln1(out + self.ff(out))


class ISAB(nn.Module):
    """Induced self-attention block."""

    def __init__(self, dim: int, num_heads: int, num_inds: int, dropout: float = 0.1):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, num_inds, dim))
        self.mab0 = MAB(dim, dim, dim, num_heads, dropout)
        self.mab1 = MAB(dim, dim, dim, num_heads, dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        i = self.inducing.expand(x.size(0), -1, -1)
        h = self.mab0(i, x, key_padding_mask=key_padding_mask)
        return self.mab1(x, h)


class PMA(nn.Module):
    """Pooling by multihead attention."""

    def __init__(
        self, dim: int, num_heads: int, num_seeds: int = 1, dropout: float = 0.1
    ):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MAB(dim, dim, dim, num_heads, dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        s = self.seeds.expand(x.size(0), -1, -1)
        return self.mab(s, x, key_padding_mask=key_padding_mask)


# ──────────────────────────────────────────────────────────────────────────────
#  Peak embedding (for spectrum encoder)
# ──────────────────────────────────────────────────────────────────────────────


class PeakEmbedding(nn.Module):
    """
    Convert a peak tuple (ppm_bin, mult_idx, j_bin, intensity) → d_model vector.

    Each component gets its own embedding table; the three embeddings are
    summed (like token + segment + position in BERT).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.ppm_emb = nn.Embedding(C.NUM_PPM_BINS, d_model)
        self.mult_emb = nn.Embedding(len(C.MULT_VOCAB), d_model)
        self.j_emb = nn.Embedding(C.NUM_J_BINS, d_model)
        self.intensity_emb = nn.Embedding(C.MAX_PEAK_INTENSITY + 1, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        """peaks: (batch, n_peaks, 4) → (batch, n_peaks, d_model)"""
        ppm = self.ppm_emb(peaks[:, :, 0])
        mult = self.mult_emb(peaks[:, :, 1])
        j = self.j_emb(peaks[:, :, 2])
        intensity = self.intensity_emb(peaks[:, :, 3].clamp(0, C.MAX_PEAK_INTENSITY))
        return self.norm(ppm + mult + j + intensity)


# ──────────────────────────────────────────────────────────────────────────────
#  Global feature projector
# ──────────────────────────────────────────────────────────────────────────────


class GlobalProjector(nn.Module):
    """
    Project the global feature vector into d_model space and prepend it as a
    special [CLS]-like token to the encoder sequence.
    """

    def __init__(self, d_model: int, n_features: int = 20):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, gf: torch.Tensor) -> torch.Tensor:
        """gf: (batch, n_features) → (batch, 1, d_model)"""
        return self.proj(gf).unsqueeze(1)


class SpectrumSetEncoder(nn.Module):
    """Permutation-invariant set encoder for encoded NMR peaks."""

    def __init__(self, cfg: C.ModelConfig):
        super().__init__()
        d = cfg.enc_d_model
        self.layers = nn.ModuleList(
            [
                ISAB(
                    dim=d,
                    num_heads=cfg.enc_nhead,
                    num_inds=cfg.enc_num_inds,
                    dropout=cfg.enc_dropout,
                )
                for _ in range(cfg.enc_layers)
            ]
        )
        self.pma = PMA(
            d, cfg.enc_nhead, num_seeds=cfg.enc_num_seeds, dropout=cfg.enc_dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        peak_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = layer(x, key_padding_mask=peak_mask)
        pooled = self.pma(x, key_padding_mask=peak_mask)
        return x, pooled


# ──────────────────────────────────────────────────────────────────────────────
#  Inverse Model: Spectrum → SELFIES
# ──────────────────────────────────────────────────────────────────────────────


class InverseModel(nn.Module):
    """
    Encoder: peak embeddings + global features → contextualised representations
    Decoder: autoregressive SELFIES generation with cross-attention to encoder
    """

    def __init__(self, vocab_size: int, cfg: C.ModelConfig = C.ModelConfig()):
        super().__init__()
        self.cfg = cfg
        d = cfg.enc_d_model

        # ── Encoder ───────────────────────────────────────────────────────────
        self.peak_emb = PeakEmbedding(d)
        self.set_encoder = SpectrumSetEncoder(cfg)
        self.global_proj = GlobalProjector(d, cfg.num_global_features)
        self.formula_proj = GlobalProjector(d, cfg.num_formula_features)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.enc_nhead,
            dim_feedforward=cfg.enc_dim_ff,
            dropout=cfg.enc_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        self.tok_emb = nn.Embedding(vocab_size, cfg.dec_d_model, padding_idx=0)
        self.dec_pe = SinusoidalPE(
            cfg.dec_d_model, max_len=C.MAX_SELFIES_LEN, dropout=cfg.dec_dropout
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.dec_d_model,
            nhead=cfg.dec_nhead,
            dim_feedforward=cfg.dec_dim_ff,
            dropout=cfg.dec_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.dec_layers)

        # ── Output head ──────────────────────────────────────────────────────
        self.out_proj = nn.Linear(cfg.dec_d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self,
        peaks: torch.Tensor,
        peak_mask: torch.Tensor,
        global_feats: torch.Tensor,
        formula_vector: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the spectrum.

        Returns:
            memory     – (batch, 1 + n_peaks, d_model)
            memory_mask – (batch, 1 + n_peaks) bool – True where padded
        """
        peak_tokens = self.peak_emb(peaks)
        peak_tokens, pooled = self.set_encoder(peak_tokens, peak_mask)

        cond_tokens = [self.global_proj(global_feats), pooled]
        cond_masks = [
            torch.zeros(peaks.size(0), 1, dtype=torch.bool, device=peaks.device),
            torch.zeros(
                pooled.size(0), pooled.size(1), dtype=torch.bool, device=peaks.device
            ),
        ]

        if formula_vector is not None:
            cond_tokens.insert(1, self.formula_proj(formula_vector))
            cond_masks.insert(
                1,
                torch.zeros(peaks.size(0), 1, dtype=torch.bool, device=peaks.device),
            )

        enc_in = torch.cat(cond_tokens + [peak_tokens], dim=1)
        enc_mask = torch.cat(cond_masks + [peak_mask], dim=1)
        memory = self.encoder(enc_in, src_key_padding_mask=enc_mask)
        return memory, enc_mask

    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode one step (teacher-forced).

        Args:
            tgt_ids     – (batch, tgt_len) token IDs
            memory      – encoder output
            memory_mask – encoder padding mask

        Returns:
            logits – (batch, tgt_len, vocab_size)
        """
        tgt_emb = self.tok_emb(tgt_ids)
        tgt_emb = self.dec_pe(tgt_emb)

        # Causal mask
        T = tgt_ids.size(1)
        causal = _causal_bool_mask(T, device=tgt_ids.device)

        # Padding mask for target
        tgt_pad_mask = tgt_ids == 0

        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_mask,
        )
        return self.out_proj(out)

    def forward(
        self,
        peaks: torch.Tensor,
        peak_mask: torch.Tensor,
        global_feats: torch.Tensor,
        formula_vector: Optional[torch.Tensor],
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass (teacher-forced training).

        Returns logits: (batch, tgt_len, vocab_size)
        """
        memory, mem_mask = self.encode(peaks, peak_mask, global_feats, formula_vector)
        logits = self.decode(tgt_ids, memory, mem_mask)
        return logits

    @torch.no_grad()
    def generate_greedy(
        self,
        peaks: torch.Tensor,
        peak_mask: torch.Tensor,
        global_feats: torch.Tensor,
        formula_vector: Optional[torch.Tensor] = None,
        max_len: int = C.MAX_SELFIES_LEN,
        bos_idx: int = 1,
        eos_idx: int = 2,
        unk_idx: int = 3,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding. Returns (batch, seq_len) IDs.
        Masks out <unk> so it is never selected during generation."""
        self.eval()
        B = peaks.size(0)
        memory, mem_mask = self.encode(peaks, peak_mask, global_feats, formula_vector)

        generated = torch.full((B, 1), bos_idx, dtype=torch.long, device=peaks.device)
        finished = torch.zeros(B, dtype=torch.bool, device=peaks.device)

        for _ in range(max_len - 1):
            logits = self.decode(generated, memory, mem_mask)
            # Mask out <unk> token
            logits[:, -1, unk_idx] = torch.finfo(logits.dtype).min
            step_logits = logits[:, -1] / max(temperature, 1e-6)
            next_tok = step_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            generated = torch.cat([generated, next_tok], dim=1)
            finished |= next_tok.squeeze(-1) == eos_idx
            if finished.all():
                break

        return generated

    @torch.no_grad()
    def generate_beam(
        self,
        peaks: torch.Tensor,
        peak_mask: torch.Tensor,
        global_feats: torch.Tensor,
        formula_vector: Optional[torch.Tensor] = None,
        beam_size: int = 10,
        max_len: int = C.MAX_SELFIES_LEN,
        bos_idx: int = 1,
        eos_idx: int = 2,
        unk_idx: int = 3,
        length_penalty: float = 0.7,
        num_return_sequences: Optional[int] = None,
    ) -> list[list[tuple[list[int], float]]]:
        """
        Beam search decoding.  Processes ONE sample at a time for simplicity.

        Returns:
            For each item in the batch, a list of (token_ids, log_prob) sorted
            by descending score.
        """
        self.eval()
        B = peaks.size(0)
        results = []
        num_return_sequences = num_return_sequences or beam_size

        for b in range(B):
            p = peaks[b : b + 1]  # (1, P, 4)
            pm = peak_mask[b : b + 1]  # (1, P)
            gf = global_feats[b : b + 1]  # (1, 20)
            fv = formula_vector[b : b + 1] if formula_vector is not None else None
            memory, mem_mask = self.encode(p, pm, gf, fv)

            # Each beam: (token_ids_list, cumulative_log_prob)
            beams: list[tuple[list[int], float]] = [([bos_idx], 0.0)]
            completed: list[tuple[list[int], float]] = []

            for _step in range(max_len - 1):
                all_candidates: list[tuple[list[int], float]] = []

                for seq, score in beams:
                    tgt = torch.tensor([seq], dtype=torch.long, device=peaks.device)
                    logits = self.decode(tgt, memory, mem_mask)  # (1, T, V)
                    logits[0, -1, unk_idx] = torch.finfo(logits.dtype).min
                    log_probs = F.log_softmax(logits[0, -1], dim=-1)

                    topk_lp, topk_idx = log_probs.topk(beam_size)
                    for lp, idx in zip(topk_lp.tolist(), topk_idx.tolist()):
                        new_seq = seq + [idx]
                        new_score = score + lp
                        if idx == eos_idx:
                            completed.append((new_seq, new_score))
                        else:
                            all_candidates.append((new_seq, new_score))

                # Keep top-k beams
                all_candidates.sort(key=lambda x: -x[1])
                beams = all_candidates[:beam_size]

                if not beams:
                    break

            # Add unfinished beams
            completed.extend(beams)
            completed.sort(
                key=lambda x: -(x[1] / (max(len(x[0]), 1) ** max(length_penalty, 1e-6)))
            )
            results.append(completed[:num_return_sequences])

        return results


# ──────────────────────────────────────────────────────────────────────────────
#  Forward Model: SELFIES → Spectrum
# ──────────────────────────────────────────────────────────────────────────────


class ForwardModel(nn.Module):
    """
    Predicts ¹³C NMR spectrum features from a molecular structure (SELFIES).

    Output: for each molecule, predicts the bin histogram (regression on the 7
    bin counts) and a set of peak positions (ppm regression).

    This is used for candidate re-scoring: generate candidate → predict its
    spectrum → compare with query spectrum.
    """

    def __init__(self, vocab_size: int, cfg: C.ModelConfig = C.ModelConfig()):
        super().__init__()
        d = cfg.enc_d_model

        # Token embeddings
        self.tok_emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pe = SinusoidalPE(d, max_len=C.MAX_SELFIES_LEN, dropout=cfg.enc_dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.enc_nhead,
            dim_feedforward=cfg.enc_dim_ff,
            dropout=cfg.enc_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=cfg.enc_layers,
            enable_nested_tensor=False,
        )

        # Output heads
        # 1. Bin histogram (7 values: bin_0_50 … bin_out)
        self.bin_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 7),
        )

        # 2. Peak count
        self.count_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Linear(d // 2, 1),
        )

        # 3. Peak positions (predict up to MAX_PEAKS ppm values via regression)
        self.peak_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, C.MAX_PEAKS),  # each output = predicted ppm (normalised)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sf_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            sf_ids: (batch, seq_len) SELFIES token IDs

        Returns dict with:
            bins   – (batch, 7)          predicted bin counts
            count  – (batch, 1)          predicted peak count
            peaks  – (batch, MAX_PEAKS)  predicted ppm values (0–1 normalised)
        """
        pad_mask = sf_ids == 0
        x = self.tok_emb(sf_ids)
        x = self.pe(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        # Pool: mean over non-padded positions
        mask_expanded = (~pad_mask).unsqueeze(-1).float()
        pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        return {
            "bins": self.bin_head(pooled),
            "count": self.count_head(pooled),
            "peaks": torch.sigmoid(self.peak_head(pooled)),  # normalised 0-1
        }
