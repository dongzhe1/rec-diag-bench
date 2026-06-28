"""SASRec: self-attentive sequential recommendation with BPR loss."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .base import infer_shape, user_pos_sets

logger = logging.getLogger(__name__)


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        dim: int = 64,
        max_seq_len: int = 50,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_items = n_items
        self.max_seq_len = max_seq_len
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """(B, L) item indices -> (B, dim) last-position hidden."""
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.ln(self.drop(x))

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=seq.device
        )
        pad_mask = seq == 0

        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        lens = (seq != 0).sum(dim=1).clamp(min=1) - 1
        last_hidden = x[torch.arange(B, device=seq.device), lens]
        return last_hidden

    def score_items(self, seq: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Score items by dot product with sequence hidden state."""
        hidden = self.forward(seq)
        item_vecs = self.item_emb(item_ids)
        if item_vecs.dim() == 2:
            return (hidden * item_vecs).sum(dim=-1)
        return torch.einsum("bd,bkd->bk", hidden, item_vecs)


def _build_sequences(train: pd.DataFrame, max_seq_len: int) -> dict[int, list[int]]:
    seqs: dict[int, list[int]] = {}
    for user, grp in train.sort_values("timestamp").groupby("user_idx", sort=False):
        items = (grp["item_idx"].values + 1).tolist()
        seqs[int(user)] = items[-max_seq_len:]
    return seqs


def _pad_seq(seq: list[int], max_len: int) -> list[int]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


def train_sasrec(
    train: pd.DataFrame,
    shape,
    dim: int = 64,
    max_seq_len: int = 50,
    n_heads: int = 2,
    n_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    device: str = "cpu",
    seed: int = 42,
    patience: int = 20,
) -> tuple[SASRec, dict[int, list[int]]]:
    """Train SASRec and return ``(model, per-user sequences)``.

    Split out from :func:`train_sasrec_score_candidates` so callers that need the
    trained model directly (e.g. full-catalogue retrieval) reuse the same loop.
    Sequences are 1-indexed item ids (0 = padding).
    """
    rng = np.random.default_rng(seed)
    model = SASRec(
        n_items=shape.n_items,
        dim=dim,
        max_seq_len=max_seq_len,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_sets = user_pos_sets(train)
    seqs = _build_sequences(train, max_seq_len)
    users_with_seqs = [u for u, s in seqs.items() if len(s) >= 2]

    if not users_with_seqs:
        logger.warning("[SASRec] no users with >=2 interactions; model left untrained")
        model.eval()
        return model, seqs

    best_loss = float("inf")
    wait = 0

    for epoch in range(epochs):
        model.train()
        rng.shuffle(users_with_seqs)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(users_with_seqs), batch_size):
            batch_users = users_with_seqs[start : start + batch_size]
            batch_seqs = []
            batch_pos = []
            batch_neg = []

            for u in batch_users:
                seq = seqs[u]
                if len(seq) < 2:
                    continue
                input_seq = _pad_seq(seq[:-1], max_seq_len)
                pos_item = seq[-1]
                neg_item = rng.integers(1, shape.n_items + 1)
                while (neg_item - 1) in pos_sets.get(u, set()):
                    neg_item = rng.integers(1, shape.n_items + 1)

                batch_seqs.append(input_seq)
                batch_pos.append(pos_item)
                batch_neg.append(int(neg_item))

            if not batch_seqs:
                continue

            seq_t = torch.tensor(batch_seqs, dtype=torch.long, device=device)
            pos_t = torch.tensor(batch_pos, dtype=torch.long, device=device)
            neg_t = torch.tensor(batch_neg, dtype=torch.long, device=device)

            pos_scores = model.score_items(seq_t, pos_t)
            neg_scores = model.score_items(seq_t, neg_t)
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"[SASRec] epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info(f"[SASRec] early stop at epoch {epoch + 1}")
                break

    model.eval()
    return model, seqs


def train_sasrec_score_candidates(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    items: pd.DataFrame,
    dim: int = 64,
    max_seq_len: int = 50,
    n_heads: int = 2,
    n_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    device: str = "cpu",
    seed: int = 42,
    patience: int = 20,
) -> pd.DataFrame:
    shape = infer_shape(train, candidates, items)
    model, seqs = train_sasrec(
        train,
        shape,
        dim,
        max_seq_len,
        n_heads,
        n_layers,
        dropout,
        epochs,
        batch_size,
        lr,
        weight_decay,
        device,
        seed,
        patience,
    )

    if not any(len(s) >= 2 for s in seqs.values()):
        return candidates[["user_idx", "item_idx"]].assign(score=0.0)

    all_scores = np.empty(len(candidates), dtype=np.float32)
    u_idx = candidates["user_idx"].values
    i_idx = candidates["item_idx"].values

    with torch.no_grad():
        for start in range(0, len(candidates), batch_size * max_seq_len):
            end = min(start + batch_size * max_seq_len, len(candidates))
            chunk_users = u_idx[start:end]
            chunk_items = i_idx[start:end]

            unique_users = np.unique(chunk_users)
            batch_seq_list = []
            for u in unique_users:
                seq = seqs.get(int(u), [])
                batch_seq_list.append(_pad_seq(seq, max_seq_len))

            seq_t = torch.tensor(batch_seq_list, dtype=torch.long, device=device)
            hidden = model.forward(seq_t)
            hidden_np = hidden.cpu().numpy()

            user_to_idx = {int(u): i for i, u in enumerate(unique_users)}
            items_1indexed = chunk_items + 1
            item_emb = model.item_emb.weight.data.cpu().numpy()

            for j in range(end - start):
                u = int(chunk_users[j])
                i = int(items_1indexed[j])
                h = hidden_np[user_to_idx[u]]
                all_scores[start + j] = np.dot(h, item_emb[i])

    return candidates[["user_idx", "item_idx"]].assign(score=all_scores.astype(float))
