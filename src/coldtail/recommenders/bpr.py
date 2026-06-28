"""Bayesian Personalized Ranking Matrix Factorization (BPR-MF) recommender."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .base import (
    _build_pos_arrays,
    build_pos_csr,
    infer_shape,
    sample_batch,
    user_pos_sets,
)

logger = logging.getLogger(__name__)


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_emb(users) * self.item_emb(items)).sum(dim=-1)


def train_bpr(
    train: pd.DataFrame,
    shape,
    dim: int = 64,
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    device: str = "cpu",
    seed: int = 42,
) -> BPRMF:
    """Train a BPR-MF model and return it.

    Split out from :func:`train_bpr_score_candidates` so callers that need the
    trained embeddings directly (e.g. full-catalog retrieval) can reuse the same
    training loop instead of re-implementing it.
    """
    rng = np.random.default_rng(seed)
    model = BPRMF(shape.n_users, shape.n_items, dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    user_pos = user_pos_sets(train)
    users_arr = np.asarray(list(user_pos.keys()), dtype=np.int64)

    # Pre-build sampling structures once; reused every step
    pos_arrays = _build_pos_arrays(user_pos, users_arr)
    pos_indptr, pos_indices = build_pos_csr(user_pos, users_arr)

    steps_per_epoch = max(1, len(train) // batch_size)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            u, p, n = sample_batch(
                rng,
                users_arr,
                user_pos,
                shape.n_items,
                batch_size,
                _pos_arrays=pos_arrays,
                _pos_indptr=pos_indptr,
                _pos_indices=pos_indices,
            )
            u_t = torch.from_numpy(u)
            p_t = torch.from_numpy(p)
            n_t = torch.from_numpy(n)
            if device != "cpu":
                u_t = u_t.pin_memory()
                p_t = p_t.pin_memory()
                n_t = n_t.pin_memory()
            u_t = u_t.to(device, non_blocking=(device != "cpu"))
            p_t = p_t.to(device, non_blocking=(device != "cpu"))
            n_t = n_t.to(device, non_blocking=(device != "cpu"))
            pos_score = model.score(u_t, p_t)
            neg_score = model.score(u_t, n_t)
            loss = -F.logsigmoid(pos_score - neg_score).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())
        logger.info("BPR epoch %d/%d loss=%.4f", epoch + 1, epochs, total_loss / steps_per_epoch)
    return model


def train_bpr_score_candidates(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    items: pd.DataFrame | None = None,
    dim: int = 64,
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    device: str = "cpu",
    seed: int = 42,
) -> pd.DataFrame:
    shape = infer_shape(train, candidates, items)
    model = train_bpr(
        train, shape, dim, epochs, batch_size, lr, weight_decay, device, seed
    )
    return score_bpr_model(model, candidates, device)


def score_bpr_model(
    model: BPRMF,
    candidates: pd.DataFrame,
    device: str,
    batch_size: int = 65536,
) -> pd.DataFrame:
    """Score all candidates; convert indices to tensors once for efficiency."""
    model.eval()

    all_u = torch.as_tensor(candidates.user_idx.to_numpy(dtype=np.int32), device=device)
    all_it = torch.as_tensor(
        candidates.item_idx.to_numpy(dtype=np.int32), device=device
    )

    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(all_u), batch_size):
            s = (
                model.score(
                    all_u[start : start + batch_size],
                    all_it[start : start + batch_size],
                )
                .cpu()
                .numpy()
            )
            scores.append(s)

    return candidates[["user_idx", "item_idx"]].assign(
        score=np.concatenate(scores).astype(float)
    )
