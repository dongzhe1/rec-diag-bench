"""Light Graph Convolution Network (LightGCN) recommender."""

from __future__ import annotations

import logging
from typing import Tuple

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


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.02)
        nn.init.normal_(self.item_emb.weight, std=0.02)

    def propagate(self, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Graph propagation with in-place layer accumulation."""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        out = all_emb
        x = all_emb
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            out = out + x
        out = out / (self.n_layers + 1)
        return out[: self.n_users], out[self.n_users :]


def _build_norm_adj(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    device: str,
) -> torch.Tensor:
    """Build symmetrically-normalized adjacency matrix (sparse CSR format)."""
    users = train.user_idx.astype(int).to_numpy()
    items = train.item_idx.astype(int).to_numpy() + n_users
    rows = np.concatenate([users, items])
    cols = np.concatenate([items, users])
    vals = np.ones(len(rows), dtype=np.float32)
    n = n_users + n_items

    deg = np.bincount(rows, minlength=n).astype(np.float32)
    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-8)
    norm_vals = vals * deg_inv_sqrt[rows] * deg_inv_sqrt[cols]

    idx = torch.as_tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
    val = torch.as_tensor(norm_vals, dtype=torch.float32, device=device)

    adj_coo = torch.sparse_coo_tensor(idx, val, size=(n, n), device=device).coalesce()
    return adj_coo.to_sparse_csr()


def train_lightgcn(
    train: pd.DataFrame,
    shape,
    dim: int = 64,
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    n_layers: int = 2,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[LightGCN, torch.Tensor]:
    """Train LightGCN and return ``(model, normalised_adjacency)``.

    Split out from :func:`train_lightgcn_score_candidates` so callers that need
    the trained embeddings directly (e.g. full-catalog retrieval) can reuse the
    exact same training without re-implementing it.
    """
    rng = np.random.default_rng(seed)
    model = LightGCN(shape.n_users, shape.n_items, dim, n_layers).to(device)
    adj = _build_norm_adj(train, shape.n_users, shape.n_items, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    user_pos = user_pos_sets(train)
    users_arr = np.asarray(list(user_pos.keys()), dtype=np.int64)

    pos_arrays = _build_pos_arrays(user_pos, users_arr)
    pos_indptr, pos_indices = build_pos_csr(user_pos, users_arr)

    steps_per_epoch = max(1, len(train) // batch_size)
    model.train()
    for epoch in range(epochs):
        # Propagate once per epoch; computation graph shared across all steps
        user_final, item_final = model.propagate(adj)

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
            u_t = torch.as_tensor(u, device=device)
            p_t = torch.as_tensor(p, device=device)
            n_t = torch.as_tensor(n, device=device)
            pos_score = (user_final[u_t] * item_final[p_t]).sum(dim=-1)
            neg_score = (user_final[u_t] * item_final[n_t]).sum(dim=-1)
            loss = -F.logsigmoid(pos_score - neg_score).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            opt.step()
            total_loss += float(loss.detach().cpu())
        logger.info("LightGCN epoch %d/%d loss=%.4f", epoch + 1, epochs, total_loss / steps_per_epoch)
    return model, adj


def train_lightgcn_score_candidates(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    items: pd.DataFrame | None = None,
    dim: int = 64,
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    n_layers: int = 2,
    device: str = "cpu",
    seed: int = 42,
) -> pd.DataFrame:
    shape = infer_shape(train, candidates, items)
    model, adj = train_lightgcn(
        train, shape, dim, epochs, batch_size, lr, weight_decay, n_layers, device, seed
    )
    return score_lightgcn_model(model, adj, candidates, device)


def score_lightgcn_model(
    model: LightGCN,
    adj: torch.Tensor,
    candidates: pd.DataFrame,
    device: str,
    batch_size: int = 65536,
) -> pd.DataFrame:
    """Score all candidates efficiently (int32 tensors, no per-batch pandas overhead)."""
    model.eval()

    all_u = torch.as_tensor(candidates.user_idx.to_numpy(dtype=np.int32), device=device)
    all_it = torch.as_tensor(
        candidates.item_idx.to_numpy(dtype=np.int32), device=device
    )

    scores: list[np.ndarray] = []
    with torch.no_grad():
        user_final, item_final = model.propagate(adj)
        for start in range(0, len(all_u), batch_size):
            u = all_u[start : start + batch_size]
            it = all_it[start : start + batch_size]
            s = (user_final[u] * item_final[it]).sum(dim=-1).cpu().numpy()
            scores.append(s)

    return candidates[["user_idx", "item_idx"]].assign(
        score=np.concatenate(scores).astype(float)
    )
