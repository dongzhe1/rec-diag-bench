"""Two-tower retriever: user tower (history text) + item tower (item text).

Trains a lightweight projection head on top of a frozen text encoder, using
InfoNCE loss on training interactions. At inference, all items are encoded once
and user embeddings are built from recent history — full-catalogue ANN retrieval.

This is the "Priority C" experiment from the WSDM directions: a split-specific
fine-tuned retriever that should narrow (but likely not close) the item_new
coverage gap.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch import nn

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int = 128):
        super().__init__()
        self.user_proj = ProjectionHead(input_dim, proj_dim)
        self.item_proj = ProjectionHead(input_dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07)))

    def forward(self, user_emb: torch.Tensor, item_emb: torch.Tensor):
        u = self.user_proj(user_emb)
        v = self.item_proj(item_emb)
        return u, v


def _encode_texts(
    texts: list[str], model_name: str, device: str, batch_size: int = 256
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model_name, device=device)
    emb = st.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    del st
    return emb.astype(np.float32)


def train_two_tower(
    train: pd.DataFrame,
    items: pd.DataFrame,
    shape,
    model_name: str = "BAAI/bge-base-en-v1.5",
    proj_dim: int = 128,
    max_hist: int = 10,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-4,
    neg_per_pos: int = 15,
    device: str = "cpu",
    seed: int = 42,
    encoder_batch_size: int = 256,
) -> tuple[TwoTowerModel, np.ndarray, np.ndarray]:
    """Train a two-tower model and return (model, user_base_emb, item_emb).

    user_base_emb: frozen encoder embeddings per item (used to build user profiles).
    item_emb: frozen encoder embeddings per item.
    The projection heads are the learned part.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # 1. Encode all item texts with frozen encoder
    texts = (
        items.set_index("item_idx")["text"]
        .reindex(range(shape.n_items))
        .fillna("")
        .astype(str)
        .tolist()
    )
    logger.info("[two-tower] encoding %d item texts with %s...", len(texts), model_name)
    item_emb_np = _encode_texts(texts, model_name, device, encoder_batch_size)
    enc_dim = item_emb_np.shape[1]

    # 2. Build user profiles (mean of recent history item embeddings)
    logger.info("[two-tower] building user profiles (max_hist=%d)...", max_hist)
    recent = (
        train.sort_values("timestamp")
        .groupby("user_idx", sort=False)
        .tail(max_hist)
        .reset_index(drop=True)
    )
    valid_mask = recent["item_idx"].values < len(item_emb_np)
    recent = recent.loc[valid_mask]
    users_arr = recent["user_idx"].values.astype(np.int64)
    items_arr = recent["item_idx"].values.astype(np.int64)
    user_profiles = np.zeros((shape.n_users, enc_dim), dtype=np.float32)
    np.add.at(user_profiles, users_arr, item_emb_np[items_arr])
    user_counts = np.bincount(users_arr, minlength=shape.n_users)
    mask = user_counts > 0
    user_profiles[mask] /= user_counts[mask, None]
    user_profiles = normalize(user_profiles, norm="l2", axis=1).astype(np.float32)

    # 3. Build training pairs
    user_pos = {}
    for u, g in train.groupby("user_idx"):
        user_pos[int(u)] = set(int(x) for x in g.item_idx)
    train_users = np.array(sorted(user_pos.keys()), dtype=np.int64)

    # 4. Train projection heads with InfoNCE
    item_emb_t = torch.as_tensor(item_emb_np, device=device)
    user_prof_t = torch.as_tensor(user_profiles, device=device)

    model = TwoTowerModel(enc_dim, proj_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    n_items = shape.n_items
    steps_per_epoch = max(1, len(train_users) // batch_size)

    logger.info(
        "[two-tower] training | epochs=%d batch=%d neg=%d proj_dim=%d",
        epochs,
        batch_size,
        neg_per_pos,
        proj_dim,
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        perm = rng.permutation(len(train_users))
        for step in range(steps_per_epoch):
            start = step * batch_size
            batch_idx = perm[start : start + batch_size]
            batch_users = train_users[batch_idx]

            # sample one positive per user + neg_per_pos negatives
            pos_items = np.array(
                [rng.choice(list(user_pos[u])) for u in batch_users], dtype=np.int64
            )

            neg_items = np.array(
                [
                    rng.choice(
                        [
                            x
                            for x in rng.integers(0, n_items, size=neg_per_pos * 3)
                            if x not in user_pos[u]
                        ][:neg_per_pos]
                        or [rng.integers(0, n_items)],
                        size=neg_per_pos,
                        replace=True,
                    )
                    for u in batch_users
                ],
                dtype=np.int64,
            )  # (B, neg_per_pos)

            u_emb = user_prof_t[batch_users]  # (B, enc_dim)
            p_emb = item_emb_t[pos_items]  # (B, enc_dim)
            n_emb = item_emb_t[neg_items.ravel()].view(
                len(batch_users), neg_per_pos, enc_dim
            )

            u_proj, p_proj = model(u_emb, p_emb)  # (B, proj_dim) each
            _, n_proj = model(
                u_emb.unsqueeze(1).expand_as(n_emb).reshape(-1, enc_dim),
                n_emb.reshape(-1, enc_dim),
            )
            n_proj = n_proj.view(len(batch_users), neg_per_pos, proj_dim)

            # InfoNCE: positive similarity vs negative similarities
            scale = model.logit_scale.exp().clamp(max=100)
            pos_sim = (u_proj * p_proj).sum(dim=-1, keepdim=True) * scale  # (B, 1)
            neg_sim = (
                torch.bmm(n_proj, u_proj.unsqueeze(-1)).squeeze(-1) * scale
            )  # (B, neg)
            logits = torch.cat([pos_sim, neg_sim], dim=-1)  # (B, 1+neg)
            labels = torch.zeros(len(batch_users), dtype=torch.long, device=device)

            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / steps_per_epoch
        logger.info(
            "[two-tower] epoch %d/%d loss=%.4f scale=%.1f",
            epoch + 1,
            epochs,
            avg_loss,
            model.logit_scale.exp().item(),
        )

    model.eval()
    return model, user_profiles, item_emb_np


def two_tower_topn(
    model: TwoTowerModel,
    user_profiles: np.ndarray,
    item_emb: np.ndarray,
    eval_users: list[int],
    seen: dict[int, set[int]],
    N: int,
    device: str = "cpu",
    batch_size: int = 256,
) -> dict[int, np.ndarray]:
    """Full-catalogue top-N retrieval using the trained two-tower model."""
    model.eval()

    item_emb_t = torch.as_tensor(item_emb, device=device)
    with torch.no_grad():
        item_proj = model.item_proj(item_emb_t)  # (n_items, proj_dim)

    n_items = item_proj.shape[0]
    K = min(N, n_items)
    item_proj_t = item_proj.t().contiguous()

    eu = np.asarray(eval_users, dtype=np.int64)
    user_prof_t = torch.as_tensor(user_profiles, device=device)
    out: dict[int, np.ndarray] = {}

    with torch.no_grad():
        for start in range(0, len(eu), batch_size):
            bu = eu[start : start + batch_size]
            u_emb = user_prof_t[bu]
            u_proj = model.user_proj(u_emb)
            scores = u_proj @ item_proj_t  # (b, n_items)

            for i, u in enumerate(bu):
                s = seen.get(int(u))
                if s:
                    scores[
                        i, torch.as_tensor(list(s), device=device, dtype=torch.long)
                    ] = float("-inf")

            topi = torch.topk(scores, K, dim=1).indices.cpu().numpy()
            for i, u in enumerate(bu):
                out[int(u)] = topi[i].astype(np.int64)

    return out
