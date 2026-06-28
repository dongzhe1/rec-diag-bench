"""Dense semantic retrieval baseline using sentence-transformers (SBERT/BGE)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from .base import infer_shape

logger = logging.getLogger(__name__)


def build_sbert_embeddings(
    train: pd.DataFrame,
    items: pd.DataFrame,
    shape=None,
    model_name: str = "BAAI/bge-base-en-v1.5",
    max_hist: int = 10,
    device: str = "cpu",
    batch_size: int = 256,
    doc_prefix: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Build L2-normalised ``(user_profiles, item_emb)`` SBERT embeddings.

    Split out from :func:`score_sbert` so callers that need the embeddings
    directly (e.g. full-catalogue retrieval — the interaction-free content
    signal that can reach cold/new items) reuse the same construction. Item
    embeddings depend only on text, so a cold item with no interactions is still
    retrievable by its own semantics.
    """
    from sentence_transformers import SentenceTransformer

    if shape is None:
        shape = infer_shape(train, None, items)

    logger.info(f"[SBERT] loading model: {model_name}")
    st_model = SentenceTransformer(model_name, device=device)

    texts = (
        items.set_index("item_idx")["text"]
        .reindex(range(shape.n_items))
        .fillna("")
        .astype(str)
        .tolist()
    )

    if doc_prefix:
        # Some encoders (e.g. E5) expect an instruction prefix on the encoded passage.
        texts = [doc_prefix + t for t in texts]

    logger.info(f"[SBERT] encoding {len(texts):,} item texts...")
    item_emb = st_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    if max_hist is not None:
        train = (
            train.sort_values("timestamp")
            .groupby("user_idx", sort=False)
            .tail(max_hist)
            .reset_index(drop=True)
        )

    logger.info("[SBERT] building user profiles...")
    user_profiles = np.zeros((shape.n_users, item_emb.shape[1]), dtype=np.float32)
    user_counts = np.zeros(shape.n_users, dtype=np.int32)
    for _, row in train.iterrows():
        u, i = int(row["user_idx"]), int(row["item_idx"])
        if i < len(item_emb):
            user_profiles[u] += item_emb[i]
            user_counts[u] += 1

    mask = user_counts > 0
    user_profiles[mask] /= user_counts[mask, None]
    user_profiles = normalize(user_profiles, norm="l2", axis=1).astype(np.float32)

    del st_model
    return user_profiles, item_emb


def score_sbert(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    items: pd.DataFrame,
    model_name: str = "BAAI/bge-base-en-v1.5",
    max_hist: int = 10,
    device: str = "cpu",
    batch_size: int = 256,
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """Score candidates by cosine similarity of user profile vs item embeddings."""
    shape = infer_shape(train, candidates, items)
    user_profiles, item_emb = build_sbert_embeddings(
        train, items, shape, model_name, max_hist, device, batch_size
    )

    logger.info(f"[SBERT] scoring {len(candidates):,} candidate pairs...")
    u_idx = candidates["user_idx"].astype(int).to_numpy()
    i_idx = candidates["item_idx"].astype(int).to_numpy()
    scores = np.empty(len(candidates), dtype=np.float32)

    for start in range(0, len(candidates), chunk_size):
        end = min(start + chunk_size, len(candidates))
        u = u_idx[start:end]
        i = i_idx[start:end]
        scores[start:end] = (user_profiles[u] * item_emb[i]).sum(axis=1)

    del item_emb, user_profiles

    return candidates[["user_idx", "item_idx"]].assign(score=scores.astype(float))
