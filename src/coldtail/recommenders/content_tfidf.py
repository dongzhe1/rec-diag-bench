"""Content-based filtering recommender using TF-IDF vectors over item text."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .base import build_user_item_matrix, infer_shape


def build_tfidf_vectors(
    train: pd.DataFrame,
    items: pd.DataFrame,
    shape=None,
    max_hist: int = 10,
):
    """Build L2-normalised TF-IDF ``(user_profile, item_vec)`` (both sparse).

    Split out from :func:`score_tfidf` so callers that need the vectors directly
    (e.g. full-catalogue retrieval — a lexical content signal that reaches cold
    items by text) reuse the same construction.
    """
    if shape is None:
        shape = infer_shape(train, None, items)

    # Truncate to recent max_hist interactions per user (matches cross-encoder/LLM history)
    if max_hist is not None:
        train = (
            train.sort_values("timestamp")
            .groupby("user_idx", sort=False)
            .tail(max_hist)
            .reset_index(drop=True)
        )

    text = (
        items.set_index("item_idx")["text"]
        .reindex(range(shape.n_items))
        .fillna("")
        .astype(str)
        .tolist()
    )

    vectorizer = TfidfVectorizer(min_df=1, max_features=50_000, ngram_range=(1, 2))
    item_vec = normalize(
        vectorizer.fit_transform(text).astype(np.float32), norm="l2", axis=1
    )

    ui = build_user_item_matrix(train, shape).astype(np.float32)
    user_profile = normalize(ui @ item_vec, norm="l2", axis=1)
    return user_profile, item_vec


def score_tfidf(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    items: pd.DataFrame,
    chunk_size: int = 50_000,
    max_hist: int = 10,
) -> pd.DataFrame:
    """Score using TF-IDF content-based filtering with chunked processing."""
    shape = infer_shape(train, candidates, items)
    user_profile, item_vec = build_tfidf_vectors(train, items, shape, max_hist)

    u_idx = candidates.user_idx.astype(int).to_numpy()
    i_idx = candidates.item_idx.astype(int).to_numpy()

    scores = np.empty(len(candidates), dtype=np.float32)
    for start in range(0, len(candidates), chunk_size):
        end = min(start + chunk_size, len(candidates))
        u = u_idx[start:end]
        i = i_idx[start:end]
        u_vecs = user_profile[u].toarray()
        i_vecs = item_vec[i].toarray()
        scores[start:end] = (u_vecs * i_vecs).sum(axis=1)

    return candidates[["user_idx", "item_idx"]].assign(score=scores.astype(float))
