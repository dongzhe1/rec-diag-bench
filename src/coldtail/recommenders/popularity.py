"""Popularity-based recommender: scores each item by its training-set interaction count."""

from __future__ import annotations

import pandas as pd

from .base import infer_shape


def score_popularity(
    train: pd.DataFrame, candidates: pd.DataFrame, items: pd.DataFrame | None = None
) -> pd.DataFrame:
    shape = infer_shape(train, candidates, items)
    counts = (
        train.groupby("item_idx")
        .size()
        .reindex(range(shape.n_items), fill_value=0)
        .astype(float)
        .to_numpy()
    )
    scores = counts[candidates.item_idx.astype(int).to_numpy()]
    return candidates[["user_idx", "item_idx"]].assign(score=scores)
