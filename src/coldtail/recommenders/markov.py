"""First-order Markov chain recommender for sequential recommendation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .base import infer_shape


def build_markov(train: pd.DataFrame, shape):
    """Build the row-normalised first-order transition matrix and each user's
    last item. Split out from :func:`score_markov` for reuse (e.g. full-catalogue
    retrieval).
    """
    df = train.sort_values(["user_idx", "timestamp"])

    user_arr = df["user_idx"].to_numpy()
    item_arr = df["item_idx"].astype(int).to_numpy()

    # Build same_user without np.roll — no extra array allocation
    same_user = np.empty(len(user_arr), dtype=bool)
    same_user[:-1] = user_arr[:-1] == user_arr[1:]
    same_user[-1] = False

    rows = item_arr[:-1][same_user[:-1]]
    cols = item_arr[1:][same_user[:-1]]

    trans = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(shape.n_items, shape.n_items),
    )
    row_sum = np.asarray(trans.sum(axis=1)).ravel() + 1e-8
    trans = sparse.diags(1.0 / row_sum) @ trans

    last_item = (
        train.sort_values("timestamp")
        .groupby("user_idx")
        .tail(1)
        .set_index("user_idx")["item_idx"]
        .astype(int)
        .to_dict()
    )
    return trans.tocsr(), last_item


def score_markov(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    items: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score using first-order Markov chain transitions."""
    shape = infer_shape(train, candidates, items)
    trans, last_item = build_markov(train, shape)

    last_items = candidates["user_idx"].map(last_item)
    valid = last_items.notna()
    scores = np.zeros(len(candidates), dtype=float)
    if valid.any():
        li_arr = last_items[valid].astype(int).to_numpy()
        it_arr = candidates.loc[valid, "item_idx"].astype(int).to_numpy()
        scores[valid.to_numpy()] = np.asarray(trans[li_arr, it_arr]).ravel()
    return candidates[["user_idx", "item_idx"]].assign(score=scores)
