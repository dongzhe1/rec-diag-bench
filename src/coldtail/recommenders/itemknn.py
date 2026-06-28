"""Item-based k-nearest-neighbour recommender using cosine similarity of interaction vectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .base import build_user_item_matrix, infer_shape, score_candidates_from_matrix


def fit_itemknn_sim(
    train: pd.DataFrame,
    shape,
    topk_sim: int = 100,
    block_size: int = 512,
) -> sparse.csr_matrix:
    """Build the top-k cosine item-item similarity matrix.

    Split out from :func:`fit_itemknn_scores` so callers that need the raw
    similarity (e.g. full-catalog retrieval or graph-expansion neighbours) can
    reuse the exact same blocked computation.
    """
    ui = build_user_item_matrix(train, shape).astype(np.float32)

    item_norms = np.sqrt(ui.power(2).sum(axis=0)).A1 + 1e-8
    norm_ui = ui @ sparse.diags(1.0 / item_norms)

    n_items = shape.n_items
    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    vals_list: list[np.ndarray] = []

    for start in range(0, n_items, block_size):
        end = min(start + block_size, n_items)
        block_len = end - start

        block_sim: np.ndarray = (norm_ui.T[start:end] @ norm_ui).toarray()

        local_idx = np.arange(block_len)
        global_idx = local_idx + start
        valid = global_idx < n_items
        block_sim[local_idx[valid], global_idx[valid]] = 0.0

        k = min(topk_sim, n_items - 1)
        if k < block_sim.shape[1]:
            kth_vals = np.partition(block_sim, -k, axis=1)[:, -k, np.newaxis]
            mask = (block_sim >= kth_vals) & (block_sim > 0)
        else:
            mask = block_sim > 0

        local_i_idx, col_idx = np.where(mask)
        if len(local_i_idx) == 0:
            continue

        rows_list.append((local_i_idx + start).astype(np.int32))
        cols_list.append(col_idx.astype(np.int32))
        vals_list.append(block_sim[local_i_idx, col_idx].astype(np.float32))

    if rows_list:
        sim = sparse.csr_matrix(
            (
                np.concatenate(vals_list),
                (np.concatenate(rows_list), np.concatenate(cols_list)),
            ),
            shape=(n_items, n_items),
        )
    else:
        sim = sparse.csr_matrix((n_items, n_items), dtype=np.float32)

    return sim


def fit_itemknn_scores(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    topk_sim: int = 100,
    items: pd.DataFrame | None = None,
    block_size: int = 512,
) -> pd.DataFrame:
    shape = infer_shape(train, candidates, items)
    ui = build_user_item_matrix(train, shape).astype(np.float32)
    sim = fit_itemknn_sim(train, shape, topk_sim, block_size)
    user_scores = ui @ sim
    return score_candidates_from_matrix(user_scores.tocsr(), candidates)
