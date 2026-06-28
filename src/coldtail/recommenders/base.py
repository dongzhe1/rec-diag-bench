"""Shared utilities for recommender baselines: shapes, sparse matrices, and BPR sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import numba as nb

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


# Dataset shape


@dataclass
class DatasetShape:
    n_users: int
    n_items: int


def infer_shape(
    train: pd.DataFrame,
    candidates: pd.DataFrame | None = None,
    items: pd.DataFrame | None = None,
) -> DatasetShape:
    n_users = int(train.user_idx.max()) + 1
    n_items = int(train.item_idx.max()) + 1
    if candidates is not None and len(candidates) > 0:
        n_users = max(n_users, int(candidates.user_idx.max()) + 1)
        n_items = max(n_items, int(candidates.item_idx.max()) + 1)
    if items is not None and len(items) > 0 and "item_idx" in items.columns:
        n_items = max(n_items, int(items.item_idx.max()) + 1)
    return DatasetShape(n_users=n_users, n_items=n_items)


# Sparse UI matrix helpers


def build_user_item_matrix(
    train: pd.DataFrame,
    shape: DatasetShape | None = None,
) -> sparse.csr_matrix:
    if shape is None:
        shape = infer_shape(train)
    data = np.ones(len(train), dtype=np.float32)
    return sparse.csr_matrix(
        (data, (train.user_idx.astype(int), train.item_idx.astype(int))),
        shape=(shape.n_users, shape.n_items),
    )


def score_candidates_from_matrix(
    score_matrix: sparse.csr_matrix | np.ndarray,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    users = candidates.user_idx.astype(int).to_numpy()
    items = candidates.item_idx.astype(int).to_numpy()
    if sparse.issparse(score_matrix):
        scores = np.asarray(score_matrix[users, items]).ravel()
    else:
        scores = score_matrix[users, items]
    return candidates[["user_idx", "item_idx"]].assign(score=scores.astype(float))


def save_scores(scored: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)


# BPR / LightGCN training helpers


def user_pos_sets(train: pd.DataFrame) -> Dict[int, Set[int]]:
    """Build a dict mapping user_idx -> set of interacted item_idx."""
    return (
        train.groupby("user_idx")["item_idx"]
        .apply(lambda s: set(map(int, s)))
        .to_dict()
    )


def _build_pos_arrays(
    user_pos: Dict[int, Set[int]],
    users_arr: np.ndarray,
) -> List[np.ndarray]:
    """Pre-compute per-user positive-item arrays (used by the fallback sampler)."""
    return [np.asarray(list(user_pos[int(u)]), dtype=np.int64) for u in users_arr]


def build_pos_csr(
    user_pos: Dict[int, Set[int]],
    users_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert user_pos to CSR arrays indexed by position in users_arr.

    Returns
    -------
    pos_indptr  : (n_train_users + 1,) int64
    pos_indices : (total_pos_interactions,) int64

    """
    arrays = [np.asarray(list(user_pos[int(u)]), dtype=np.int64) for u in users_arr]
    counts = np.array([len(a) for a in arrays], dtype=np.int64)
    pos_indptr = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum(counts, out=pos_indptr[1:])
    pos_indices = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
    return pos_indptr, pos_indices


# Numba unified pos+neg sampler (compiled once, cached on disk)

if _NUMBA_AVAILABLE:

    @nb.njit(parallel=True, cache=True)
    def _sample_pos_neg_numba(
        user_indices: np.ndarray,
        pos_indptr: np.ndarray,
        pos_indices: np.ndarray,
        n_items: int,
        max_tries: int,
        seeds: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parallel positive and negative sampling in one numba kernel."""
        pos = np.empty(len(user_indices), dtype=np.int64)
        neg = np.empty(len(user_indices), dtype=np.int64)

        for i in nb.prange(len(user_indices)):
            ui = user_indices[i]
            start = pos_indptr[ui]
            end = pos_indptr[ui + 1]
            items = pos_indices[start:end]
            n_pos = end - start

            state = seeds[i]

            # Sample positive: one LCG step
            state = nb.uint64(6364136223846793005) * state + nb.uint64(
                1442695040888963407
            )
            pos[i] = items[nb.int64(state >> nb.uint64(33)) % n_pos]

            # Sample negative: rejection loop
            sampled_neg = nb.int64(0)
            for _ in range(max_tries):
                state = nb.uint64(6364136223846793005) * state + nb.uint64(
                    1442695040888963407
                )
                candidate = nb.int64(state >> nb.uint64(33)) % n_items
                hit = False
                for p in items:
                    if p == candidate:
                        hit = True
                        break
                if not hit:
                    sampled_neg = candidate
                    break
            else:
                sampled_neg = candidate

            neg[i] = sampled_neg

        return pos, neg

    # Keep the legacy neg-only kernel for any external callers that depend on it.
    @nb.njit(parallel=True, cache=True)
    def _sample_negatives_numba(
        user_indices: np.ndarray,
        pos_indptr: np.ndarray,
        pos_indices: np.ndarray,
        n_items: int,
        max_tries: int,
        seeds: np.ndarray,
    ) -> np.ndarray:
        """Neg-only sampler retained for backward compatibility."""
        neg = np.empty(len(user_indices), dtype=np.int64)
        for i in nb.prange(len(user_indices)):
            ui = user_indices[i]
            start = pos_indptr[ui]
            end = pos_indptr[ui + 1]
            items = pos_indices[start:end]
            state = seeds[i]
            for _ in range(max_tries):
                state = nb.uint64(6364136223846793005) * state + nb.uint64(
                    1442695040888963407
                )
                n = nb.int64(state >> nb.uint64(33)) % n_items
                hit = False
                for p in items:
                    if p == n:
                        hit = True
                        break
                if not hit:
                    neg[i] = n
                    break
            else:
                neg[i] = n
        return neg


# Public sample_batch — dispatches to unified numba kernel or numpy fallback


def sample_batch(
    rng: np.random.Generator,
    users_arr: np.ndarray,
    user_pos: Dict[int, Set[int]],
    n_items: int,
    batch_size: int,
    max_neg_tries: int = 30,
    # Pre-built structures — pass once outside the training loop.
    _pos_arrays: List[np.ndarray] | None = None,  # fallback path only
    _pos_indptr: np.ndarray | None = None,  # numba path
    _pos_indices: np.ndarray | None = None,  # numba path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample (user, pos_item, neg_item) triples for BPR / LightGCN training.

    Dispatches to a parallel numba kernel when pre-built CSR structures are
    available; falls back to a Python + numpy loop otherwise.  Call
    ``build_pos_csr()`` once before the training loop to enable the fast path.
    """
    # sample users (always vectorised)
    user_indices = rng.integers(0, len(users_arr), size=batch_size)
    users = users_arr[user_indices]

    if _NUMBA_AVAILABLE and _pos_indptr is not None and _pos_indices is not None:
        seeds = rng.integers(
            1, np.iinfo(np.uint64).max, size=batch_size, dtype=np.uint64
        )
        pos, neg = _sample_pos_neg_numba(
            user_indices.astype(np.int64),
            _pos_indptr,
            _pos_indices,
            n_items,
            max_neg_tries,
            seeds,
        )
    else:
        # Python loop over ragged list-of-arrays; cannot be vectorized without numba
        if _pos_arrays is None:
            _pos_arrays = _build_pos_arrays(user_pos, users_arr)
        pos = np.empty(batch_size, dtype=np.int64)
        for i, ui in enumerate(user_indices):
            pa = _pos_arrays[ui]
            pos[i] = pa[rng.integers(0, len(pa))]

        # negative: vectorised redraw
        neg = rng.integers(0, n_items, size=batch_size)
        for _ in range(max_neg_tries):
            bad = np.zeros(batch_size, dtype=bool)
            for i, (u, n) in enumerate(zip(users, neg)):
                if n in user_pos[int(u)]:
                    bad[i] = True
            if not bad.any():
                break
            neg[bad] = rng.integers(0, n_items, size=int(bad.sum()))

    return users.astype(np.int64), pos, neg
