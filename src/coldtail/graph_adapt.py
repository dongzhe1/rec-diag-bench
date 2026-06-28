"""Fuse semantic, graph-based, and popularity signals into recommendation scores."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def normalize_per_user(
    df: pd.DataFrame, score_col: str = "score", out_col: str = "score_norm"
) -> pd.DataFrame:
    out = df.copy()

    def norm(s: pd.Series) -> pd.Series:
        lo = s.min()
        hi = s.max()
        if hi <= lo:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - lo) / (hi - lo)

    out[out_col] = out.groupby("user_idx")[score_col].transform(norm)
    return out


def load_score_or_zero(
    path: str | Path, candidates: pd.DataFrame, name: str
) -> pd.DataFrame:
    path = Path(path)
    base = candidates[["user_idx", "item_idx"]].copy()
    if path.exists():
        df = pd.read_csv(path)[["user_idx", "item_idx", "score"]]
        df = normalize_per_user(df, "score", name)
        return base.merge(
            df[["user_idx", "item_idx", name]], on=["user_idx", "item_idx"], how="left"
        ).fillna({name: 0.0})
    base[name] = 0.0
    return base


def graph_aware_fusion(
    candidates: pd.DataFrame,
    scores_dir: str | Path,
    item_stats_path: str | Path,
    alpha_semantic: float = 0.35,
    beta_graph: float = 0.45,
    gamma_base: float = 0.30,
    lambda_popularity: float = 0.10,
    tail_boost: float = 0.05,
    base_model_for_graph: str = "lightgcn",
) -> pd.DataFrame:
    """Fuse multiple scoring signals into a single recommendation score.

    Signal layout
    -------------
    semantic   : TF-IDF content similarity (alpha_semantic)
    graph      : Graph-based collaborative signal from `base_model_for_graph` (beta_graph)
    base       : Base collaborative filtering scores (gamma_base)
    popularity : Global item popularity, subtracted to reduce bias (lambda_popularity)
    tail_bonus : Flat additive boost for long-tail items (tail_boost)

    Note on weights
    ---------------
    alpha + beta + gamma - lambda = 0.35 + 0.45 + 0.30 - 0.10 = 1.00
    The tail_bonus is an additive offset on top of the [0,1]-normalised sum,
    so the final score can slightly exceed 1.0 for tail items.
    """
    scores_dir = Path(scores_dir)
    out = candidates[["user_idx", "item_idx"]].copy()

    # semantic: content-based TF-IDF signal
    semantic = load_score_or_zero(
        scores_dir / "tfidf_scores.csv", candidates, "semantic"
    )
    # graph: collaborative signal from the designated graph model
    base_model_for_base: str = "bpr"
    graph = load_score_or_zero(
        scores_dir / f"{base_model_for_graph}_scores.csv", candidates, "graph"
    )
    base = load_score_or_zero(
        scores_dir / f"{base_model_for_base}_scores.csv", candidates, "base"
    )
    pop = load_score_or_zero(
        scores_dir / "popularity_scores.csv", candidates, "popularity"
    )

    out = out.merge(semantic, on=["user_idx", "item_idx"], how="left")
    out = out.merge(graph, on=["user_idx", "item_idx"], how="left")
    out = out.merge(base, on=["user_idx", "item_idx"], how="left")
    out = out.merge(pop, on=["user_idx", "item_idx"], how="left")
    for col in ["semantic", "graph", "base", "popularity"]:
        out[col] = out[col].fillna(0.0)

    stats = pd.read_csv(item_stats_path)
    stats = stats[["item_idx", "is_tail"]]
    out = out.merge(stats, on="item_idx", how="left")
    out["is_tail"] = out["is_tail"].fillna(False).astype(bool)
    out["tail_bonus"] = out["is_tail"].astype(float) * tail_boost

    out["score"] = (
        alpha_semantic * out["semantic"]
        + beta_graph * out["graph"]
        + gamma_base * out["base"]
        - lambda_popularity * out["popularity"]
        + out["tail_bonus"]
    )
    return out[
        [
            "user_idx",
            "item_idx",
            "score",
            "semantic",
            "graph",
            "base",
            "popularity",
            "tail_bonus",
        ]
    ]
