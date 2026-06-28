"""GALA evidence builder: per-candidate graph signals for LLM prompt injection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _build_cooccur_index(
    train: pd.DataFrame, min_cooccur: int = 2
) -> dict[int, set[int]]:
    """Build symmetric item co-occurrence index from training data."""
    from collections import defaultdict
    from itertools import combinations

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)

    for _, grp in train.groupby("user_idx")["item_idx"]:
        items = grp.astype(int).tolist()[-50:]
        for a, b in combinations(items, 2):
            pair_counts[(a, b)] += 1

    cooccur: dict[int, set[int]] = defaultdict(set)
    for (a, b), cnt in pair_counts.items():
        if cnt >= min_cooccur:
            cooccur[a].add(b)
            cooccur[b].add(a)
    return dict(cooccur)


def _percentile_rank(series: pd.Series) -> pd.Series:
    """Per-element percentile rank (0-100)."""
    return series.rank(pct=True, method="average") * 100


def _add_pct_per_user(df: pd.DataFrame, score_col: str, out_col: str) -> pd.DataFrame:
    """Add per-user percentile rank column."""
    df = df.copy()
    df[out_col] = df.groupby("user_idx")[score_col].transform(_percentile_rank)
    return df


def build_graph_evidence(
    candidates: pd.DataFrame,
    train: pd.DataFrame,
    items: pd.DataFrame,
    item_stats: pd.DataFrame,
    scores_dir: Path,
    max_hist: int = 10,
    min_cooccur: int = 2,
) -> pd.DataFrame:
    """Build evidence DataFrame aligned with candidates."""
    scores_dir = Path(scores_dir)
    cand = candidates[["user_idx", "item_idx"]].copy().reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1. Graph score percentile (LightGCN)
    # ------------------------------------------------------------------
    gcn_path = scores_dir / "lightgcn_scores.csv"
    if gcn_path.exists():
        gcn = pd.read_csv(gcn_path)[["user_idx", "item_idx", "score"]].rename(
            columns={"score": "graph_score"}
        )
        cand = cand.merge(gcn, on=["user_idx", "item_idx"], how="left")
        cand["graph_score"] = cand["graph_score"].fillna(0.0)
        cand = _add_pct_per_user(cand, "graph_score", "graph_score_pct")
        cand.drop(columns=["graph_score"], inplace=True)
    else:
        cand["graph_score_pct"] = 50.0  # neutral if scores unavailable

    tfidf_path = scores_dir / "tfidf_scores.csv"
    if tfidf_path.exists():
        tfidf = pd.read_csv(tfidf_path)[["user_idx", "item_idx", "score"]].rename(
            columns={"score": "semantic_score"}
        )
        cand = cand.merge(tfidf, on=["user_idx", "item_idx"], how="left")
        cand["semantic_score"] = cand["semantic_score"].fillna(0.0)
        cand = _add_pct_per_user(cand, "semantic_score", "semantic_score_pct")
        cand.drop(columns=["semantic_score"], inplace=True)
    else:
        cand["semantic_score_pct"] = 50.0

    cooccur = _build_cooccur_index(train, min_cooccur=min_cooccur)
    user_hist: dict[int, list[int]] = (
        train.sort_values("timestamp")
        .groupby("user_idx")["item_idx"]
        .apply(lambda s: s.astype(int).tolist()[-max_hist:])
        .to_dict()
    )

    item_text_map = items.set_index("item_idx")["text"].fillna("").astype(str).to_dict()

    cooccur_strengths = []
    overlap_texts = []
    for row in cand.itertuples(index=False):
        hist = user_hist.get(int(row.user_idx), [])
        neighbors = cooccur.get(int(row.item_idx), set())
        overlapping = [i for i in hist if i in neighbors]
        cooccur_strengths.append(len(overlapping))
        if overlapping:
            names = [item_text_map.get(i, str(i))[:40] for i in overlapping[:3]]
            overlap_texts.append("; ".join(names))
        else:
            overlap_texts.append("")

    cand["cooccur_strength"] = cooccur_strengths
    cand["hist_overlap_text"] = overlap_texts

    stats = item_stats[["item_idx", "train_count", "is_tail"]].copy()
    stats["popularity_pct"] = _percentile_rank(stats["train_count"])
    cand = cand.merge(
        stats[["item_idx", "popularity_pct", "is_tail"]],
        on="item_idx",
        how="left",
    )
    cand["popularity_pct"] = cand["popularity_pct"].fillna(0.0)
    cand["is_tail"] = cand["is_tail"].fillna(False).astype(bool)

    return cand
