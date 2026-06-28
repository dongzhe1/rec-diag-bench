"""Ranking evaluation metrics and diagnostic tools for recommendation systems."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Dict, List

import numpy as np
import pandas as pd


def dcg_at_k(relevance: List[int], k: int) -> float:
    rel = np.asarray(relevance[:k], dtype=np.float64)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float(np.sum(rel * discounts))


def ndcg_single(ranked_items: List[int], true_item: int, k: int) -> float:
    rel = [1 if item == true_item else 0 for item in ranked_items[:k]]
    ideal = 1.0
    return dcg_at_k(rel, k) / ideal


def recall_single(ranked_items: List[int], true_item: int, k: int) -> float:
    return 1.0 if true_item in ranked_items[:k] else 0.0


def mrr_single(ranked_items: List[int], true_item: int, k: int) -> float:
    for idx, item in enumerate(ranked_items[:k], start=1):
        if item == true_item:
            return 1.0 / idx
    return 0.0


def gini(values: Iterable[float]) -> float:
    """Compute the Gini coefficient for a distribution of values.

    Uses the sorted-values formula with index weighting:
        G = sum((2i - n - 1) * sorted_v) / (n * sum(v))
    Negative values are shifted to non-negative; a small epsilon prevents
    division by zero when the sum is near zero.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if np.amin(arr) < 0:
        arr -= np.amin(arr)
    arr += 1e-12
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))


def limit_candidates_per_user(
    candidates: pd.DataFrame, top_candidates: int | None
) -> pd.DataFrame:
    """Keep a nested per-user candidate budget before scoring.

    The expected workflow is to create one large candidate pool (for example,
    top500) and then evaluate smaller reranking budgets by keeping the same
    positive rows plus the first N negative rows from that pool. This preserves
    the split while changing the number of distractors each model scores.
    """
    if top_candidates is None:
        return candidates
    if top_candidates <= 0:
        raise ValueError(f"top_candidates must be positive, got {top_candidates}")
    if "label" not in candidates.columns:
        return (
            candidates.groupby("user_idx", sort=False)
            .head(top_candidates)
            .reset_index(drop=True)
        )

    parts = []
    for _, group in candidates.groupby("user_idx", sort=False):
        positives = group[group["label"].astype(int) > 0]
        if len(positives) >= top_candidates:
            parts.append(positives.head(top_candidates))
            continue
        selected = group.head(top_candidates).copy()
        missing_pos = positives[~positives.index.isin(selected.index)]
        if len(missing_pos):
            drop_idx = (
                selected[selected["label"].astype(int) <= 0]
                .tail(len(missing_pos))
                .index
            )
            selected = pd.concat(
                [selected.drop(index=drop_idx), missing_pos], ignore_index=False
            )
        parts.append(selected)
    if not parts:
        return candidates.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def _ranked_by_user(
    scored: pd.DataFrame, top_candidates: int | None, seed: int = 42
) -> Dict[int, List[int]]:
    """Rank each user's candidates by score, breaking ties with a seeded random
    key so equal scores don't systematically favour the same items.
    """
    rng = np.random.default_rng(seed)
    scored = scored.assign(_tie=rng.random(len(scored)))
    ranked_by_user: Dict[int, List[int]] = {}
    for user, group in scored.sort_values(
        ["user_idx", "score", "_tie"], ascending=[True, False, False]
    ).groupby("user_idx"):
        ranked_by_user[int(user)] = group.item_idx.astype(int).tolist()[:top_candidates]
    return ranked_by_user


def _per_user_metrics(
    scored: pd.DataFrame,
    truth: Dict[int, int],
    k_list: List[int],
    top_candidates: int | None,
    item_tail_map: dict | None = None,
):
    """Compute per-user ranking metrics plus exposure counters.

    Returns (per_user_df, exposure_counter, tail_exposure_counter). The
    per_user_df has one row per evaluated user with recall@k/ndcg@k/mrr@k
    columns, used directly by evaluate_rankings and resampled by the bootstrap.
    """
    ranked_by_user = _ranked_by_user(scored, top_candidates)
    rows = []
    exposure_counter: Dict[int, int] = defaultdict(int)
    tail_exposure_counter: Dict[int, int] = defaultdict(int)
    max_k = max(k_list)

    for user, true_item in truth.items():
        ranked = ranked_by_user.get(int(user), [])
        row = {"user_idx": int(user), "true_item": int(true_item)}
        for item in ranked[:max_k]:
            exposure_counter[int(item)] += 1
            if item_tail_map is not None and item_tail_map.get(int(item), False):
                tail_exposure_counter[int(item)] += 1
        for k in k_list:
            row[f"recall@{k}"] = recall_single(ranked, true_item, k)
            row[f"ndcg@{k}"] = ndcg_single(ranked, true_item, k)
            row[f"mrr@{k}"] = mrr_single(ranked, true_item, k)
        rows.append(row)

    return pd.DataFrame(rows), exposure_counter, tail_exposure_counter


def evaluate_rankings(
    scored: pd.DataFrame,
    test_df: pd.DataFrame,
    k_list: List[int],
    item_tail_flags: pd.Series | None = None,
    top_candidates: int | None = None,
) -> Dict[str, float]:
    """Evaluate candidate rankings.

    scored columns: user_idx, item_idx, score.
    test_df columns: user_idx, item_idx, plus optional scenario flags.
    """
    truth = dict(zip(test_df.user_idx.astype(int), test_df.item_idx.astype(int)))
    item_tail_map = item_tail_flags.to_dict() if item_tail_flags is not None else None

    per_user, exposure_counter, tail_exposure_counter = _per_user_metrics(
        scored, truth, k_list, top_candidates, item_tail_map
    )

    out: Dict[str, float] = {}
    for col in per_user.columns:
        if "@" in col:
            out[col] = float(per_user[col].mean())

    out["unique_exposed_items"] = float(len(exposure_counter))
    out["gini_exposure"] = gini(exposure_counter.values()) if exposure_counter else 0.0
    if item_tail_map is not None:
        out["unique_tail_exposed_items"] = float(len(tail_exposure_counter))
    return out


def subgroup_metrics(
    scored: pd.DataFrame,
    test_df: pd.DataFrame,
    k_list: List[int],
    top_candidates: int | None = None,
) -> pd.DataFrame:
    scenario_cols = [
        "is_user_cold",
        "is_item_cold",
        "is_long_tail",
        "is_warm",
    ]
    rows = []
    for col in scenario_cols:
        if col not in test_df.columns:
            continue
        sub = test_df[test_df[col].astype(bool)]
        if len(sub) == 0:
            continue
        metrics = evaluate_rankings(
            scored[scored.user_idx.isin(sub.user_idx)],
            sub,
            k_list,
            top_candidates=top_candidates,
        )
        metrics["scenario"] = col
        metrics["num_users"] = len(sub)
        rows.append(metrics)
    return pd.DataFrame(rows)


def bootstrap_ranking_ci(
    scored: pd.DataFrame,
    test_df: pd.DataFrame,
    k_list: List[int],
    top_candidates: int | None = None,
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Bootstrap confidence intervals for the per-user ranking metrics.

    Resamples the evaluated *users* with replacement ``n_boot`` times and
    recomputes the mean of each recall@k / ndcg@k / mrr@k. Returns the point
    estimate plus the lower/upper percentile bounds so small cross-model gaps
    can be read against sampling noise. Exposure/Gini are population-coupled and
    not bootstrapped here.
    """
    truth = dict(zip(test_df.user_idx.astype(int), test_df.item_idx.astype(int)))
    per_user, _, _ = _per_user_metrics(scored, truth, k_list, top_candidates)
    metric_cols = [c for c in per_user.columns if "@" in c]

    out: Dict[str, float] = {"n_users": len(per_user)}
    if len(per_user) == 0:
        return out

    vals = per_user[metric_cols].to_numpy(dtype=np.float64)  # [N, M]
    n = vals.shape[0]
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, vals.shape[1]), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = vals[idx].mean(axis=0)

    lo_q = (1.0 - ci) / 2.0 * 100.0
    hi_q = (1.0 + ci) / 2.0 * 100.0
    point = vals.mean(axis=0)
    lo = np.percentile(boots, lo_q, axis=0)
    hi = np.percentile(boots, hi_q, axis=0)
    for i, col in enumerate(metric_cols):
        out[col] = float(point[i])
        out[f"{col}_ci_lo"] = float(lo[i])
        out[f"{col}_ci_hi"] = float(hi[i])
    return out


def oracle_recall(
    candidates: pd.DataFrame,
    test_df: pd.DataFrame,
    k_list: List[int],
) -> Dict[str, float]:
    """Measure retrieval coverage: how often is the true item in the candidate pool?

    This directly answers [11]'s central claim that "it's all retrieval."
    Our positive-controlled candidate pool guarantees 100% oracle recall by
    construction, which lets us isolate reranker quality. But when called on a
    real-retriever pool (no positive guarantee) the number can be well below 1.

    For each k in k_list, reports the fraction of eval users whose true item
    appears within the top-k candidates (ordered by the pool's natural order, i.e.
    the retriever's own ranking). Also reports pool_coverage (fraction who have
    their true item *anywhere* in the pool regardless of k) and
    median_true_rank (median rank of the true item among users who have it,
    1-indexed; NaN for users where it is absent).

    Columns expected in candidates: user_idx, item_idx. An optional score column
    is used to order candidates per user if present; otherwise pool order is used.
    test_df must have user_idx and item_idx (the positive item).
    """
    truth = dict(zip(test_df.user_idx.astype(int), test_df.item_idx.astype(int)))
    out: Dict[str, float] = {"n_eval_users": len(truth)}

    # Order candidates per user: by score descending if available, else by row order.
    if "score" in candidates.columns:
        ordered = candidates.sort_values(["user_idx", "score"], ascending=[True, False])
    else:
        ordered = candidates.sort_values("user_idx")

    # Build per-user ranked item list.
    ranked_pool: Dict[int, List[int]] = {}
    for user, grp in ordered.groupby("user_idx"):
        ranked_pool[int(user)] = grp["item_idx"].astype(int).tolist()

    true_ranks: List[float] = []  # rank of true item (1-indexed); nan if absent
    for user, true_item in truth.items():
        pool = ranked_pool.get(int(user), [])
        if true_item in pool:
            true_ranks.append(float(pool.index(true_item) + 1))
        else:
            true_ranks.append(float("nan"))

    true_ranks_arr = np.array(true_ranks, dtype=np.float64)
    present = ~np.isnan(true_ranks_arr)

    out["pool_coverage"] = float(present.mean())
    out["median_true_rank"] = (
        float(np.nanmedian(true_ranks_arr)) if present.any() else float("nan")
    )

    for k in k_list:
        # oracle_recall@k = fraction of eval users with true item in top-k of pool
        out[f"oracle_recall@{k}"] = float(np.mean(true_ranks_arr <= k))

    return out


def score_separation(
    scored: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, float]:
    """Measure how well a model's scores separate relevant from irrelevant items.

    A model that cannot discriminate relevant from irrelevant candidates will
    show near-zero mean difference, near-zero Cohen's d, and near-zero Spearman
    correlation. [11] found Cohen's d = 0.11 and Spearman r = 0.004 for a
    cross-encoder — essentially random. This function lets us report the same
    numbers for every model, enabling direct comparison.

    Annotates each row in scored as relevant (true item for that user) or
    irrelevant, then computes:
    - mean_relevant, mean_irrelevant, mean_diff
    - std_relevant, std_irrelevant
    - cohens_d  (pooled-std standardised mean difference)
    - spearman_r  (rank correlation between score and binary relevance)
    - overlap_coef  (Weitzman overlap coefficient: integral of min(f,g))
    - t_stat, p_value  (Welch's t-test, does not assume equal variance)
    - n_relevant, n_irrelevant
    """
    from scipy import stats as _stats

    truth = dict(zip(test_df.user_idx.astype(int), test_df.item_idx.astype(int)))
    df = scored[["user_idx", "item_idx", "score"]].copy()
    df["user_idx"] = df["user_idx"].astype(int)
    df["item_idx"] = df["item_idx"].astype(int)
    df["relevant"] = df.apply(
        lambda r: int(truth.get(r["user_idx"], -1) == r["item_idx"]), axis=1
    )

    rel = df.loc[df["relevant"] == 1, "score"].to_numpy(dtype=np.float64)
    irr = df.loc[df["relevant"] == 0, "score"].to_numpy(dtype=np.float64)

    out: Dict[str, float] = {
        "n_relevant": len(rel),
        "n_irrelevant": len(irr),
    }

    if len(rel) == 0 or len(irr) == 0:
        return out

    mean_rel = float(np.mean(rel))
    mean_irr = float(np.mean(irr))
    std_rel = float(np.std(rel, ddof=1)) if len(rel) > 1 else 0.0
    std_irr = float(np.std(irr, ddof=1)) if len(irr) > 1 else 0.0
    n_rel, n_irr = len(rel), len(irr)

    pooled_std = (
        np.sqrt(
            ((n_rel - 1) * std_rel**2 + (n_irr - 1) * std_irr**2) / (n_rel + n_irr - 2)
        )
        if (n_rel + n_irr) > 2
        else 0.0
    )

    cohens_d = (mean_rel - mean_irr) / pooled_std if pooled_std > 0 else 0.0

    # Spearman rank correlation between score and binary relevance label.
    # Returns NaN when scores are constant (e.g. Markov all-zero output); treat as 0.
    sp_r, sp_p = _stats.spearmanr(df["score"].to_numpy(), df["relevant"].to_numpy())
    if np.isnan(sp_r):
        sp_r, sp_p = 0.0, 1.0

    # Welch t-test (unequal variance).
    t_stat, p_val = _stats.ttest_ind(rel, irr, equal_var=False)
    if np.isnan(t_stat):
        t_stat, p_val = 0.0, 1.0

    # Overlap coefficient via histogram approximation (100 bins over joint range).
    joint_min = min(rel.min(), irr.min())
    joint_max = max(rel.max(), irr.max())
    if joint_max > joint_min:
        bins = np.linspace(joint_min, joint_max, 101)
        h_rel, _ = np.histogram(rel, bins=bins, density=True)
        h_irr, _ = np.histogram(irr, bins=bins, density=True)
        bin_width = bins[1] - bins[0]
        overlap = float(np.sum(np.minimum(h_rel, h_irr)) * bin_width)
    else:
        overlap = 1.0

    out.update(
        {
            "mean_relevant": round(mean_rel, 6),
            "mean_irrelevant": round(mean_irr, 6),
            "mean_diff": round(mean_rel - mean_irr, 6),
            "std_relevant": round(std_rel, 6),
            "std_irrelevant": round(std_irr, 6),
            "cohens_d": round(float(cohens_d), 4),
            "spearman_r": round(float(sp_r), 4),
            "spearman_p": round(float(sp_p), 6),
            "t_stat": round(float(t_stat), 4),
            "p_value": round(float(p_val), 6),
            "overlap_coef": round(float(overlap), 4),
        }
    )
    return out


def tie_diagnostic(
    scored: pd.DataFrame,
    test_df: pd.DataFrame,
    k_list: List[int],
    top_candidates: int | None = None,
) -> Dict[str, float]:
    """Quantify how much score ties affect the headline metrics.

    The ranker breaks ties randomly, so when a model emits many equal scores
    (e.g. integer LLM outputs) part of the ranking is arbitrary. For each
    evaluated user this computes the true item's best-case rank (tied items
    placed *after* it) and worst-case rank (tied items placed *before* it), then
    reports recall@k / ndcg@k under both orderings. A large best-vs-worst gap
    means the random-tie-break headline number is unreliable for this model.

    Returns a flat dict with ``tie_rate`` (fraction of users whose true item
    shares its score with at least one other candidate) and, per k,
    ``recall@k_best`` / ``recall@k_worst`` / ``recall@k_gap`` and the ndcg
    equivalents.
    """
    truth = test_df[["user_idx", "item_idx"]].rename(columns={"item_idx": "true_item"})
    df = scored.merge(truth, on="user_idx", how="inner")
    out: Dict[str, float] = {"n_users": 0, "tie_rate": 0.0}
    if df.empty:
        return out

    # True item's score per user (skip users whose true item isn't scored).
    true_scores = df[df["item_idx"] == df["true_item"]][["user_idx", "score"]].rename(
        columns={"score": "true_score"}
    )
    df = df.merge(true_scores, on="user_idx", how="inner")

    df["gt"] = (df["score"] > df["true_score"]).astype(int)  # strictly better
    df["ge"] = (df["score"] >= df["true_score"]).astype(
        int
    )  # better-or-equal (incl. self)
    agg = df.groupby("user_idx").agg(n_gt=("gt", "sum"), n_ge=("ge", "sum"))
    agg["rank_best"] = agg["n_gt"] + 1  # tied items ranked after the true item
    agg["rank_worst"] = agg["n_ge"]  # tied items ranked before the true item
    # n_ge - n_gt counts items with score == true_score, including the true item
    # itself; > 1 means at least one *other* candidate ties the true item.
    agg["true_tie"] = (agg["n_ge"] - agg["n_gt"]) > 1

    n = len(agg)
    out["n_users"] = int(n)
    out["tie_rate"] = float(agg["true_tie"].mean())

    rb = agg["rank_best"].to_numpy()
    rw = agg["rank_worst"].to_numpy()
    for k in k_list:
        recall_best = float(np.mean(rb <= k))
        recall_worst = float(np.mean(rw <= k))
        out[f"recall@{k}_best"] = recall_best
        out[f"recall@{k}_worst"] = recall_worst
        out[f"recall@{k}_gap"] = recall_best - recall_worst
        with np.errstate(divide="ignore", invalid="ignore"):
            ndcg_best = float(np.mean(np.where(rb <= k, 1.0 / np.log2(rb + 1), 0.0)))
            ndcg_worst = float(np.mean(np.where(rw <= k, 1.0 / np.log2(rw + 1), 0.0)))
        out[f"ndcg@{k}_best"] = ndcg_best
        out[f"ndcg@{k}_worst"] = ndcg_worst
        out[f"ndcg@{k}_gap"] = ndcg_best - ndcg_worst
    return out
