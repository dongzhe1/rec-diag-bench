"""Fuse baseline scores with graph-aware signals via weighted combination.

Combines LightGCN, BPR, TF-IDF, and popularity scores with configurable
weights, a palette-cleaning step, and a long-tail exposure boost.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from coldtail.graph_adapt import graph_aware_fusion
from coldtail.metrics import (
    evaluate_rankings,
    limit_candidates_per_user,
    subgroup_metrics,
)

logger = logging.getLogger(__name__)


def run_graph_aware(
    processed_dir: Path,
    out_dir: Path,
    cfg: dict,
    top_candidates: int | None = None,
    rerank_users_path: Path | str | None = None,
) -> None:
    base_model = cfg.get("graph_aware", {}).get("base_model_for_graph", "lightgcn")
    required = [
        f"{base_model}_scores.csv",
        "bpr_scores.csv",
        "tfidf_scores.csv",
        "popularity_scores.csv",
    ]
    missing = [f for f in required if not (out_dir / f).exists()]
    if missing:
        logger.error(
            f"run_graph_aware: missing upstream score files {missing}. "
            f"Make sure run_all_baselines completed successfully before this step."
        )
        return

    candidates = pd.read_csv(processed_dir / "candidates.csv")
    candidates = limit_candidates_per_user(candidates, top_candidates)
    test = pd.read_csv(processed_dir / "test.csv")
    item_stats = pd.read_csv(processed_dir / "item_stats.csv")
    gcfg = cfg.get("graph_aware", {})

    if rerank_users_path is not None and Path(rerank_users_path).exists():
        rerank_users = pd.read_csv(rerank_users_path)["user_idx"].tolist()
        test = test[test.user_idx.isin(rerank_users)].copy()
        logger.info(
            f"[GraphAware] using rerank subset | num_users={len(rerank_users):,}"
        )
    else:
        logger.info("[GraphAware] using full test set")

    try:
        scored = graph_aware_fusion(
            candidates,
            out_dir,
            processed_dir / "item_stats.csv",
            alpha_semantic=gcfg.get("alpha_semantic", 0.35),
            beta_graph=gcfg.get("beta_graph", 0.45),
            gamma_base=gcfg.get("gamma_base", 0.30),
            lambda_popularity=gcfg.get("lambda_popularity", 0.10),
            tail_boost=gcfg.get("tail_boost", 0.05),
            base_model_for_graph=base_model,
        )
    except Exception as e:
        logger.error(f"graph_aware_fusion failed: {e}", exc_info=True)
        return

    scored.to_csv(out_dir / "graph_aware_scores.csv", index=False)
    metrics = evaluate_rankings(
        scored,
        test,
        cfg["metrics"]["k_list"],
        item_stats.set_index("item_idx")["is_tail"],
        top_candidates=top_candidates,
    )
    metrics["model"] = "graph_aware"
    pd.DataFrame([metrics]).to_csv(out_dir / "graph_aware_metrics.csv", index=False)
    sub = subgroup_metrics(
        scored, test, cfg["metrics"]["k_list"], top_candidates=top_candidates
    )
    if len(sub):
        sub["model"] = "graph_aware"
        sub.to_csv(out_dir / "graph_aware_subgroup_metrics.csv", index=False)
    logger.info(
        "[GraphAware] "
        + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
    )
