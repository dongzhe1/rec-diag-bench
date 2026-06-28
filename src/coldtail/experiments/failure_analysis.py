"""Aggregate and compare per-model metrics, bootstrap CIs, subgroup breakdowns,
diagnostics (tie sensitivity, score separation), and oracle recall."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from coldtail.metrics import (
    bootstrap_ranking_ci,
    evaluate_rankings,
    limit_candidates_per_user,
    oracle_recall,
    score_separation,
    subgroup_metrics,
    tie_diagnostic,
)

logger = logging.getLogger(__name__)


def collect_scores(out_dir: Path) -> dict[str, pd.DataFrame]:
    scores = {}
    for path in sorted(out_dir.glob("*_scores.csv")):
        name = path.name.replace("_scores.csv", "")
        scores[name] = pd.read_csv(path)
    return scores


def run_failure_analysis(
    processed_dir: Path, out_dir: Path, cfg: dict, top_candidates: int | None = None
) -> None:
    test = pd.read_csv(processed_dir / "test.csv")
    item_stats = pd.read_csv(processed_dir / "item_stats.csv")
    scores = collect_scores(out_dir)
    k_list = cfg["metrics"]["k_list"]
    is_tail = item_stats.set_index("item_idx")["is_tail"]

    rerank_users_path = processed_dir / "rerank_eval_users.csv"
    if rerank_users_path.exists():
        rerank_user_set = set(pd.read_csv(rerank_users_path)["user_idx"].tolist())
        test_rerank = test[test["user_idx"].isin(rerank_user_set)].copy()
    else:
        rerank_user_set = None
        test_rerank = None

    n_boot = int(cfg.get("metrics", {}).get("bootstrap_n", 1000))

    cand_raw = pd.read_csv(processed_dir / "candidates.csv")
    cand_eval = limit_candidates_per_user(cand_raw, top_candidates)
    test_for_oracle = test_rerank if test_rerank is not None else test
    oracle = oracle_recall(cand_eval, test_for_oracle, k_list)
    oracle["pool_type"] = "positive_controlled"
    pd.DataFrame([oracle]).to_csv(out_dir / "oracle_recall.csv", index=False)
    logger.info(
        "[failure_analysis] oracle recall | pool_coverage=%.3f | "
        "median_true_rank=%.1f | oracle_recall@%d=%.3f",
        oracle.get("pool_coverage", 0),
        oracle.get("median_true_rank", float("nan")),
        k_list[0],
        oracle.get(f"oracle_recall@{k_list[0]}", 0),
    )

    rows = []
    sub_rows = []
    ci_rows = []
    tie_rows = []
    sep_rows = []
    for name, scored in scores.items():
        if not {"user_idx", "item_idx", "score"}.issubset(scored.columns):
            continue

        if test_rerank is not None:
            test_for_model = test_rerank
        else:
            test_for_model = test

        eval_user_set = set(test_for_model["user_idx"].unique())
        scored_user_set = set(scored["user_idx"].unique())
        n_covered = len(eval_user_set & scored_user_set)
        coverage = n_covered / len(eval_user_set) if eval_user_set else 0.0
        if coverage < 1.0:
            logger.warning(
                "[failure_analysis] model=%s covers %d/%d eval users (%.1f%%); "
                "the %d missing users are scored as misses (recall/ndcg/mrr=0). "
                "Check for a truncated or crashed score file.",
                name,
                n_covered,
                len(eval_user_set),
                100 * coverage,
                len(eval_user_set) - n_covered,
            )

        met = evaluate_rankings(
            scored, test_for_model, k_list, is_tail, top_candidates=top_candidates
        )
        met["model"] = name
        met["eval_users"] = len(eval_user_set)
        met["eval_user_coverage"] = round(coverage, 4)
        rows.append(met)

        sub = subgroup_metrics(
            scored, test_for_model, k_list, top_candidates=top_candidates
        )
        if len(sub):
            sub["model"] = name
            sub_rows.append(sub)

        ci = bootstrap_ranking_ci(
            scored, test_for_model, k_list, top_candidates=top_candidates, n_boot=n_boot
        )
        ci["model"] = name
        ci_rows.append(ci)

        sep = score_separation(scored, test_for_model)
        sep["model"] = name
        sep["scenario"] = "overall"
        sep_rows.append(sep)
        cohens_d = sep.get("cohens_d", float("nan"))
        if abs(cohens_d) < 0.2:
            logger.warning(
                "[failure_analysis] model=%s Cohen's d=%.3f (small effect) — "
                "scores barely discriminate relevant from irrelevant items; "
                "reranking quality may be near-random",
                name,
                cohens_d,
            )

        scenario_cols = ["is_user_cold", "is_item_cold", "is_long_tail", "is_warm"]
        for col in scenario_cols:
            if col not in test_for_model.columns:
                continue
            sub_test = test_for_model[test_for_model[col].astype(bool)]
            if len(sub_test) < 5:
                continue
            sub_scored = scored[scored["user_idx"].isin(sub_test["user_idx"])]
            if sub_scored.empty:
                continue
            sub_sep = score_separation(sub_scored, sub_test)
            sub_sep["model"] = name
            sub_sep["scenario"] = col
            sep_rows.append(sub_sep)

        tie = tie_diagnostic(
            scored, test_for_model, k_list, top_candidates=top_candidates
        )
        tie["model"] = name
        tie_rows.append(tie)
        if tie.get("tie_rate", 0.0) > 0.05:
            worst_gap = max(
                (tie.get(f"recall@{k}_gap", 0.0) for k in k_list), default=0.0
            )
            logger.warning(
                "[failure_analysis] model=%s tie_rate=%.1f%% | max recall best-vs-worst "
                "gap=%.3f — headline ranking is partly arbitrary; prefer log-prob scoring",
                name,
                100 * tie["tie_rate"],
                worst_gap,
            )

    if rows:
        all_metrics = pd.DataFrame(rows)
        if all_metrics["eval_users"].nunique() > 1:
            logger.warning(
                "[failure_analysis] models evaluated on differing population "
                "sizes %s — all_model_metrics.csv is NOT apples-to-apples",
                dict(zip(all_metrics["model"], all_metrics["eval_users"])),
            )
        all_metrics.to_csv(out_dir / "all_model_metrics.csv", index=False)
    if sub_rows:
        pd.concat(sub_rows, ignore_index=True).to_csv(
            out_dir / "all_model_subgroup_metrics.csv", index=False
        )
    if ci_rows:
        pd.DataFrame(ci_rows).to_csv(out_dir / "all_model_metrics_ci.csv", index=False)
    if tie_rows:
        pd.DataFrame(tie_rows).to_csv(out_dir / "tie_diagnostic.csv", index=False)
    if sep_rows:
        pd.DataFrame(sep_rows).to_csv(out_dir / "score_separation.csv", index=False)

    cand = pd.read_csv(processed_dir / "candidates.csv")
    cand = limit_candidates_per_user(cand, top_candidates)
    candidate_size = (
        cand.groupby("user_idx").size().describe().to_frame("candidate_size")
    )
    scenario_cols = ["is_user_cold", "is_item_cold", "is_long_tail", "is_warm"]
    available = [c for c in scenario_cols if c in test.columns]
    if available:
        scenario_counts = (
            test[available]
            .sum()
            .rename_axis("scenario")
            .reset_index(name="num_test_cases")
        )
        scenario_counts.to_csv(out_dir / "scenario_counts.csv", index=False)
    candidate_size.to_csv(out_dir / "candidate_size_summary.csv", index=False)
