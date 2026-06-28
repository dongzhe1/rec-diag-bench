"""Generate a markdown report aggregating metrics, diagnostics, oracle recall,
and score separation from completed experiment outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _safe_read(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or len(df) == 0:
        return "No data.\n"
    return df.head(max_rows).to_markdown(index=False)


def make_report(processed_dir: Path, out_dir: Path, cfg: dict) -> Path:
    report = []
    report.append("# ColdTail-LLMRec Local MVP Report\n")
    report.append("## Dataset summary\n")
    for fname in ["train.csv", "valid.csv", "test.csv", "candidates.csv"]:
        path = processed_dir / fname
        if path.exists():
            df = pd.read_csv(path)
            report.append(f"- `{fname}`: {len(df):,} rows")
    if (processed_dir / "items_mapped.csv").exists():
        items = pd.read_csv(processed_dir / "items_mapped.csv")
        report.append(f"- items: {len(items):,}")

    scenario_counts = _safe_read(out_dir / "scenario_counts.csv")
    report.append("\n## Failure-mode scenario counts\n")
    if scenario_counts is not None:
        report.append(_md_table(scenario_counts))

    metrics = _safe_read(out_dir / "all_model_metrics.csv")
    from_subset = metrics is not None
    if metrics is None:
        base = _safe_read(out_dir / "baseline_metrics.csv")
        graph = _safe_read(out_dir / "graph_aware_metrics.csv")
        frames = [x for x in [base, graph] if x is not None]
        metrics = pd.concat(frames, ignore_index=True) if frames else None
    report.append("\n## Overall metrics\n")
    if from_subset:
        n_eval = ""
        if "eval_users" in metrics.columns and metrics["eval_users"].nunique() == 1:
            n_eval = f" (N={int(metrics['eval_users'].iloc[0])} users)"
        report.append(
            f"> Computed on the fixed **stratified rerank-eval user subset**{n_eval}, "
            "which deliberately over-samples cold-start and long-tail users. All "
            "models share this identical population, so cross-model comparison here "
            "is apples-to-apples — but these overall numbers are an average over a "
            "cold/tail-weighted population and do **not** represent natural "
            "dataset-level performance. Read the per-scenario subgroup table below "
            "for the unbiased within-group comparison.\n"
        )
    elif metrics is not None:
        report.append(
            "> ⚠️ `all_model_metrics.csv` not found — falling back to per-model CSVs "
            "(`baseline_metrics.csv`, `graph_aware_metrics.csv`), computed on the "
            "**full test set** (baselines only). These are **not** comparable to "
            "reranker/LLM/GALA numbers, which use the rerank subset. Run "
            "failure_analysis to produce the apples-to-apples table.\n"
        )
    if metrics is not None:
        cols = [
            c
            for c in [
                "model",
                "recall@5",
                "recall@10",
                "ndcg@10",
                "mrr@10",
                "unique_exposed_items",
                "unique_tail_exposed_items",
                "gini_exposure",
            ]
            if c in metrics.columns
        ]
        sort_col = next(
            (c for c in ["recall@10", "recall@5", "ndcg@10"] if c in cols), "model"
        )
        report.append(_md_table(metrics[cols].sort_values(sort_col, ascending=False)))

    sub = _safe_read(out_dir / "all_model_subgroup_metrics.csv")
    report.append("\n## Subgroup metrics by failure mode\n")
    if sub is not None:
        report.append(
            "> **Primary result.** Within each scenario every model is evaluated on "
            "the same users, so this comparison is unaffected by the subset's "
            "stratified sampling. `num_users` is the per-scenario sample size.\n"
        )
        cols = [
            c
            for c in [
                "model",
                "scenario",
                "num_users",
                "recall@10",
                "ndcg@10",
                "mrr@10",
                "unique_tail_exposed_items",
                "gini_exposure",
            ]
            if c in sub.columns
        ]
        report.append(_md_table(sub[cols].sort_values(["scenario", "model"])))

    ci = _safe_read(out_dir / "all_model_metrics_ci.csv")
    if ci is not None:
        report.append("\n## Bootstrap 95% CIs (recall@10 / ndcg@10)\n")
        report.append(
            "> Resampled over eval users. If two models' intervals overlap, the "
            "gap is within sampling noise — don't over-read it.\n"
        )
        cols = [
            c
            for c in [
                "model",
                "recall@10",
                "recall@10_ci_lo",
                "recall@10_ci_hi",
                "ndcg@10",
                "ndcg@10_ci_lo",
                "ndcg@10_ci_hi",
                "n_users",
            ]
            if c in ci.columns
        ]
        sort_col = "recall@10" if "recall@10" in cols else "model"
        report.append(_md_table(ci[cols].sort_values(sort_col, ascending=False)))

    tie = _safe_read(out_dir / "tie_diagnostic.csv")
    if tie is not None:
        report.append("\n## Tie sensitivity\n")
        report.append(
            "> `tie_rate` = fraction of users whose true item shares its score with "
            "another candidate. `recall@10_gap` = best-case minus worst-case "
            "recall@10 under tie ordering. A large gap means that model's headline "
            "ranking is partly arbitrary (typical of integer-score LLMs — prefer "
            "`llm_scoring_mode: logprob`).\n"
        )
        cols = [
            c
            for c in ["model", "tie_rate", "recall@10_gap", "ndcg@10_gap", "n_users"]
            if c in tie.columns
        ]
        sort_col = "tie_rate" if "tie_rate" in cols else "model"
        report.append(_md_table(tie[cols].sort_values(sort_col, ascending=False)))

    oracle = _safe_read(out_dir / "oracle_recall.csv")
    if oracle is not None:
        report.append("\n## Oracle recall (retrieval bottleneck)\n")
        report.append(
            "> Measures whether the true item is present in the candidate pool "
            "**before** any model scores it. `pool_coverage` = fraction of eval users "
            "whose true item is in the pool at all. `oracle_recall@k` = fraction with "
            "true item in the top-k of the pool's own ordering. "
            "Our positive-controlled pool guarantees ~1.0 by construction — this "
            "isolates reranker quality from retrieval coverage. For real-retriever "
            "pools these numbers can be much lower (cf. arXiv:2604.16318 who observed "
            "pool_coverage ≈ 0.11 @ k=200, which dominated all downstream quality).\n"
        )
        report.append(_md_table(oracle))

    sep = _safe_read(out_dir / "score_separation.csv")
    if sep is not None:
        report.append("\n## Score separation (relevant vs irrelevant)\n")
        report.append(
            "> Measures whether a model's scores actually discriminate relevant items "
            "from irrelevant ones. `cohens_d` < 0.2 = small effect (ranking is "
            "near-random despite model cost). `spearman_r` near 0 = near-zero rank "
            "correlation with relevance. "
            "Baseline for comparison: arXiv:2604.16318 found Cohen's d = 0.11 and "
            "Spearman r = 0.004 for an out-of-domain cross-encoder — essentially "
            "random. Higher is better for both metrics.\n"
        )
        cols = [
            c
            for c in [
                "model",
                "scenario",
                "cohens_d",
                "spearman_r",
                "mean_diff",
                "overlap_coef",
                "p_value",
                "n_relevant",
                "n_irrelevant",
            ]
            if c in sep.columns
        ]
        sort_cols = ["scenario", "cohens_d"] if "scenario" in cols else ["cohens_d"]
        report.append(
            _md_table(
                sep[cols].sort_values(sort_cols, ascending=[True, False]),
                max_rows=50,
            )
        )

    report.append("\n## Interpretation checklist\n")
    report.append("- Does graph-aware fusion improve cold-user Recall/NDCG?")
    report.append(
        "- Does it improve long-tail item exposure without excessive accuracy loss?"
    )
    report.append(
        "- Is exposure less concentrated than popularity or pure graph models?"
    )
    report.append(
        "- Which scenario remains weak and should drive the next method iteration?"
    )

    out_path = out_dir / "report.md"
    out_path.write_text("\n".join(report), encoding="utf-8")
    return out_path
