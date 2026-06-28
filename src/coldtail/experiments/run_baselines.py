"""Train and evaluate baseline recommenders on the candidate pool.

Runs popularity, item-kNN, TF-IDF, Markov, SBERT, BPR, LightGCN, and SASRec,
each scoring the candidate pool and producing per-model metrics files.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path

import pandas as pd

from coldtail.metrics import (
    evaluate_rankings,
    limit_candidates_per_user,
    subgroup_metrics,
)
from coldtail.recommenders.bpr import train_bpr_score_candidates
from coldtail.recommenders.content_tfidf import score_tfidf
from coldtail.recommenders.itemknn import fit_itemknn_scores
from coldtail.recommenders.lightgcn import train_lightgcn_score_candidates
from coldtail.recommenders.markov import score_markov
from coldtail.recommenders.popularity import score_popularity
from coldtail.recommenders.sasrec import train_sasrec_score_candidates
from coldtail.recommenders.sbert_retrieval import score_sbert
from coldtail.utils import get_device, save_json

logger = logging.getLogger(__name__)


def _configure_blas_threads(cfg: dict) -> None:
    num_cpu = os.cpu_count() or 1
    training_cfg = cfg.get("training", {})
    num_threads = str(training_cfg.get("blas_threads", max(1, num_cpu // 2)))

    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, num_threads)

    logger.info(
        f"[Config] BLAS threads | num_threads={num_threads} | num_cpu={num_cpu}"
    )


def _gpu_peak_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return None


def run_all_baselines(
    processed_dir: Path,
    out_dir: Path,
    cfg: dict,
    top_candidates: int | None = None,
    rerank_users_path: Path | str | None = None,
    timing_collector=None,
) -> None:
    _configure_blas_threads(cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(processed_dir / "train.csv")

    max_history = cfg.get("max_history")
    if max_history:
        before = len(train)
        train = (
            train.sort_values("timestamp")
            .groupby("user_idx", sort=False)
            .tail(max_history)
            .reset_index(drop=True)
        )
        logger.info(
            f"[History] truncated train to last {max_history} per user | {before:,} -> {len(train):,} rows"
        )

    test = pd.read_csv(processed_dir / "test.csv")
    candidates = pd.read_csv(processed_dir / "candidates.csv")
    candidates = limit_candidates_per_user(candidates, top_candidates)
    items = pd.read_csv(processed_dir / "items_mapped.csv")
    item_stats = pd.read_csv(processed_dir / "item_stats.csv")

    if rerank_users_path is not None and Path(rerank_users_path).exists():
        rerank_users = pd.read_csv(rerank_users_path)["user_idx"].tolist()
        test = test[test.user_idx.isin(rerank_users)].copy()
        logger.info(
            f"[Eval] baseline metrics on rerank subset | num_users={len(rerank_users):,}"
        )
    else:
        logger.info(
            "[Eval] rerank_users_path not provided/found | baseline metrics on full test set"
        )

    k_list = cfg["metrics"]["k_list"]
    metrics_rows = []
    subgroup_rows = []
    failed = []
    is_tail = item_stats.set_index("item_idx")["is_tail"]
    num_pairs = len(candidates)

    def eval_and_save(name: str, scored: pd.DataFrame) -> None:
        scored.to_csv(out_dir / f"{name}_scores.csv", index=False)
        met = evaluate_rankings(
            scored, test, k_list, is_tail, top_candidates=top_candidates
        )
        met["model"] = name
        metrics_rows.append(met)
        sub = subgroup_metrics(scored, test, k_list, top_candidates=top_candidates)
        if len(sub):
            sub["model"] = name
            subgroup_rows.append(sub)
        metrics_str = ", ".join(
            f"{k}={v:.4f}" for k, v in met.items() if isinstance(v, float)
        )
        logger.info(f"[{name}] metrics | {metrics_str}")

    def run_one(name: str, fn) -> None:
        logger.info(f"[{name}] starting")
        t0 = time.perf_counter()
        try:
            eval_and_save(name, fn())
            elapsed = time.perf_counter() - t0
            if timing_collector is not None:
                timing_collector.record(name, num_pairs, elapsed, _gpu_peak_mb())
        except Exception as e:
            logger.error(f"[{name}] failed | error={e}", exc_info=True)
            failed.append(name)
        finally:
            gc.collect()

    baselines_cfg = cfg.get("baselines", {})
    device = get_device(cfg.get("training", {}).get("device", "auto"))
    training_cfg = cfg.get("training", {})
    lgcn_cfg = cfg.get("lightgcn", {})
    sasrec_cfg = cfg.get("sasrec", {})
    sbert_cfg = cfg.get("sbert", {})

    topk_sim = cfg.get("itemknn", {}).get("topk_sim", 100)
    embedding_dim = training_cfg.get("embedding_dim", 64)
    epochs_bpr = training_cfg.get("epochs_bpr", 20)
    epochs_lgcn = training_cfg.get("epochs_lightgcn", 20)
    batch_size = training_cfg.get("batch_size", 4096)
    learning_rate = training_cfg.get("lr", 1e-3)
    weight_decay = training_cfg.get("weight_decay", 1e-6)
    num_layers = lgcn_cfg.get("n_layers", 2)
    seed = cfg.get("seed", 42)

    if baselines_cfg.get("run_popularity", True):
        run_one("popularity", lambda: score_popularity(train, candidates, items))
    if baselines_cfg.get("run_itemknn", True):
        run_one(
            "itemknn", lambda: fit_itemknn_scores(train, candidates, topk_sim, items)
        )
    if baselines_cfg.get("run_tfidf", True):
        run_one("tfidf", lambda: score_tfidf(train, candidates, items))
    if baselines_cfg.get("run_markov", True):
        run_one("markov", lambda: score_markov(train, candidates, items))

    if baselines_cfg.get("run_sbert", True):
        sbert_model = sbert_cfg.get("model_name", "BAAI/bge-base-en-v1.5")
        sbert_batch = sbert_cfg.get("batch_size", 256)
        run_one(
            "sbert",
            lambda: score_sbert(
                train,
                candidates,
                items,
                model_name=sbert_model,
                device=device,
                batch_size=sbert_batch,
            ),
        )

    if baselines_cfg.get("run_bpr", True):
        run_one(
            "bpr",
            lambda: train_bpr_score_candidates(
                train,
                candidates,
                items,
                dim=embedding_dim,
                epochs=epochs_bpr,
                batch_size=batch_size,
                lr=learning_rate,
                weight_decay=weight_decay,
                device=device,
                seed=seed,
            ),
        )
    if baselines_cfg.get("run_lightgcn", True):
        run_one(
            "lightgcn",
            lambda: train_lightgcn_score_candidates(
                train,
                candidates,
                items,
                dim=embedding_dim,
                epochs=epochs_lgcn,
                batch_size=batch_size,
                lr=learning_rate,
                weight_decay=weight_decay,
                n_layers=num_layers,
                device=device,
                seed=seed,
            ),
        )
    if baselines_cfg.get("run_sasrec", True):
        run_one(
            "sasrec",
            lambda: train_sasrec_score_candidates(
                train,
                candidates,
                items,
                dim=embedding_dim,
                max_seq_len=sasrec_cfg.get("max_seq_len", 50),
                n_heads=sasrec_cfg.get("n_heads", 2),
                n_layers=sasrec_cfg.get("n_layers", 2),
                dropout=sasrec_cfg.get("dropout", 0.2),
                epochs=training_cfg.get("epochs_sasrec", 100),
                batch_size=min(batch_size, 256),
                lr=learning_rate,
                weight_decay=weight_decay,
                device=device,
                seed=seed,
            ),
        )

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(out_dir / "baseline_metrics.csv", index=False)
    if subgroup_rows:
        pd.concat(subgroup_rows, ignore_index=True).to_csv(
            out_dir / "baseline_subgroup_metrics.csv", index=False
        )

    status = "partial" if failed else "ok"
    save_json(
        {"status": status, "num_models": len(metrics_rows), "failed": failed},
        out_dir / "baseline_run.json",
    )
    if failed:
        logger.warning(f"[Baselines] completed with failures | failed_models={failed}")
    else:
        logger.info("[Baselines] all models completed successfully")
