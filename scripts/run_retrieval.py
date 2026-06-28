"""Entry point for the retrieval-realistic coverage experiment.

Tests the candidate-expansion pillar by letting each retriever propose its own
top-N from the full catalogue with no gold-guarantee, then measuring how often
the gold item is covered — overall and per cold/tail scenario.

This REUSES an existing seed-specific split (read-only); it never re-splits and
never touches the main pipeline's candidate pool. Results go to a separate run
dir so the original outputs/ are untouched:

    outputs/{dataset}-retrieval-s{seed}-N{n}/

Prerequisite: the split must already exist at
    data/processed/{dataset}/s{seed}/
(build it once with `run_dataset.py --split_only` if absent).

Usage:
    python scripts/run_retrieval.py --dataset ml-20m --seed 42 \
        --config configs/hpc.yaml --retrieval_n 100
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401

from coldtail.config import load_config
from coldtail.experiments.retrieval_coverage import (
    DEFAULT_RETRIEVERS,
    run_retrieval_coverage,
)
from coldtail.utils import seed_everything

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_REQUIRED_SPLIT_FILES = ["train.csv", "test.csv", "items_mapped.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval-realistic coverage experiment"
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset name (must already be split)"
    )
    parser.add_argument(
        "--data_dir", default="data", help="Root directory for data (raw/processed)"
    )
    parser.add_argument("--output_dir", default="outputs", help="Output root")
    parser.add_argument(
        "--config", default="configs/local.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed of the split to reuse"
    )
    parser.add_argument(
        "--retrieval_n",
        type=int,
        default=200,
        help="Per-user retrieved pool size N (gold NOT injected). Default 200, "
        "matching the top-200 reranking budget so coverage = the retrieval ceiling "
        "for the main-table reranking runs.",
    )
    parser.add_argument(
        "--retrievers",
        default=None,
        help="Comma-separated subset of retrievers to run "
        f"(default: {','.join(DEFAULT_RETRIEVERS)})",
    )
    parser.add_argument(
        "--write_pools",
        action="store_true",
        help="Also write each retriever's realistic pool CSV (for downstream rerank).",
    )
    parser.add_argument(
        "--eval_split",
        choices=["test", "valid"],
        default="test",
        help="Prediction target split. 'valid' builds pools whose gold is the validation "
        "positive (only train is masked as seen) — used to train the learned fusion (LHF) "
        "without test leakage. Writes to a separate '-valid' run dir.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    processed_dir = data_dir / "processed" / args.dataset / f"s{args.seed}"

    missing = [f for f in _REQUIRED_SPLIT_FILES if not (processed_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Split not found at {processed_dir} (missing {missing}). "
            f"Build it first with:\n"
            f"  python scripts/run_dataset.py --dataset {args.dataset} "
            f"--data_dir {args.data_dir} --config {args.config} "
            f"--top_k 500 --seed {args.seed} --split_only --force_split"
        )

    split_tag = "-valid" if args.eval_split == "valid" else ""
    out_dir = (
        Path(args.output_dir).resolve()
        / f"{args.dataset}-retrieval{split_tag}-s{args.seed}-N{args.retrieval_n}"
    )

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    seed_everything(args.seed)

    retrievers = (
        [r.strip() for r in args.retrievers.split(",")] if args.retrievers else None
    )

    logger.info(
        "[retrieval] starting | dataset=%s seed=%d N=%d split=%s | data=%s | out=%s",
        args.dataset,
        args.seed,
        args.retrieval_n,
        args.eval_split,
        processed_dir,
        out_dir,
    )
    report = run_retrieval_coverage(
        processed_dir=processed_dir,
        out_dir=out_dir,
        cfg=cfg,
        retrieval_n=args.retrieval_n,
        retrievers=retrievers,
        write_pools=args.write_pools,
        eval_split=args.eval_split,
    )
    logger.info("[retrieval] complete | report=%s", report)


if __name__ == "__main__":
    main()
