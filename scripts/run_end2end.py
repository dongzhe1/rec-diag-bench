"""End-to-end retrieve -> LLM-rerank evaluation.

Closes the gap between positive-controlled and retrieval-realistic evaluation:
instead of handing the LLM the positive-controlled pool (gold guaranteed present),
feed it each retriever's realistic pool (full-catalogue top-N, gold NOT injected)
and measure end-to-end accuracy. This decomposes accuracy as:

    end-to-end recall@10  =  retrieval coverage@N  x  conditional rerank success

so the bottleneck (retrieval coverage) is directly visible.

Reuses the realistic pools written by run_retrieval.py --write_pools
(outputs/{ds}-retrieval-s{seed}-N{n}/{retriever}_scores.csv). The split is read-only.

Output: outputs/{dataset}-e2e-s{seed}/
  end2end_summary.csv     one row per retriever (coverage / retriever-only / +LLM)
  {retriever}/llm_*.csv   the LLM's end-to-end scores+metrics on that pool

Usage:
  python scripts/run_end2end.py --dataset ml-20m --seed 42 --top_k 200 \
      --config configs/hpc.yaml --retrievers lightgcn,sbert,fusion,cara

GPU required (runs LLM reranking on realistic pools).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from coldtail.config import load_config
from coldtail.experiments.run_llm_rerank import run_llm_rerank
from coldtail.metrics import evaluate_rankings
from coldtail.utils import seed_everything

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_RETRIEVERS = ["lightgcn", "sbert", "fusion", "fusion_il", "cara"]


def _find_retrieval_dir(out_root: Path, dataset: str, seed: int) -> Path | None:
    import glob

    cands = sorted(glob.glob(str(out_root / f"{dataset}-retrieval-s{seed}-N*")))
    cands += [str(out_root / f"{dataset}-top200-s{seed}" / "retrieval")]
    for d in cands:
        if list(Path(d).glob("*_scores.csv")):
            return Path(d)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end retrieve -> LLM rerank")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--config", default="configs/local.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--top_k", type=int, default=200, help="pool size N (= retrieval N)"
    )
    ap.add_argument(
        "--retrievers",
        default=",".join(DEFAULT_RETRIEVERS),
        help="comma list of retriever pools to rerank end-to-end",
    )
    ap.add_argument(
        "--retrieval_dir", default=None, help="override the retrieval pool dir"
    )
    ap.add_argument(
        "--n_users",
        type=int,
        default=None,
        help="Evaluate on this many sampled users instead of the stratified "
        "rerank_eval_users.csv (subset-size robustness, review B3).",
    )
    ap.add_argument(
        "--user_sample",
        choices=["stratified", "natural"],
        default="natural",
        help="With --n_users: 'natural' samples uniformly from all test users "
        "(realistic traffic); 'stratified' samples from rerank_eval_users.csv.",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    processed_dir = data_dir / "processed" / args.dataset / f"s{args.seed}"
    rerank_users_path = processed_dir / "rerank_eval_users.csv"
    out_root = Path(args.output_dir).resolve()
    retr_dir = (
        Path(args.retrieval_dir)
        if args.retrieval_dir
        else _find_retrieval_dir(out_root, args.dataset, args.seed)
    )
    if retr_dir is None or not retr_dir.exists():
        raise FileNotFoundError(
            f"No retrieval pool dir for {args.dataset} s{args.seed} under {out_root}. "
            f"Run run_retrieval.py --write_pools first."
        )

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    seed_everything(args.seed)
    k_list = cfg.get("metrics", {}).get("k_list", [5, 10, 20, 49])
    cov_k = args.top_k  # coverage = recall at the full pool depth
    eval_ks = sorted(set(k_list + [cov_k]))

    test = pd.read_csv(processed_dir / "test.csv")
    item_stats = pd.read_csv(processed_dir / "item_stats.csv")
    is_tail = item_stats.set_index("item_idx")["is_tail"]
    base_out = out_root / f"{args.dataset}-e2e-s{args.seed}"

    rerank_users = (
        set(pd.read_csv(rerank_users_path)["user_idx"])
        if rerank_users_path.exists()
        else None
    )
    if args.n_users is not None:
        # Sample n_users to test subset-size robustness.
        rng = np.random.default_rng(args.seed)
        if args.user_sample == "natural" or rerank_users is None:
            pool_users = test.user_idx.unique()
        else:
            pool_users = np.array(sorted(rerank_users))
        n = min(args.n_users, len(pool_users))
        rerank_users = set(
            int(u) for u in rng.choice(pool_users, size=n, replace=False)
        )
        base_out.mkdir(parents=True, exist_ok=True)
        rerank_users_path = base_out / "e2e_eval_users.csv"
        pd.DataFrame({"user_idx": sorted(rerank_users)}).to_csv(
            rerank_users_path, index=False
        )
        logger.info(
            "[e2e] sampled %d users (%s) -> %s", n, args.user_sample, rerank_users_path
        )
    if rerank_users is not None:
        test_sub = test[test.user_idx.isin(rerank_users)].copy()
    else:
        test_sub = test

    rows = []
    for retr in [r.strip() for r in args.retrievers.split(",")]:
        pool_path = retr_dir / f"{retr}_scores.csv"
        if not pool_path.exists():
            logger.warning("[e2e] pool %s not found — skipped", pool_path)
            continue
        pool = pd.read_csv(pool_path)
        pool_sub = (
            pool[pool.user_idx.isin(test_sub.user_idx)].copy()
            if rerank_users is not None
            else pool
        )

        # LLM reranks this realistic pool (gold may be absent)
        retr_metrics = evaluate_rankings(
            pool_sub, test_sub, eval_ks, is_tail, top_candidates=cov_k
        )
        coverage = retr_metrics.get(f"recall@{cov_k}", float("nan"))
        work = base_out / retr
        logger.info(
            "[e2e] %s | LLM reranking realistic pool | coverage@%d=%.4f",
            retr,
            cov_k,
            coverage,
        )
        run_llm_rerank(
            processed_dir=processed_dir,
            out_dir=work,
            cfg=cfg,
            top_candidates=cov_k,
            rerank_users_path=rerank_users_path if rerank_users_path.exists() else None,
            run_baseline=True,
            gala_variants=[],
            candidates_override=pool_sub,
        )
        llm_m = pd.read_csv(work / "llm_metrics.csv").iloc[0]

        def cond(k):  # conditional reranking success = e2e / coverage
            c = retr_metrics.get(f"recall@{cov_k}", float("nan"))
            return float(llm_m[f"recall@{k}"]) / c if c and c > 0 else float("nan")

        rows.append(
            {
                "retriever": retr,
                f"coverage@{cov_k}": round(coverage, 4),
                "retriever_recall@10": round(
                    retr_metrics.get("recall@10", float("nan")), 4
                ),
                "e2e_llm_recall@10": round(float(llm_m["recall@10"]), 4),
                "e2e_llm_ndcg@10": round(float(llm_m["ndcg@10"]), 4),
                "conditional_llm_recall@10": round(cond(10), 4),
            }
        )
        logger.info(
            "[e2e] %s | e2e_llm R@10=%.4f (= coverage %.4f x conditional %.4f)",
            retr,
            float(llm_m["recall@10"]),
            coverage,
            cond(10),
        )

    base_out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(base_out / "end2end_summary.csv", index=False)
    print("\n=== END-TO-END retrieve -> LLM rerank (decomposition) ===")
    print(
        f"dataset={args.dataset} seed={args.seed} N={cov_k} | eval users={len(test_sub):,}"
    )
    print(summary.to_string(index=False))
    print(
        "\nread: e2e_llm_recall@10 ≈ coverage@N × conditional_llm_recall@10 — the bottleneck is coverage."
    )


if __name__ == "__main__":
    main()
