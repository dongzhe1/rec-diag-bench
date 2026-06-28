"""Add counts (n) and 95% Wilson CIs to the end-to-end retrieve->rerank tables.

Post-hoc, from the existing e2e outputs (no LLM re-run): for each (dataset, retriever)
it reports N, covered-n, retriever-top10-n, LLM-top10-n, and CIs, so conditional rates
(esp. MIND, where coverage is 4-7% and covered cases are few) come with explicit
denominators and uncertainty.

Reads:
  outputs/{ds}-e2e-s{seed}/{retriever}/llm_scores.csv   (LLM ranking of the realistic pool)
  outputs/{ds}-retrieval-s{seed}-N*/{retriever}_scores.csv  (retriever-only pool + coverage)
  {data_dir}/processed/{ds}/s{seed}/test.csv  (gold) + rerank_eval_users.csv

Usage:
  python scripts/end2end_ci.py --glob 'outputs/*-e2e-s42' --data_dir data
"""

from __future__ import annotations

import argparse
import glob
import logging
import math
import os
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def top_hits(
    scores: pd.DataFrame, gold: dict[int, int], k: int | None, seed: int = 42
) -> set[int]:
    """Users whose gold item is in their top-k (k=None -> anywhere in pool)."""
    rng = np.random.default_rng(seed)
    s = scores.assign(_t=rng.random(len(scores))).sort_values(
        ["user_idx", "score", "_t"], ascending=[True, False, False]
    )
    hit = set()
    for u, g in s.groupby("user_idx"):
        u = int(u)
        if u not in gold:
            continue
        items = g["item_idx"].astype(int).tolist()
        if gold[u] in (items[:k] if k else items):
            hit.add(u)
    return hit


def find_retr_dir(out_root: str, dataset: str, seed: int) -> str | None:
    for d in sorted(
        glob.glob(os.path.join(out_root, f"{dataset}-retrieval-s{seed}-N*"))
    ):
        if glob.glob(os.path.join(d, "*_scores.csv")):
            return d
    return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/*-e2e-s*")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_root", default="outputs")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    for e2e_dir in sorted(glob.glob(args.glob)):
        m = re.match(r"(.+)-e2e-s(\d+)", os.path.basename(e2e_dir.rstrip("/")))
        if not m:
            continue
        dataset, seed = m.group(1), int(m.group(2))
        sp = os.path.join(args.data_dir, "processed", dataset, f"s{seed}")
        if not os.path.exists(os.path.join(sp, "test.csv")):
            print(f"\n### {dataset}: split not found — skipped")
            continue
        test = pd.read_csv(os.path.join(sp, "test.csv"))
        ru_path = os.path.join(sp, "rerank_eval_users.csv")
        rerank_users = (
            set(pd.read_csv(ru_path)["user_idx"].astype(int))
            if os.path.exists(ru_path)
            else None
        )
        if rerank_users is not None:
            test = test[test.user_idx.astype(int).isin(rerank_users)]
        gold = dict(zip(test.user_idx.astype(int), test.item_idx.astype(int)))
        N = len(gold)
        retr_dir = find_retr_dir(args.out_root, dataset, seed)

        print("\n" + "=" * 100)
        print(f"### {dataset}-e2e-s{seed}   N(rerank users with gold) = {N}")
        print("=" * 100)
        print(
            f"{'retriever':<11}{'cov n':>7}{'cov%':>7}{'cov 95%CI':>15}"
            f"{'retr@10 n':>10}{'llm@10 n':>9}{'e2e llm%':>9}{'cond%':>7}{'cond 95%CI':>15}"
        )
        for sub in sorted(glob.glob(os.path.join(e2e_dir, "*", "llm_scores.csv"))):
            retr = os.path.basename(os.path.dirname(sub))
            llm = pd.read_csv(sub)
            pool_path = os.path.join(retr_dir, f"{retr}_scores.csv") if retr_dir else ""
            if not (retr_dir and os.path.exists(pool_path)):
                print(
                    f"{retr:<11}  (retriever pool not found — coverage/retr-only skipped)"
                )
                continue
            pool = pd.read_csv(pool_path)
            pool = pool[pool.user_idx.astype(int).isin(gold)]
            covered = top_hits(pool, gold, None)
            retr_hit = top_hits(pool, gold, args.k)
            llm_hit = top_hits(llm, gold, args.k)
            ncov, nretr, nllm = len(covered), len(retr_hit), len(llm_hit)
            cov_lo, cov_hi = wilson(ncov, N)
            cond_lo, cond_hi = wilson(nllm, ncov)
            cond = nllm / ncov if ncov else float("nan")
            print(
                f"{retr:<11}{ncov:>7}{ncov / N * 100:>6.1f}%{f'[{cov_lo:.2f},{cov_hi:.2f}]':>15}"
                f"{nretr:>10}{nllm:>9}{nllm / N * 100:>8.1f}%{cond * 100:>6.1f}%"
                f"{f'[{cond_lo:.2f},{cond_hi:.2f}]':>15}"
            )
        print(
            "\nread: e2e_llm% = LLM-top10 / N; cond% = LLM-top10 / covered. Wilson 95% CIs; when"
        )
        print(
            "covered-n is tiny (e.g. MIND) the conditional CI is wide — do not over-read point values."
        )


if __name__ == "__main__":
    main()
