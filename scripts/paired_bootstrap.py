"""Paired significance test for end-to-end +LLM vs retriever-only.

The correct test for "does adding the LLM change recall@k?" is **paired**: the same users
are ranked twice (retriever-only, then +LLM), so we compare per-user indicators.

For each (dataset, retriever) over the N rerank users:
  delta_u = 1[gold in +LLM top-k]  -  1[gold in retriever-only top-k]
We report mean delta with a **paired bootstrap 95% CI** (resample users) and the
**McNemar discordant counts** (b = LLM-gained, c = LLM-lost) with an exact two-sided
p-value. Post-hoc, GPU-free: reads the e2e LLM scores + the retriever pools already on disk.

Usage:
  python scripts/paired_bootstrap.py --glob 'outputs/*-e2e-s42' --data_dir data --k 10
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from end2end_ci import find_retr_dir, top_hits  # reuse the exact ranking convention

logger = logging.getLogger(__name__)


def mcnemar_two_sided_p(b: int, c: int) -> float:
    """Exact binomial two-sided p-value on the discordant pairs (n=b+c, p=0.5)."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    x = min(b, c)
    tail = sum(comb(n, i) for i in range(x + 1)) / (2**n)
    return min(1.0, 2 * tail)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/*-e2e-s*")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_root", default="outputs")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

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
        ru = os.path.join(sp, "rerank_eval_users.csv")
        e2e_users = os.path.join(
            e2e_dir, "e2e_eval_users.csv"
        )  # B3 expanded subset, if present
        sub_path = (
            e2e_users
            if os.path.exists(e2e_users)
            else (ru if os.path.exists(ru) else None)
        )
        if sub_path:
            keep = set(pd.read_csv(sub_path)["user_idx"].astype(int))
            test = test[test.user_idx.astype(int).isin(keep)]
        gold = dict(zip(test.user_idx.astype(int), test.item_idx.astype(int)))
        users = sorted(gold)
        N = len(users)
        retr_dir = find_retr_dir(args.out_root, dataset, seed)

        print("\n" + "=" * 96)
        print(
            f"### {dataset}-e2e-s{seed}   N={N}   k={args.k}   (subset: {os.path.basename(sub_path) if sub_path else 'all test'})"
        )
        print("=" * 96)
        print(
            f"{'retriever':<11}{'retr@k':>8}{'llm@k':>7}{'meanΔ':>9}{'Δ 95% CI (paired)':>22}"
            f"{'b(gain)':>9}{'c(loss)':>9}{'McNemar p':>11}"
        )
        for sub in sorted(glob.glob(os.path.join(e2e_dir, "*", "llm_scores.csv"))):
            retr = os.path.basename(os.path.dirname(sub))
            pool_path = os.path.join(retr_dir, f"{retr}_scores.csv") if retr_dir else ""
            if not (retr_dir and os.path.exists(pool_path)):
                print(f"{retr:<11}  (retriever pool not found — skipped)")
                continue
            pool = pd.read_csv(pool_path)
            pool = pool[pool.user_idx.astype(int).isin(gold)]
            llm = pd.read_csv(sub)
            retr_hit = top_hits(pool, gold, args.k)
            llm_hit = top_hits(llm, gold, args.k)
            idx = {u: i for i, u in enumerate(users)}
            delta = np.zeros(N)
            for u in llm_hit:
                if u in idx:
                    delta[idx[u]] += 1
            for u in retr_hit:
                if u in idx:
                    delta[idx[u]] -= 1
            mean_d = float(delta.mean())
            boot = (
                np.array(
                    [delta[rng.integers(0, N, N)].mean() for _ in range(args.n_boot)]
                )
                if N
                else np.array([0.0])
            )
            lo, hi = np.percentile(boot, [2.5, 97.5])
            b = len(llm_hit - retr_hit)
            c = len(retr_hit - llm_hit)
            p = mcnemar_two_sided_p(b, c)
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(
                f"{retr:<11}{len(retr_hit):>8}{len(llm_hit):>7}{mean_d:>+9.4f}"
                f"{f'[{lo:+.4f},{hi:+.4f}]{sig}':>22}{b:>9}{c:>9}{p:>11.3f}"
            )
        print(
            "\nread: Δ = +LLM top-k − retriever-only top-k per user (paired). CI excludes 0 => '*'."
        )
        print(
            "b = users LLM moved INTO top-k that the retriever missed; c = users LLM dropped OUT."
        )
        print("Negative Δ / c>b means the LLM degrades the retriever's ranking.")


if __name__ == "__main__":
    main()
