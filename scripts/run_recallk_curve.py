"""Recall@K curve comparison across methods.

Produces a table and scenario-wise curves comparing Recall@K at
K=10,50,100,200,500 for: best single retriever, RRF, CARA, LHF, oracle union.
Also produces scenario-wise item_new / warm curves.

POST-HOC & CPU-ONLY. Reads existing pool CSVs.

Usage:
  python scripts/run_recallk_curve.py --dataset yelp-Philadelphia-Restaurants \
      --seed 42 --data_dir data
"""

from __future__ import annotations

import argparse
import logging
import os

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SCENARIOS = ["is_item_new", "is_item_cold", "is_long_tail", "is_user_cold", "is_warm"]
BASE_RETRIEVERS = [
    "lightgcn",
    "itemknn",
    "sbert",
    "tfidf",
    "popularity",
    "bpr",
    "markov",
    "graph_cooccur",
    "graph_emb",
    "sasrec",
]
K_VALUES = [10, 50, 100, 200, 500]


def _ranked(pool: pd.DataFrame, seed: int = 42) -> dict[int, list[int]]:
    rng = np.random.default_rng(seed)
    s = pool.assign(_t=rng.random(len(pool))).sort_values(
        ["user_idx", "score", "_t"], ascending=[True, False, False]
    )
    return {
        int(u): g["item_idx"].astype(int).tolist() for u, g in s.groupby("user_idx")
    }


def _recall_at_k(topn_by_user, gold, users, k):
    hits = 0
    for u in users:
        if gold.get(u) in topn_by_user.get(u, [])[:k]:
            hits += 1
    return hits / len(users) if users else 0.0


def _scenario_recall(topn_by_user, gold, test, users, k, scenario_col):
    if scenario_col not in test.columns:
        return float("nan"), 0
    flags = dict(zip(test.user_idx.astype(int), test[scenario_col].astype(bool)))
    sub = [u for u in users if flags.get(u, False)]
    if not sub:
        return float("nan"), 0
    return _recall_at_k(topn_by_user, gold, sub, k), len(sub)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--ks", default=",".join(str(k) for k in K_VALUES))
    args = ap.parse_args()

    rdir = os.path.join(
        args.output_dir, f"{args.dataset}-retrieval-s{args.seed}-N{args.N}"
    )
    sp = os.path.join(args.data_dir, "processed", args.dataset, f"s{args.seed}")
    ks = [int(k) for k in args.ks.split(",")]

    test = pd.read_csv(os.path.join(sp, "test.csv"))
    gold = dict(zip(test.user_idx.astype(int), test.item_idx.astype(int)))

    methods = {}
    for r in BASE_RETRIEVERS:
        p = os.path.join(rdir, f"{r}_scores.csv")
        if os.path.exists(p):
            methods[r] = _ranked(pd.read_csv(p))

    for extra in ["fusion", "cara", "lhf"]:
        p = os.path.join(rdir, f"{extra}_scores.csv")
        if os.path.exists(p):
            methods[extra] = _ranked(pd.read_csv(p))

    if not methods:
        logger.warning("no pool CSVs found in %s", rdir)
        return

    users = sorted({u for m in methods.values() for u in m})
    logger.info(
        "%s | methods=%s | users=%d", args.dataset, list(methods.keys()), len(users)
    )

    # Oracle union of all base retrievers.
    base_present = [r for r in BASE_RETRIEVERS if r in methods]
    oracle_topn = {}
    for u in users:
        seen = set()
        items = []
        for r in base_present:
            for it in methods[r].get(u, []):
                if it not in seen:
                    seen.add(it)
                    items.append(it)
        oracle_topn[u] = items
    methods["oracle_union"] = oracle_topn

    # Find best single retriever at the max K.
    max_k = max(ks)
    best_name = max(
        base_present, key=lambda r: _recall_at_k(methods[r], gold, users, max_k)
    )

    # === Overall Recall@K table ===
    print("\n=== Overall Recall@K ===")
    header = f"{'method':<20}" + "".join(f"{'R@' + str(k):>10}" for k in ks)
    print(header)

    display_order = [best_name, "fusion", "cara", "lhf", "oracle_union"]
    display_order = [m for m in display_order if m in methods]
    table_rows = []

    for name in display_order:
        cells = []
        row = {"method": name}
        for k in ks:
            r = _recall_at_k(methods[name], gold, users, k)
            cells.append(f"{r:.4f}")
            row[f"R@{k}"] = r
        print(f"{name:<20}" + "".join(f"{c:>10}" for c in cells))
        table_rows.append(row)

    # === Scenario-wise curves (item_new and warm) ===
    for scenario in ["is_item_new", "is_warm"]:
        if scenario not in test.columns:
            continue
        print(f"\n=== {scenario} Recall@K ===")
        print(header)
        for name in display_order:
            cells = []
            for k in ks:
                r, n = _scenario_recall(methods[name], gold, test, users, k, scenario)
                cells.append(f"{r:.4f}" if not np.isnan(r) else "-")
            print(f"{name:<20}" + "".join(f"{c:>10}" for c in cells))

    # Save results.
    outdir = os.path.join(args.output_dir, f"{args.dataset}-recallk-s{args.seed}")
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(table_rows).to_csv(
        os.path.join(outdir, "recallk_overall.csv"), index=False
    )

    for scenario in ["is_item_new", "is_warm"]:
        if scenario not in test.columns:
            continue
        scen_rows = []
        for name in display_order:
            row = {"method": name}
            for k in ks:
                r, _ = _scenario_recall(methods[name], gold, test, users, k, scenario)
                row[f"R@{k}"] = r
            scen_rows.append(row)
        pd.DataFrame(scen_rows).to_csv(
            os.path.join(outdir, f"recallk_{scenario.replace('is_', '')}.csv"),
            index=False,
        )

    logger.info("wrote results to %s/", outdir)


if __name__ == "__main__":
    main()
