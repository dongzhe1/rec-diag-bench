"""Aggregate LHF / coverage-aware results across seeds with CIs.

Recomputes coverage@200 directly from each method's score file vs the test gold
(overall + per-scenario), for every seed that has outputs, then reports
mean / std / standard-error across seeds. Robust: no log parsing, no dependence
on the per-scenario lines printed during the runs.

Methods: best_single (the single standard retriever with the highest overall
coverage that seed), rrf (fusion), cara, lhf, regwt_n5.
Metrics: coverage@200 overall, item_new, warm.

Writes a per-seed long CSV and a mean±SE summary CSV.

Usage:
  python scripts/aggregate_multiseed.py --seeds 42,1,2026 \
      --out paper/data_raw/new_lhf_multiseed.csv
  python scripts/aggregate_multiseed.py --seeds 42   # validate on seed 42 alone
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATASETS = {
    "amazon-videogames": "VideoG",
    "mind-small": "MIND",
    "yelp-Philadelphia-Restaurants": "Yelp",
    "ml-20m": "ML-20M",
    "amazon-arts": "Arts",
}
STANDARD_RETRIEVERS = [
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
SCEN_FLAGS = {"item_new": "is_item_new", "warm": "is_warm"}


def _coverage(score_path: str, gold: dict, flags: dict) -> dict | None:
    """coverage@200 overall + per scenario from a {user_idx,item_idx,score} file."""
    if not os.path.exists(score_path):
        return None
    df = pd.read_csv(score_path)
    by_user = (
        df.groupby("user_idx")["item_idx"]
        .apply(lambda s: set(int(x) for x in s))
        .to_dict()
    )
    users = [u for u in gold if u in by_user]
    if not users:
        return None
    hit = {u: int(gold[u] in by_user[u]) for u in users}
    out = {"overall": float(np.mean([hit[u] for u in users]))}
    for scen, col in SCEN_FLAGS.items():
        su = [u for u in users if flags.get(col, {}).get(u, False)]
        out[scen] = float(np.mean([hit[u] for u in su])) if su else np.nan
    return out


def _best_single(rdir: str, gold: dict, flags: dict) -> dict | None:
    """The standard retriever with the highest overall coverage this seed."""
    best, best_cov = None, -1.0
    for r in STANDARD_RETRIEVERS:
        c = _coverage(os.path.join(rdir, f"{r}_scores.csv"), gold, flags)
        if c and c["overall"] > best_cov:
            best, best_cov = c, c["overall"]
    return best


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,2026")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--out", default="paper/data_raw/new_lhf_multiseed.csv")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    methods = {
        "best_single": None,  # special-cased
        "rrf": "fusion_scores.csv",
        "cara": "cara_scores.csv",
        "lhf": "lhf_scores.csv",
        "regwt_n5": "lhf_regwt_n5_scores.csv",
    }

    long_rows = []
    for seed in seeds:
        for ds, short in DATASETS.items():
            rdir = os.path.join(args.output_dir, f"{ds}-retrieval-s{seed}-N{args.N}")
            sp = os.path.join(args.data_dir, "processed", ds, f"s{seed}")
            test_f = os.path.join(sp, "test.csv")
            if not os.path.isdir(rdir) or not os.path.exists(test_f):
                continue
            test = pd.read_csv(test_f)
            gold = dict(zip(test.user_idx.astype(int), test.item_idx.astype(int)))
            flags = {
                col: dict(zip(test.user_idx.astype(int), test[col].astype(bool)))
                for col in SCEN_FLAGS.values()
                if col in test.columns
            }
            for method, fname in methods.items():
                cov = (
                    _best_single(rdir, gold, flags)
                    if method == "best_single"
                    else _coverage(os.path.join(rdir, fname), gold, flags)
                )
                if cov is None:
                    continue
                long_rows.append(
                    {
                        "seed": seed,
                        "domain": short,
                        "method": method,
                        **{
                            k: round(v, 4) if not np.isnan(v) else np.nan
                            for k, v in cov.items()
                        },
                    }
                )

    if not long_rows:
        print(
            "[multiseed] no outputs found for the requested seeds — nothing to aggregate"
        )
        return

    long = pd.DataFrame(long_rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    long.to_csv(args.out.replace(".csv", "_per_seed.csv"), index=False)

    # mean / std / sem across seeds
    agg_rows = []
    for (dom, method), g in long.groupby(["domain", "method"]):
        row = {
            "domain": dom,
            "method": method,
            "n_seeds": g["seed"].nunique(),
            "seeds": ",".join(str(s) for s in sorted(g["seed"].unique())),
        }
        for metric in ["overall", "item_new", "warm"]:
            vals = g[metric].dropna().to_numpy()
            if len(vals):
                row[f"{metric}_mean"] = round(float(vals.mean()), 4)
                row[f"{metric}_std"] = round(
                    float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4
                )
                row[f"{metric}_sem"] = round(
                    float(vals.std(ddof=1) / np.sqrt(len(vals)))
                    if len(vals) > 1
                    else 0.0,
                    4,
                )
        agg_rows.append(row)
    summary = (
        pd.DataFrame(agg_rows).sort_values(["domain", "method"]).reset_index(drop=True)
    )
    summary.to_csv(args.out, index=False)

    print(f"[multiseed] seeds={seeds}")
    print(f"[multiseed] wrote {args.out} ({len(summary)} rows) + per-seed CSV")
    show = summary[summary.method.isin(["best_single", "lhf", "regwt_n5"])]
    print(
        show[
            [
                "domain",
                "method",
                "overall_mean",
                "overall_std",
                "item_new_mean",
                "item_new_std",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
