"""Build the CARA pool POST-HOC from existing retrieval component pools.

CARA only needs the component retrievers' top-N lists, which run_retrieval.py
--write_pools already wrote to disk as {retriever}_scores.csv. So we can produce
cara_scores.csv WITHOUT re-running the retrieval experiment (no retraining, no GPU)
— useful when the components are already on disk but the sweep predated CARA.

For each retrieval run dir it reads the component pools, reconstructs per-user
ranked lists, applies the regime-aware allocation (user_cold flag from test.csv),
and writes cara_scores.csv into the same dir. Then re-run analyze_oracle_ceiling.py.

Usage:
  python scripts/build_cara_from_pools.py --glob 'outputs/*-retrieval-s42-N200' \
      --data_dir data
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

from coldtail.experiments.retrieval_coverage import CARA_PROFILES, _topn_cara

logger = logging.getLogger(__name__)


def _ranked(pool: pd.DataFrame, seed: int = 42) -> dict[int, list[int]]:
    rng = np.random.default_rng(seed)
    s = pool.assign(_t=rng.random(len(pool))).sort_values(
        ["user_idx", "score", "_t"], ascending=[True, False, False]
    )
    return {
        int(u): g["item_idx"].astype(int).tolist() for u, g in s.groupby("user_idx")
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/*-retrieval-s*-N*")
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()

    comps = sorted({r for prof in CARA_PROFILES.values() for r in prof})
    dirs = sorted(d for d in glob.glob(args.glob) if os.path.isdir(d))
    if not dirs:
        print(f"no retrieval dirs matched {args.glob!r}")
        return

    for d in dirs:
        name = os.path.basename(d.rstrip("/"))
        m = re.match(r"(.+)-retrieval-s(\d+)-N(\d+)", name)
        if not m:
            print(f"  {name}: cannot parse dataset/seed/N — skipped")
            continue
        dataset, seed, N = m.group(1), int(m.group(2)), int(m.group(3))

        topn = {}
        for r in comps:
            p = os.path.join(d, f"{r}_scores.csv")
            if os.path.exists(p):
                topn[r] = _ranked(pd.read_csv(p))
        if len(topn) < 2:
            print(f"  {name}: <2 component pools present ({list(topn)}) — skipped")
            continue

        test_path = os.path.join(
            args.data_dir, "processed", dataset, f"s{seed}", "test.csv"
        )
        if not os.path.exists(test_path):
            print(f"  {name}: test.csv not found at {test_path} — skipped")
            continue
        test = pd.read_csv(test_path)
        user_cold = (
            {int(u) for u in test.loc[test["is_user_cold"].astype(bool), "user_idx"]}
            if "is_user_cold" in test.columns
            else set()
        )
        eval_users = sorted({u for r in topn.values() for u in r})

        cara = _topn_cara(topn, CARA_PROFILES, user_cold, eval_users, N)
        # flatten to scores frame (descending score reproduces the order)
        rows_u, rows_i, rows_s = [], [], []
        for u, arr in cara.items():
            n = len(arr)
            rows_u += [u] * n
            rows_i += [int(x) for x in arr]
            rows_s += list(range(n, 0, -1))
        pd.DataFrame({"user_idx": rows_u, "item_idx": rows_i, "score": rows_s}).to_csv(
            os.path.join(d, "cara_scores.csv"), index=False
        )
        print(
            f"  {name}: wrote cara_scores.csv | components={list(topn)} | "
            f"users={len(eval_users)} | user_cold={len(user_cold & set(eval_users))}"
        )

    print(
        "\nDone. Re-run: python scripts/analyze_oracle_ceiling.py "
        "--run_glob 'outputs/*-top200-s<seed>' --data_dir <data_dir> --k 10"
    )


if __name__ == "__main__":
    main()
