"""Per-domain coverage diagnostic table: why is content retrieval coverage on MIND so low?.

For each dataset it reports, from existing splits + retrieval pools:
  item_new share   fraction of test positives whose item never appeared in train
  avg hist len     mean #train interactions per evaluated user
  avg text len     mean character length of the gold item's text
  best CF cov@N     max coverage@N over CF retrievers (lightgcn/bpr/popularity)
  best content cov@N  max over content retrievers (sbert/tfidf/itemknn)
  catalogue        #items

A high item_new share on MIND substantiates the explanation that the bottleneck is
genuinely brand-new/topically-novel items, not a broken content retriever.

Usage:
  python scripts/diagnose_coverage.py --glob 'outputs/*-retrieval-s42-N200' \
      --data_dir data
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CF = ["lightgcn", "bpr", "popularity"]
CONTENT = ["sbert", "tfidf", "itemknn"]


def coverage(pool_path: str, gold: dict[int, int]) -> float | None:
    if not os.path.exists(pool_path):
        return None
    pool = pd.read_csv(pool_path)
    by_user = pool.groupby("user_idx")["item_idx"].apply(
        lambda s: set(int(x) for x in s)
    )
    hits = [1.0 if gold[u] in by_user.get(u, set()) else 0.0 for u in gold]
    return float(np.mean(hits)) if hits else None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/*-retrieval-s*-N*")
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(args.glob)):
        m = re.match(r"(.+)-retrieval-s(\d+)-N(\d+)", os.path.basename(d.rstrip("/")))
        if not m:
            continue
        dataset, seed, N = m.group(1), int(m.group(2)), int(m.group(3))
        sp = os.path.join(args.data_dir, "processed", dataset, f"s{seed}")
        if not os.path.exists(os.path.join(sp, "test.csv")):
            print(f"  {dataset}: split not found at {sp} — skipped")
            continue

        test = pd.read_csv(os.path.join(sp, "test.csv"))
        train = pd.read_csv(os.path.join(sp, "train.csv"))
        items = pd.read_csv(os.path.join(sp, "items_mapped.csv"))

        gold = dict(zip(test.user_idx.astype(int), test.item_idx.astype(int)))
        item_new = (
            float(test["is_item_new"].mean())
            if "is_item_new" in test.columns
            else float("nan")
        )
        hist = train.groupby("user_idx").size()
        avg_hist = float(hist.reindex(test.user_idx.unique()).fillna(0).mean())
        text = items.set_index("item_idx")["text"].astype(str)
        gold_text_len = float(
            text.reindex(list(gold.values())).fillna("").str.len().mean()
        )
        best_cf = max(
            (coverage(os.path.join(d, f"{r}_scores.csv"), gold) or 0) for r in CF
        )
        best_content = max(
            (coverage(os.path.join(d, f"{r}_scores.csv"), gold) or 0) for r in CONTENT
        )

        rows.append(
            {
                "dataset": dataset,
                "n_items": int(items.item_idx.max()) + 1,
                "n_test": len(test),
                "item_new_share": round(item_new, 3),
                "avg_hist_len": round(avg_hist, 1),
                "avg_gold_textlen": round(gold_text_len, 0),
                f"bestCF_cov@{N}": round(best_cf, 3),
                f"bestContent_cov@{N}": round(best_content, 3),
            }
        )

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n=== Per-domain coverage diagnostic ===")
    print(df.to_string(index=False))
    print(
        "\nRead: a high item_new_share (esp. MIND) explains why even content retrieval coverage is"
    )
    print(
        "low — the gold item never appeared in training, so no retriever (CF or content) can rank it"
    )
    print(
        "from interaction history; content helps only for cold-but-seen items on content-rich domains."
    )


if __name__ == "__main__":
    main()
