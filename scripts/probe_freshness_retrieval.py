"""Go/No-Go probe for freshness / trend-aware candidate generation.

Hypothesis: for ephemeral-item domains (news), recency/popularity dynamics are a reachable retrieval
signal that history-based retrieval ignores. We build a recency-weighted popularity retriever
(score_i = sum over training interactions of exp(-(t_max - t)/tau)) and compare coverage@N to plain
popularity and to the best content/CF retriever (from the existing retrieval run).

POST-HOC, CPU only: uses train timestamps + the split, and reads existing coverage CSVs for context.
NOTE: item_new (zero training interactions) has score 0 here too -> recency cannot reach truly new
items without item publication timestamps (which the data lacks). The reachable cell is item_cold /
recent-but-seen. The probe makes that boundary explicit.

Usage:
  python scripts/probe_freshness_retrieval.py --dataset mind-small --N 200 --tau_frac 0.1
"""

from __future__ import annotations

import argparse
import logging
import os

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCEN = ["is_item_new", "is_item_cold", "is_long_tail", "is_user_cold", "is_warm"]


def coverage(global_order, seen_by_u, gold, users, flags, N):
    """coverage@N of a GLOBAL ranking (same for all users, minus their seen)."""
    hit = {}
    for u in users:
        seen = seen_by_u.get(u, set())
        pool, c = [], 0
        for it in global_order:
            if it in seen:
                continue
            pool.append(it)
            c += 1
            if c >= N:
                break
        hit[u] = gold[u] in pool
    out = {"overall": float(np.mean([hit[u] for u in users]))}
    for c in SCEN:
        us = [u for u in users if flags.get(c, {}).get(u, False)]
        out[c.replace("is_", "")] = (
            (float(np.mean([hit[u] for u in us])), len(us)) if us else (float("nan"), 0)
        )
    return out


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mind-small")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument(
        "--tau_frac",
        type=float,
        default=0.1,
        help="recency decay scale as fraction of timestamp span",
    )
    args = ap.parse_args()

    sp = os.path.join(args.data_dir, "processed", args.dataset, f"s{args.seed}")
    train = pd.read_csv(os.path.join(sp, "train.csv"))
    test = pd.read_csv(os.path.join(sp, "test.csv"))
    items = pd.read_csv(os.path.join(sp, "items_mapped.csv"))
    n_items = int(items.item_idx.max()) + 1
    gold = dict(zip(test.user_idx.astype(int), test.item_idx.astype(int)))
    flags = {
        c: dict(zip(test.user_idx.astype(int), test[c].astype(bool)))
        for c in SCEN
        if c in test.columns
    }
    seen_by_u = {
        int(u): set(map(int, g)) for u, g in train.groupby("user_idx")["item_idx"]
    }
    users = sorted(
        set(gold) & set(seen_by_u) | set(gold)
    )  # all test users (cold users may have no train)
    users = sorted(gold)

    t = train["timestamp"].to_numpy(dtype=float)
    tmax = t.max()
    span = max(t.max() - t.min(), 1.0)
    tau = args.tau_frac * span
    w = np.exp(-(tmax - t) / tau)
    it = train["item_idx"].to_numpy(dtype=int)

    pop = np.bincount(it, minlength=n_items).astype(float)  # plain popularity
    recpop = np.bincount(it, weights=w, minlength=n_items).astype(
        float
    )  # recency-weighted

    order_pop = list(np.argsort(-pop))
    order_rec = list(np.argsort(-recpop))

    cov_pop = coverage(order_pop, seen_by_u, gold, users, flags, args.N)
    cov_rec = coverage(order_rec, seen_by_u, gold, users, flags, args.N)

    # context: best content/CF coverage from the existing retrieval run
    ctx = {}
    rp = os.path.join(
        args.output_dir,
        f"{args.dataset}-retrieval-s{args.seed}-N{args.N}",
        "all_model_metrics.csv",
    )
    if os.path.exists(rp):
        d = pd.read_csv(rp)
        ccol = max(
            (c for c in d.columns if c.startswith("recall@")),
            key=lambda c: int(c.split("@")[1]),
        )
        m = dict(zip(d.model, d[ccol]))
        ctx["best_content"] = max(m.get(k, 0) for k in ("sbert", "tfidf"))
        ctx["best_cf"] = max(
            m.get(k, 0) for k in ("lightgcn", "bpr", "itemknn", "popularity")
        )

    cols = ["overall", "item_new", "item_cold", "long_tail", "user_cold", "warm"]

    def fmt(cov):
        out = [f"{cov['overall']:.4f}"]
        for c in cols[1:]:
            v = cov.get(c)
            out.append(f"{v[0]:.3f}" if isinstance(v, tuple) and v[1] else "-")
        return out

    print(
        f"\n=== coverage@{args.N} on {args.dataset} (global ranking; tau={args.tau_frac}*span) ==="
    )
    print(f"{'method':<16}" + "".join(f"{c:>11}" for c in cols))
    print(f"{'popularity':<16}" + "".join(f"{x:>11}" for x in fmt(cov_pop)))
    print(f"{'recency-pop':<16}" + "".join(f"{x:>11}" for x in fmt(cov_rec)))
    if ctx:
        print(
            f"\n  context (existing retrievers): best_content={ctx['best_content']:.3f} | best_cf={ctx['best_cf']:.3f}"
        )

    do = cov_rec["overall"] - cov_pop["overall"]
    dic = (
        (cov_rec["item_cold"][0] - cov_pop["item_cold"][0])
        if isinstance(cov_pop.get("item_cold"), tuple) and cov_pop["item_cold"][1]
        else float("nan")
    )
    print("\n--- VERDICT ---")
    print(f"  recency vs plain popularity: overall {do:+.4f} | item_cold {dic:+.4f}")
    inew = cov_rec.get("item_new")
    if isinstance(inew, tuple) and inew[1] and inew[0] < 0.005:
        print(
            "  (item_new ~0 as expected: zero-interaction items are unreachable without publication-time signal.)"
        )
    if do > 0.01 or (dic == dic and dic > 0.01):
        print(
            "  GO-ish: recency adds reachable coverage over plain popularity (esp. item_cold/recent) ->"
        )
        print(
            "          freshness is a real lever for ephemeral domains; build Direction 3 (add trend features,"
        )
        print(
            "          and item publication-time if obtainable to also reach item_new)."
        )
    else:
        print(
            "  NO-GO-ish: recency does not beat plain popularity here -> freshness signal is weak in this data"
        )
        print(
            "            (likely needs item publication timestamps, which the split lacks). Prefer Direction 1."
        )
    print("=================================================")


if __name__ == "__main__":
    main()
