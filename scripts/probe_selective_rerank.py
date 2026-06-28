"""Go/No-Go probe for confidence-gated selective LLM reranking.

We measured an oracle CF-vs-LLM router headroom (+0.08-0.13 natural). The METHOD question: can a
*serving-time-observable* gate (no gold, no LLM features) realize that headroom — i.e. decide per user
whether to trust CF or pay for the LLM? This probe trains a lightweight gate (cross-validated) on the
positive-controlled rerank subset and reports how much of the oracle router headroom it captures.

POST-HOC, CPU only: reuses outputs/<ds>-top200-s<seed>/{llm,<cf>}_scores.csv + the split.

Usage:
  python scripts/probe_selective_rerank.py --dataset yelp-Philadelphia-Restaurants --cf lightgcn --k 10
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


def ranked(path, seed=42):
    d = pd.read_csv(path)
    rng = np.random.default_rng(seed)
    d = d.assign(_t=rng.random(len(d))).sort_values(
        ["user_idx", "score", "_t"], ascending=[True, False, False]
    )
    return {int(u): g for u, g in d.groupby("user_idx")}


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument(
        "--cf",
        default="lightgcn",
        help="CF model to route against (lightgcn/bpr/itemknn/popularity)",
    )
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    rundir = os.path.join(
        args.output_dir, f"{args.dataset}-top{args.top_k}-s{args.seed}"
    )
    sp = os.path.join(args.data_dir, "processed", args.dataset, f"s{args.seed}")
    test = pd.read_csv(os.path.join(sp, "test.csv"))
    train = pd.read_csv(os.path.join(sp, "train.csv"))
    gold = dict(zip(test.user_idx.astype(int), test.item_idx.astype(int)))
    flags = {
        c: dict(zip(test.user_idx.astype(int), test[c].astype(bool)))
        for c in SCEN
        if c in test.columns
    }
    hist_len = train.groupby("user_idx").size().to_dict()

    llm = ranked(os.path.join(rundir, "llm_scores.csv"))
    cfp = os.path.join(rundir, f"{args.cf}_scores.csv")
    if not os.path.exists(cfp):
        raise SystemExit(f"CF scores not found: {cfp}")
    cf = ranked(cfp)
    users = sorted(set(llm) & set(cf) & set(gold))
    print(f"[probe-D2] {args.dataset} | cf={args.cf} | users={len(users)} | k={args.k}")

    rows, cf_hit, llm_hit = [], [], []
    for u in users:
        g = gold[u]
        cf_g = cf[u]
        llm_g = llm[u]
        cf_items = cf_g.item_idx.astype(int).tolist()
        llm_items = llm_g.item_idx.astype(int).tolist()
        ch = int(g in cf_items[: args.k])
        lh = int(g in llm_items[: args.k])
        cf_hit.append(ch)
        llm_hit.append(lh)
        # serving-observable features (NO gold, NO llm)
        s = cf_g.score.to_numpy(dtype=float)
        s_sorted = np.sort(s)[::-1]
        margin = float(s_sorted[0] - s_sorted[1]) if len(s_sorted) > 1 else 0.0
        p = np.exp(s - s.max())
        p = p / p.sum()
        ent = float(-(p * np.log(p + 1e-12)).sum())
        rows.append(
            {
                "user_cold": int(flags.get("is_user_cold", {}).get(u, False)),
                "hist_len": float(hist_len.get(u, 0)),
                "cf_margin": margin,
                "cf_top": float(s_sorted[0]),
                "cf_entropy": ent,
            }
        )

    X = pd.DataFrame(rows).fillna(0.0)
    cf_hit = np.array(cf_hit)
    llm_hit = np.array(llm_hit)
    oracle = np.maximum(cf_hit, llm_hit)
    # routing target: prefer LLM where it strictly helps
    y = (llm_hit > cf_hit).astype(int)

    print(f"\n  CF-only   recall@{args.k} = {cf_hit.mean():.4f}")
    print(f"  LLM-only  recall@{args.k} = {llm_hit.mean():.4f}")
    print(
        f"  oracle-router (max)      = {oracle.mean():.4f}   (headroom over CF = {oracle.mean() - cf_hit.mean():+.4f})"
    )

    if y.sum() == 0 or y.sum() == len(y):
        print(
            "\n[probe-D2] no routing decisions to learn (LLM never/always strictly better). "
            "Headroom is degenerate here; try another domain/CF."
        )
        return

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict

    route = cross_val_predict(
        HistGradientBoostingClassifier(
            max_iter=200, class_weight="balanced", random_state=args.seed
        ),
        X,
        y,
        cv=5,
        method="predict",
    )
    realized = np.where(route == 1, llm_hit, cf_hit)
    gate_rec = realized.mean()
    head = oracle.mean() - cf_hit.mean()
    frac = (gate_rec - cf_hit.mean()) / head if head > 1e-9 else float("nan")
    print(
        f"  learned-gate recall@{args.k}  = {gate_rec:.4f}   -> realizes {frac * 100:.0f}% of the oracle headroom"
    )

    # per-scenario realized vs cf
    print(f"\n  {'scenario':<11}{'n':>5}{'cf':>8}{'gate':>8}{'oracle':>8}")
    for c in SCEN:
        idx = [i for i, u in enumerate(users) if flags.get(c, {}).get(u, False)]
        if idx:
            print(
                f"  {c.replace('is_', ''):<11}{len(idx):>5}{cf_hit[idx].mean():>8.3f}"
                f"{realized[idx].mean():>8.3f}{oracle[idx].mean():>8.3f}"
            )

    print("\n--- VERDICT ---")
    if head < 0.02:
        print(
            f"  NO-GO: oracle headroom over CF is tiny ({head:+.4f}); little to route for on this domain."
        )
    elif frac == frac and frac >= 0.3 and gate_rec > cf_hit.mean() + 0.005:
        print(
            f"  GO: a serving-time gate captures {frac * 100:.0f}% of the oracle headroom "
            f"({cf_hit.mean():.3f}->{gate_rec:.3f}). Selective LLM reranking is learnable -> Direction 2."
        )
    else:
        print(
            f"  PARTIAL: gate realizes only {frac * 100:.0f}% of headroom; observable features weakly predict "
            f"when LLM helps. Add cheap signals or combine with Direction 1 (fix coverage first)."
        )
    print("=================================================")


if __name__ == "__main__":
    main()
