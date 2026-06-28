"""Downstream ranker PAIRED significance test.

Extends run_lhf_downstream.py with a paired comparison of three rankers over
the SAME users on the SAME LHF candidate pool:
  - LHF-only         : rank the pool by LHF score
  - LHF -> LightGBM  : a non-LLM learned second-stage ranker
  - LHF -> LLM       : the prompt-level LLM reranker (from the e2e run)

For each pair reports mean per-user Delta in Recall@k, a paired bootstrap 95% CI
(resample users), and McNemar discordant counts (b=gained, c=lost) with an exact
two-sided p-value. Eval users = exactly the LLM-scored subset so all three methods
are compared on identical users.

CPU-only, post-hoc, low memory (<= a few thousand users x <=200 candidates).

Usage:
  python scripts/run_downstream_paired.py --dataset yelp-Philadelphia-Restaurants \
      --seed 42 --data_dir data --output_dir outputs --k 10
"""

from __future__ import annotations

import argparse
import logging
import os
from math import comb

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SCENARIOS = ["is_item_new", "is_item_cold", "is_long_tail", "is_user_cold", "is_warm"]

# Paired statistics helpers


def mcnemar_two_sided_p(b: int, c: int) -> float:
    """Exact binomial two-sided p-value on discordant pairs (n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    x = min(b, c)
    tail = sum(comb(n, i) for i in range(x + 1)) / (2**n)
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean(a-b) with a paired bootstrap 95% CI (resample users with replacement)."""
    d = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return (
        float(d.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def _pair_report(name: str, hi: np.ndarray, lo: np.ndarray, seed: int) -> dict:
    """Hi - lo paired comparison (hi = the method under test, lo = baseline)."""
    mean, ci_lo, ci_hi = paired_bootstrap_ci(hi, lo, seed=seed)
    b = int(np.sum((hi == 1) & (lo == 0)))  # hi gained
    c = int(np.sum((hi == 0) & (lo == 1)))  # hi lost vs lo
    p = mcnemar_two_sided_p(b, c)
    return {
        "pair": name,
        "mean_delta": round(mean, 4),
        "ci_low": round(ci_lo, 4),
        "ci_high": round(ci_hi, 4),
        "gained_b": b,
        "lost_c": c,
        "mcnemar_p": round(p, 5),
    }


# Feature builder (same schema as run_lhf_downstream.py)


def _feat(
    u, it, rank, score, hist_len, item_pop, item_new_map, item_textlen, user_cold_set
):
    return {
        "lhf_rank": rank,
        "lhf_score": score,
        "lhf_rank_inv": 1.0 / (rank + 1),
        "user_hist_len": hist_len.get(u, 0),
        "user_cold": 1 if u in user_cold_set else 0,
        "item_pop": item_pop.get(it, 0),
        "item_logpop": float(np.log1p(item_pop.get(it, 0))),
        "item_new": 1 if item_new_map.get(it, True) else 0,
        "item_textlen": item_textlen.get(it, 0),
    }


def _pool_by_user(df: pd.DataFrame, N: int) -> dict:
    out = {}
    for u, g in df.groupby("user_idx"):
        g2 = g.sort_values("score", ascending=False).head(N)
        out[int(u)] = list(zip(g2.item_idx.astype(int), g2.score.astype(float)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    rdir = os.path.join(
        args.output_dir, f"{args.dataset}-retrieval-s{args.seed}-N{args.N}"
    )
    sp = os.path.join(args.data_dir, "processed", args.dataset, f"s{args.seed}")
    e2e_dir = os.path.join(args.output_dir, f"{args.dataset}-e2e-s{args.seed}")

    lhf_path = os.path.join(rdir, "lhf_scores.csv")
    if not os.path.exists(lhf_path):
        logger.warning(
            "missing %s -- run run_learned_fusion.py --write first", lhf_path
        )
        return

    # Locate the LLM-scored pool (prefer the LHF pool e2e run).
    llm_path = None
    for sub in ["lhf", "cara", "lightgcn"]:
        cand = os.path.join(e2e_dir, sub, "llm_scores.csv")
        if os.path.exists(cand):
            llm_path = cand
            llm_sub = sub
            break
    if llm_path is None:
        logger.warning(
            "no llm_scores.csv under %s/(lhf|cara|lightgcn) -- run e2e first", e2e_dir
        )
        return

    test = pd.read_csv(os.path.join(sp, "test.csv"))
    train = pd.read_csv(os.path.join(sp, "train.csv"))
    valid = pd.read_csv(os.path.join(sp, "valid.csv"))
    items = pd.read_csv(os.path.join(sp, "items_mapped.csv"))
    gold_test = dict(zip(test.user_idx.astype(int), test.item_idx.astype(int)))
    gold_valid = dict(zip(valid.user_idx.astype(int), valid.item_idx.astype(int)))
    hist_len = train.groupby("user_idx").size().to_dict()
    item_pop = train.groupby("item_idx").size().to_dict()
    item_new_map = {int(i): (item_pop.get(int(i), 0) == 0) for i in items.item_idx}
    item_textlen = items.set_index("item_idx")["text"].astype(str).str.len().to_dict()
    user_cold_set = (
        {
            int(u)
            for u in test.loc[test.get("is_user_cold", False).astype(bool), "user_idx"]
        }
        if "is_user_cold" in test.columns
        else set()
    )

    lhf_by_user = _pool_by_user(pd.read_csv(lhf_path), args.N)
    llm_by_user = _pool_by_user(pd.read_csv(llm_path), args.N)

    # Train LightGBM on the validation LHF pool (no test labels).
    vdir = os.path.join(
        args.output_dir, f"{args.dataset}-retrieval-valid-s{args.seed}-N{args.N}"
    )
    vlhf = os.path.join(vdir, "lhf_scores.csv")
    if os.path.exists(vlhf):
        train_pool, train_gold = _pool_by_user(pd.read_csv(vlhf), args.N), gold_valid
        logger.info("LightGBM trained on validation LHF pool (%d users)", len(train_pool))
    else:
        train_pool, train_gold = lhf_by_user, gold_test
        logger.warning(
            "no valid LHF pool found; training on test pool (cross-fit not applied)"
        )

    Xr, yr = [], []
    for u, cands in train_pool.items():
        g = train_gold.get(u)
        if g is None:
            continue
        for rank, (it, sc) in enumerate(cands, start=1):
            Xr.append(
                _feat(
                    u,
                    it,
                    rank,
                    sc,
                    hist_len,
                    item_pop,
                    item_new_map,
                    item_textlen,
                    user_cold_set,
                )
            )
            yr.append(1 if it == g else 0)
    X_train = pd.DataFrame(Xr)
    y_train = np.array(yr, dtype=np.int8)

    from sklearn.ensemble import HistGradientBoostingClassifier

    ranker = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=5,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=args.seed,
    )
    ranker.fit(X_train, y_train)

    # Evaluate on exactly the LLM-scored users (paired).
    eval_users = sorted(set(llm_by_user) & set(lhf_by_user) & set(gold_test))
    logger.info(
        "%d paired users (LLM-scored intersect LHF pool intersect gold) | e2e/%s",
        len(eval_users),
        llm_sub,
    )

    flags = {
        c: dict(zip(test.user_idx.astype(int), test[c].astype(bool)))
        for c in SCENARIOS
        if c in test.columns
    }

    rows = []
    for u in eval_users:
        g = gold_test[u]
        cands = lhf_by_user[u]
        lhf_items = [it for it, _ in cands]
        lhf_hit = int(g in lhf_items[: args.k])

        feats = [
            _feat(
                u,
                it,
                rk,
                sc,
                hist_len,
                item_pop,
                item_new_map,
                item_textlen,
                user_cold_set,
            )
            for rk, (it, sc) in enumerate(cands, start=1)
        ]
        Xp = pd.DataFrame(feats).reindex(columns=X_train.columns, fill_value=0)
        order = np.argsort(-ranker.predict_proba(Xp)[:, 1])
        lgbm_items = [lhf_items[i] for i in order]
        lgbm_hit = int(g in lgbm_items[: args.k])

        llm_items = [it for it, _ in llm_by_user[u]]
        llm_hit = int(g in llm_items[: args.k])

        row = {
            "user_idx": u,
            "lhf_hit": lhf_hit,
            "lgbm_hit": lgbm_hit,
            "llm_hit": llm_hit,
        }
        for c in SCENARIOS:
            row[c] = int(bool(flags.get(c, {}).get(u, False)))
        rows.append(row)

    per_user = pd.DataFrame(rows)
    if per_user.empty:
        logger.warning("no overlapping users -- nothing to test")
        return

    lhf = per_user.lhf_hit.to_numpy()
    lgbm = per_user.lgbm_hit.to_numpy()
    llm = per_user.llm_hit.to_numpy()

    reports = [
        _pair_report("lgbm_vs_lhf", lgbm, lhf, args.seed),
        _pair_report("llm_vs_lhf", llm, lhf, args.seed),
        _pair_report("lgbm_vs_llm", lgbm, llm, args.seed),
    ]
    for r in reports:
        r.update(
            {
                "dataset": args.dataset,
                "seed": args.seed,
                "k": args.k,
                "n_users": len(per_user),
                "lhf_r@k": round(float(lhf.mean()), 4),
                "lgbm_r@k": round(float(lgbm.mean()), 4),
                "llm_r@k": round(float(llm.mean()), 4),
            }
        )

    summary = pd.DataFrame(reports)[
        [
            "dataset",
            "seed",
            "k",
            "n_users",
            "pair",
            "lhf_r@k",
            "lgbm_r@k",
            "llm_r@k",
            "mean_delta",
            "ci_low",
            "ci_high",
            "gained_b",
            "lost_c",
            "mcnemar_p",
        ]
    ]

    print(
        f"\n=== Paired downstream significance (Recall@{args.k}, n={len(per_user)}) ==="
    )
    print(
        f"  LHF-only={lhf.mean():.4f}  LightGBM={lgbm.mean():.4f}  LLM={llm.mean():.4f}"
    )
    print(
        summary[
            [
                "pair",
                "mean_delta",
                "ci_low",
                "ci_high",
                "gained_b",
                "lost_c",
                "mcnemar_p",
            ]
        ].to_string(index=False)
    )

    outdir = os.path.join(args.output_dir, f"{args.dataset}-downstream-s{args.seed}")
    os.makedirs(outdir, exist_ok=True)
    summary.to_csv(os.path.join(outdir, "downstream_paired_summary.csv"), index=False)
    per_user.to_csv(os.path.join(outdir, "downstream_paired_per_user.csv"), index=False)
    logger.info("wrote %s/downstream_paired_summary.csv (+ per_user.csv)", outdir)


if __name__ == "__main__":
    main()
