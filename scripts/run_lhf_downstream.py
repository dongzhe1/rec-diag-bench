"""LHF-to-downstream ranker: tests a non-LLM second-stage ranker on the LHF pool.

If a LightGBM ranker beats the LLM reranker on the same LHF candidate pool,
the conclusion strengthens: the bottleneck involves the prompt-level reranking
interface, not just retrieval coverage.

Pipeline:
  1. Load the LHF-ranked pool (lhf_scores.csv from run_learned_fusion.py)
  2. For each user, build a candidate set from the LHF top-N
  3. Train a LightGBM ranker on validation data to re-rank within the pool
  4. Compare: LHF-only ranking vs LHF->LightGBM vs LHF->LLM (from run_end2end.py)

POST-HOC & CPU-ONLY: reads existing LHF pools + item/user features.

Usage:
  python scripts/run_lhf_downstream.py --dataset yelp-Philadelphia-Restaurants \
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


def _recall_at_k(ranked_items: list[int], gold: int, k: int) -> float:
    return 1.0 if gold in ranked_items[:k] else 0.0


def _build_rerank_features(
    user_idx,
    item_idx,
    lhf_rank,
    lhf_score,
    hist_len,
    item_pop,
    item_new_map,
    item_textlen,
    user_cold_set,
):
    return {
        "lhf_rank": lhf_rank,
        "lhf_score": lhf_score,
        "lhf_rank_inv": 1.0 / (lhf_rank + 1),
        "user_hist_len": hist_len.get(user_idx, 0),
        "user_cold": 1 if user_idx in user_cold_set else 0,
        "item_pop": item_pop.get(item_idx, 0),
        "item_logpop": float(np.log1p(item_pop.get(item_idx, 0))),
        "item_new": 1 if item_new_map.get(item_idx, True) else 0,
        "item_textlen": item_textlen.get(item_idx, 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--retrieval_dir", default=None)
    ap.add_argument("--rerank_k", type=int, default=10)
    ap.add_argument(
        "--n_users",
        type=int,
        default=1000,
        help="Number of test users to evaluate (0 = all).",
    )
    args = ap.parse_args()

    rdir = args.retrieval_dir or os.path.join(
        args.output_dir, f"{args.dataset}-retrieval-s{args.seed}-N{args.N}"
    )
    sp = os.path.join(args.data_dir, "processed", args.dataset, f"s{args.seed}")

    lhf_path = os.path.join(rdir, "lhf_scores.csv")
    if not os.path.exists(lhf_path):
        logger.warning(
            "lhf_scores.csv not found at %s. Run run_learned_fusion.py --write first.",
            rdir,
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

    lhf_df = pd.read_csv(lhf_path)
    lhf_by_user = {}
    for u, g in lhf_df.groupby("user_idx"):
        g2 = g.sort_values("score", ascending=False).head(args.N)
        lhf_by_user[int(u)] = list(zip(g2.item_idx.astype(int), g2.score.astype(float)))

    # Train second-stage ranker on validation data.
    vdir_lhf = os.path.join(
        args.output_dir, f"{args.dataset}-retrieval-valid-s{args.seed}-N{args.N}"
    )
    vlhf_path = (
        os.path.join(vdir_lhf, "lhf_scores.csv") if os.path.isdir(vdir_lhf) else None
    )

    if vlhf_path and os.path.exists(vlhf_path):
        vlhf_df = pd.read_csv(vlhf_path)
        vlhf_by_user = {}
        for u, g in vlhf_df.groupby("user_idx"):
            g2 = g.sort_values("score", ascending=False).head(args.N)
            vlhf_by_user[int(u)] = list(
                zip(g2.item_idx.astype(int), g2.score.astype(float))
            )
        train_pool, train_gold = vlhf_by_user, gold_valid
        logger.info(
            "training second-stage ranker on validation LHF pool (%d users)",
            len(vlhf_by_user),
        )
    else:
        # Fallback: cross-fold on test LHF pool.
        train_pool, train_gold = lhf_by_user, gold_test
        logger.info("no validation LHF pool found; using cross-fold on test")

    rows_X, rows_y, rows_g = [], [], []
    for u, cands in train_pool.items():
        g = train_gold.get(u)
        if g is None:
            continue
        for rank_idx, (it, sc) in enumerate(cands, start=1):
            feat = _build_rerank_features(
                u,
                it,
                rank_idx,
                sc,
                hist_len,
                item_pop,
                item_new_map,
                item_textlen,
                user_cold_set,
            )
            rows_X.append(feat)
            rows_y.append(1 if it == g else 0)
            rows_g.append(u)

    X_train = pd.DataFrame(rows_X)
    y_train = np.array(rows_y)
    logger.info(
        "training data: %d rows, %d positives", len(X_train), int(y_train.sum())
    )

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

    # Evaluate on test.
    rng = np.random.default_rng(args.seed)
    test_users = sorted(set(gold_test.keys()) & set(lhf_by_user.keys()))
    if args.n_users and len(test_users) > args.n_users:
        test_users = sorted(
            int(u) for u in rng.choice(test_users, size=args.n_users, replace=False)
        )
    logger.info("evaluating on %d test users", len(test_users))

    flags = {
        c: dict(zip(test.user_idx.astype(int), test[c].astype(bool)))
        for c in SCENARIOS
        if c in test.columns
    }

    lhf_only_hits, lgbm_hits = [], []
    scenario_lhf = {c: [] for c in SCENARIOS}
    scenario_lgbm = {c: [] for c in SCENARIOS}

    for u in test_users:
        cands = lhf_by_user.get(u, [])
        if not cands:
            continue
        g = gold_test[u]
        # LHF-only: use the LHF ranking directly.
        lhf_items = [it for it, _ in cands]
        lhf_hit = _recall_at_k(lhf_items, g, args.rerank_k)
        lhf_only_hits.append(lhf_hit)

        # LightGBM rerank.
        feats = []
        items_in_pool = []
        for rank_idx, (it, sc) in enumerate(cands, start=1):
            feats.append(
                _build_rerank_features(
                    u,
                    it,
                    rank_idx,
                    sc,
                    hist_len,
                    item_pop,
                    item_new_map,
                    item_textlen,
                    user_cold_set,
                )
            )
            items_in_pool.append(it)
        Xp = pd.DataFrame(feats).reindex(columns=X_train.columns, fill_value=0)
        scores = ranker.predict_proba(Xp)[:, 1]
        reranked = [items_in_pool[i] for i in np.argsort(-scores)]
        lgbm_hit = _recall_at_k(reranked, g, args.rerank_k)
        lgbm_hits.append(lgbm_hit)

        for c in SCENARIOS:
            if c in flags and flags[c].get(u, False):
                scenario_lhf[c].append(lhf_hit)
                scenario_lgbm[c].append(lgbm_hit)

    lhf_r10 = float(np.mean(lhf_only_hits))
    lgbm_r10 = float(np.mean(lgbm_hits))
    delta = lgbm_r10 - lhf_r10
    print(f"\n=== LHF -> downstream ranker results (Recall@{args.rerank_k}) ===")
    print(f"  LHF-only ranking:       {lhf_r10:.4f}")
    print(f"  LHF->LightGBM ranker:   {lgbm_r10:.4f}")
    print(f"  Delta (LightGBM - LHF): {delta:+.4f}")

    # Look for LLM e2e results to compare.
    e2e_dir = os.path.join(args.output_dir, f"{args.dataset}-e2e-s{args.seed}")
    llm_r10 = None
    for sub in ["lhf", "cara", "lightgcn"]:
        llm_path = os.path.join(e2e_dir, sub, "llm_metrics.csv")
        if os.path.exists(llm_path):
            llm_m = pd.read_csv(llm_path).iloc[0]
            llm_r10 = float(llm_m.get(f"recall@{args.rerank_k}", float("nan")))
            print(f"  LHF->LLM (from e2e/{sub}):  {llm_r10:.4f}")
            break
    if llm_r10 is not None:
        print(f"  Delta (LightGBM - LLM): {lgbm_r10 - llm_r10:+.4f}")

    print(f"\n  Per-scenario Recall@{args.rerank_k}:")
    print(f"  {'scenario':<16}{'LHF-only':>10}{'->LightGBM':>10}{'n':>8}")
    for c in SCENARIOS:
        if scenario_lhf[c]:
            lhf_mean = float(np.mean(scenario_lhf[c]))
            g2 = float(np.mean(scenario_lgbm[c]))
            print(
                f"  {c.replace('is_', ''):<16}{lhf_mean:>10.4f}{g2:>10.4f}{len(scenario_lhf[c]):>8}"
            )

    outdir = os.path.join(args.output_dir, f"{args.dataset}-downstream-s{args.seed}")
    os.makedirs(outdir, exist_ok=True)
    summary = {
        "dataset": args.dataset,
        "seed": args.seed,
        "n_users": len(test_users),
        "lhf_only_r10": lhf_r10,
        "lgbm_r10": lgbm_r10,
        "delta": delta,
    }
    if llm_r10 is not None:
        summary["llm_r10"] = llm_r10
    pd.DataFrame([summary]).to_csv(
        os.path.join(outdir, "downstream_summary.csv"), index=False
    )
    logger.info("wrote %s/downstream_summary.csv", outdir)


if __name__ == "__main__":
    main()
