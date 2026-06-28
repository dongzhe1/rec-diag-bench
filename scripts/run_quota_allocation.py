"""Grid-search quota allocation for multi-retriever candidate generation.

Given m retrievers and a total budget N, allocate q_j candidates from each
retriever j: q_1 + q_2 + ... + q_m = N. Searches over discrete quota vectors
on the validation set (or cross-validation on test if validation pools are
unavailable).

Compares: equal quota, CF-heavy, text-heavy, CARA, validation-tuned, oracle union.

POST-HOC & CPU-ONLY. Reads existing retriever pool CSVs.

Usage:
  python scripts/run_quota_allocation.py --dataset yelp-Philadelphia-Restaurants \
      --seed 42 --data_dir data
"""

from __future__ import annotations

import argparse
import itertools
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
QUOTA_RETRIEVERS = ["lightgcn", "itemknn", "sbert", "tfidf", "popularity"]


def _ranked(pool: pd.DataFrame, seed: int = 42) -> dict[int, list[int]]:
    rng = np.random.default_rng(seed)
    s = pool.assign(_t=rng.random(len(pool))).sort_values(
        ["user_idx", "score", "_t"], ascending=[True, False, False]
    )
    return {
        int(u): g["item_idx"].astype(int).tolist() for u, g in s.groupby("user_idx")
    }


def _load_pools(rdir: str, retrievers: list[str]) -> dict[str, dict[int, list[int]]]:
    out = {}
    for r in retrievers:
        p = os.path.join(rdir, f"{r}_scores.csv")
        if os.path.exists(p):
            out[r] = _ranked(pd.read_csv(p))
    return out


def _apply_quota(
    pools: dict[str, dict[int, list[int]]], quota: dict[str, int], users: list[int]
) -> dict[int, list[int]]:
    """Apply quota allocation: take top-q_j from each retriever j, deduplicate."""
    result = {}
    for u in users:
        seen = set()
        items = []
        for r, q in quota.items():
            if r not in pools:
                continue
            for it in pools[r].get(u, [])[:q]:
                if it not in seen:
                    seen.add(it)
                    items.append(it)
        result[u] = items
    return result


def _coverage(
    topn_by_user: dict[int, list[int]], gold: dict[int, int], users: list[int]
) -> float:
    hits = [1 if gold.get(u) in topn_by_user.get(u, []) else 0 for u in users]
    return float(np.mean(hits)) if hits else 0.0


def _coverage_detailed(topn_by_user, gold, test, users):
    flags = {
        c: dict(zip(test.user_idx.astype(int), test[c].astype(bool)))
        for c in SCENARIOS
        if c in test.columns
    }
    hit = {u: (gold.get(u) in topn_by_user.get(u, [])) for u in users}
    out = {"overall": float(np.mean([hit[u] for u in users]))}
    for c in SCENARIOS:
        if c in flags:
            us = [u for u in users if flags[c].get(u, False)]
            out[c.replace("is_", "")] = (
                float(np.mean([hit[u] for u in us])) if us else float("nan")
            )
    return out


def _generate_quota_grid(
    retrievers: list[str], N: int, step: int = 20
) -> list[dict[str, int]]:
    """Generate quota vectors that sum to ~N with the given step size."""
    m = len(retrievers)
    if m == 0:
        return []
    quotas = []
    vals = list(range(0, N + 1, step))
    for combo in itertools.product(vals, repeat=m):
        if sum(combo) == N:
            quotas.append(dict(zip(retrievers, combo)))
    if not quotas:
        equal = N // m
        quotas.append(dict(zip(retrievers, [equal] * m)))
    return quotas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--retrievers", default=",".join(QUOTA_RETRIEVERS))
    ap.add_argument(
        "--step",
        type=int,
        default=40,
        help="Quota grid step size (smaller = finer search, slower).",
    )
    ap.add_argument(
        "--regime_specific",
        action="store_true",
        help="Search separate quotas for user_cold vs warm users.",
    )
    args = ap.parse_args()

    rdir = os.path.join(
        args.output_dir, f"{args.dataset}-retrieval-s{args.seed}-N{args.N}"
    )
    vdir = os.path.join(
        args.output_dir, f"{args.dataset}-retrieval-valid-s{args.seed}-N{args.N}"
    )
    sp = os.path.join(args.data_dir, "processed", args.dataset, f"s{args.seed}")
    retr_names = [r.strip() for r in args.retrievers.split(",") if r.strip()]

    test = pd.read_csv(os.path.join(sp, "test.csv"))
    gold_test = dict(zip(test.user_idx.astype(int), test.item_idx.astype(int)))

    pools_test = _load_pools(rdir, retr_names)
    available = list(pools_test.keys())
    if len(available) < 2:
        logger.warning("need >=2 pools in %s; found %s", rdir, available)
        return
    logger.info(
        "%s | retrievers=%s | N=%d | step=%d", args.dataset, available, args.N, args.step
    )

    test_users = sorted({u for r in available for u in pools_test[r]})
    user_cold_set = (
        {
            int(u)
            for u in test.loc[test.get("is_user_cold", False).astype(bool), "user_idx"]
        }
        if "is_user_cold" in test.columns
        else set()
    )

    # Search on validation set (or fall back to cross-val on test).
    use_valid = os.path.isdir(vdir)
    if use_valid:
        valid = pd.read_csv(os.path.join(sp, "valid.csv"))
        gold_valid = dict(zip(valid.user_idx.astype(int), valid.item_idx.astype(int)))
        pools_valid = _load_pools(vdir, available)
        valid_users = sorted({u for r in pools_valid for u in pools_valid[r]})
        search_pools, search_gold, search_users = pools_valid, gold_valid, valid_users
        logger.info("searching on validation set (%d users)", len(valid_users))
    else:
        search_pools, search_gold, search_users = pools_test, gold_test, test_users
        logger.info("no validation pools found; searching on test (cross-validation style)")

    grid = _generate_quota_grid(available, args.N, args.step)
    logger.info("grid size: %d quota vectors", len(grid))

    if not args.regime_specific:
        best_cov, best_quota = -1.0, None
        for quota in grid:
            topn = _apply_quota(search_pools, quota, search_users)
            cov = _coverage(topn, search_gold, search_users)
            if cov > best_cov:
                best_cov, best_quota = cov, quota
        logger.info("best validation quota: %s -> coverage=%.4f", best_quota, best_cov)
    else:
        cold_users = [u for u in search_users if u in user_cold_set]
        warm_users = [u for u in search_users if u not in user_cold_set]
        best_cold_cov, best_cold_q = -1.0, None
        best_warm_cov, best_warm_q = -1.0, None
        for quota in grid:
            topn = _apply_quota(search_pools, quota, cold_users)
            cov = _coverage(topn, search_gold, cold_users) if cold_users else 0.0
            if cov > best_cold_cov:
                best_cold_cov, best_cold_q = cov, quota
            topn = _apply_quota(search_pools, quota, warm_users)
            cov = _coverage(topn, search_gold, warm_users) if warm_users else 0.0
            if cov > best_warm_cov:
                best_warm_cov, best_warm_q = cov, quota
        best_quota = {"cold": best_cold_q, "warm": best_warm_q}
        logger.info(
            "best cold quota: %s -> cov=%.4f (n=%d)", best_cold_q, best_cold_cov, len(cold_users)
        )
        logger.info(
            "best warm quota: %s -> cov=%.4f (n=%d)", best_warm_q, best_warm_cov, len(warm_users)
        )

    # Evaluate on test.
    print(f"\n=== Quota allocation results on test (coverage@{args.N}) ===")
    results = []

    # Equal quota.
    eq = args.N // len(available)
    eq_quota = dict.fromkeys(available, eq)
    eq_topn = _apply_quota(pools_test, eq_quota, test_users)
    eq_cov = _coverage_detailed(eq_topn, gold_test, test, test_users)
    results.append(("equal_quota", eq_cov, eq_quota))

    # CF-heavy.
    cf_quota = {}
    cf_names = {"lightgcn", "itemknn", "bpr", "sasrec"}
    cf_present = [r for r in available if r in cf_names]
    txt_present = [r for r in available if r not in cf_names]
    if cf_present and txt_present:
        cf_share = int(args.N * 0.7)
        txt_share = args.N - cf_share
        for r in cf_present:
            cf_quota[r] = cf_share // len(cf_present)
        for r in txt_present:
            cf_quota[r] = txt_share // len(txt_present)
        cf_topn = _apply_quota(pools_test, cf_quota, test_users)
        cf_cov = _coverage_detailed(cf_topn, gold_test, test, test_users)
        results.append(("cf_heavy_70", cf_cov, cf_quota))

    # Text-heavy.
    if cf_present and txt_present:
        txt_quota = {}
        txt_share2 = int(args.N * 0.7)
        cf_share2 = args.N - txt_share2
        for r in txt_present:
            txt_quota[r] = txt_share2 // len(txt_present)
        for r in cf_present:
            txt_quota[r] = cf_share2 // len(cf_present)
        txt_topn = _apply_quota(pools_test, txt_quota, test_users)
        txt_cov = _coverage_detailed(txt_topn, gold_test, test, test_users)
        results.append(("text_heavy_70", txt_cov, txt_quota))

    # Validation-tuned.
    if not args.regime_specific:
        tuned_topn = _apply_quota(pools_test, best_quota, test_users)
        tuned_cov = _coverage_detailed(tuned_topn, gold_test, test, test_users)
        results.append(("val_tuned", tuned_cov, best_quota))
    else:
        regime_topn = {}
        for u in test_users:
            q = best_quota["cold"] if u in user_cold_set else best_quota["warm"]
            regime_topn[u] = _apply_quota(pools_test, q, [u])[u]
        regime_cov = _coverage_detailed(regime_topn, gold_test, test, test_users)
        results.append(("val_tuned_regime", regime_cov, best_quota))

    # Existing baselines for comparison.
    for extra in ["fusion", "cara", "lhf"]:
        p = os.path.join(rdir, f"{extra}_scores.csv")
        if os.path.exists(p):
            ext_ranked = _ranked(pd.read_csv(p))
            ext_cov = _coverage_detailed(ext_ranked, gold_test, test, test_users)
            results.append((extra, ext_cov, None))

    # Oracle union.
    union_topn = {
        u: list({i for r in available for i in pools_test[r].get(u, [])[: args.N]})
        for u in test_users
    }
    union_cov = _coverage_detailed(union_topn, gold_test, test, test_users)
    results.append(("oracle_union", union_cov, None))

    # Print comparison table.
    cols = ["overall", "item_new", "item_cold", "long_tail", "user_cold", "warm"]
    print(f"{'method':<20}" + "".join(f"{c:>12}" for c in cols))
    for name, cov, _ in results:
        cells = []
        for c in cols:
            v = cov.get(c, float("nan"))
            cells.append(f"{v:.4f}" if not np.isnan(v) else "-")
        print(f"{name:<20}" + "".join(f"{c:>12}" for c in cells))

    # Regret.
    uni_o = union_cov["overall"]
    print("\n=== Regret to oracle ===")
    print(f"{'method':<20}{'regret':>10}{'regret%':>10}")
    for name, cov, _ in results:
        reg = uni_o - cov["overall"]
        regpct = reg / uni_o * 100 if uni_o > 0 else float("nan")
        print(f"{name:<20}{reg:>10.4f}{regpct:>9.1f}%")

    # Save.
    outdir = os.path.join(args.output_dir, f"{args.dataset}-quota-s{args.seed}")
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for name, cov, quota in results:
        row = {"method": name, **cov}
        row["regret"] = uni_o - cov["overall"]
        if (quota and not isinstance(quota, dict)) or (
            isinstance(quota, dict) and all(isinstance(v, int) for v in quota.values())
        ):
            row["quota"] = str(quota)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, "quota_results.csv"), index=False)
    logger.info("wrote %s/quota_results.csv", outdir)


if __name__ == "__main__":
    main()
