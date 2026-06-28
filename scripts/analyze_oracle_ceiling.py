#!/usr/bin/env python3
"""analyze_oracle_ceiling.py — decide, from EXISTING outputs (no new training),
whether the diagnosis-driven method ideas have headroom worth building.

Three free oracle upper bounds, each gating one proposed method:

  (A) Oracle UNION retrieval coverage  ->  gates "complementarity-union retrieval"
      For each test item, is the gold in the union of several retrievers' pools
      (lightgcn ∪ itemknn ∪ popularity)? If union coverage >> best single
      retriever — especially on item_cold / long_tail — a union/ensemble
      retriever raises the ceiling for free. Needs retrieval pools written by
      run_retrieval.py --write_pools (outputs/*-retrieval-*-N*/{retriever}_scores.csv).

  (B) Oracle ROUTER reranking  ->  gates "selective / routed LLM reranking"
      Per test item, take the BETTER of {best-CF, llm} (a per-user OR of their
      hit@k). If the oracle router beats the best single model — and still wins
      after reweighting to the natural distribution — routing has real headroom.
      Needs the positive-controlled rank scores (outputs/*-top*-s*/{model}_scores.csv).

  (C) Cold-item CONTENT probe  ->  gates "LLM-as-cold-item-bridge"
      Do content/semantic retrievers (itemknn) cover item_cold / long_tail items
      that pure CF (lightgcn/bpr) misses? A clear gap is the green light that a
      content->CF embedding bridge can lift the 0-coverage cold cells. Read from
      the retrieval pools (A) or the retrieval all_model_subgroup_metrics.csv.

Verdict logic (printed per dataset + cross-dataset):
  big uplift on cold/tail cells  -> BUILD the corresponding method.
  ~no uplift even at the oracle   -> don't build; you have a deeper negative.

Usage:
  python scripts/analyze_oracle_ceiling.py
  python scripts/analyze_oracle_ceiling.py --run_glob 'outputs/*-top200-s42' --k 10
Pandas only (reads the same CSVs the rank/retrieval pipelines write).
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

# Mutually-exclusive scenario partition (priority high->low). Each test instance is
# assigned to exactly ONE bucket, so natural reweighting is a clean realistic-traffic
# estimate (not double-counting overlapping flags). item_new is included so coverage
# numbers are consistent with full-catalogue coverage (those items are uncoverable).
PARTITION = ["is_item_new", "is_item_cold", "is_long_tail", "is_user_cold", "is_warm"]
SCEN = PARTITION  # display all partition cells
CF_MODELS = [
    "lightgcn",
    "bpr",
    "itemknn",
    "popularity",
    "sasrec",
    "markov",
    "graph_aware",
]
RERANKER = "llm"
UNION_SET = [
    "lightgcn",
    "itemknn",
    "sbert",
    "popularity",
]  # complementary retrievers (A)
CONTENT_RETR = ["sbert", "tfidf", "itemknn"]  # content/semantic retrievers (C)
CF_RETR = ["lightgcn", "bpr"]  # pure CF retrievers (C)


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def parse_run(name: str):
    m = re.match(r"(.+)-top(\d+)-s(\d+)", name)
    return (m.group(1), int(m.group(3))) if m else (None, None)


def load_test(data_dir: str, dataset: str, seed: int) -> pd.DataFrame | None:
    path = os.path.join(data_dir, "processed", dataset, f"s{seed}", "test.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    keep = ["user_idx", "item_idx"] + [c for c in PARTITION if c in df.columns]
    return df[keep].copy()


def read_scores(run_dir: str, model: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, f"{model}_scores.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def ranked_by_user(
    scores: pd.DataFrame, maxk: int | None, seed: int = 42
) -> dict[int, list[int]]:
    """Per-user item list ranked by score desc, random tie-break (matches metrics.py)."""
    rng = np.random.default_rng(seed)
    s = scores.assign(_t=rng.random(len(scores))).sort_values(
        ["user_idx", "score", "_t"], ascending=[True, False, False]
    )
    out: dict[int, list[int]] = {}
    for u, g in s.groupby("user_idx"):
        items = g["item_idx"].astype(int).tolist()
        out[int(u)] = items[:maxk] if maxk else items
    return out


def hit_array(
    test: pd.DataFrame, ranked: dict[int, list[int]], k: int | None
) -> np.ndarray:
    """Per test-row hit@k (1 if gold in the user's top-k ranked items)."""
    hits = np.zeros(len(test), dtype=np.float64)
    for i, (u, gold) in enumerate(
        zip(test.user_idx.astype(int), test.item_idx.astype(int))
    ):
        lst = ranked.get(u, [])
        lst = lst[:k] if k else lst
        hits[i] = 1.0 if gold in lst else 0.0
    return hits


# --------------------------------------------------------------------------- #
# Aggregation: overall, per-scenario, natural-reweighted
# --------------------------------------------------------------------------- #
def assign_primary(df: pd.DataFrame) -> pd.Series:
    """Assign each row its single primary scenario by PARTITION priority."""
    prim = pd.Series(index=df.index, dtype=object)
    for col in PARTITION:
        if col in df.columns:
            prim[df[col].astype(bool) & prim.isna()] = col
    return prim.fillna("is_warm")


def partition_weights(full_test: pd.DataFrame) -> dict[str, float]:
    """Natural prevalence of each primary bucket on the FULL test set."""
    prim = assign_primary(full_test)
    tot = len(prim)
    vc = prim.value_counts()
    return {b: (float(vc.get(b, 0)) / tot if tot else 0.0) for b in PARTITION}


def summarize(test: pd.DataFrame, hits: np.ndarray, weights: dict[str, float]) -> dict:
    """Overall + per-primary-bucket recall + natural = bucket recall reweighted by
    `weights` (the full-test prevalence). On the full test set natural == overall;
    on a stratified subset natural un-stratifies it back to realistic traffic.
    """
    prim = assign_primary(test).to_numpy()
    out = {"overall": float(hits.mean())}
    num = den = 0.0
    for b in PARTITION:
        mask = prim == b
        v = float(hits[mask].mean()) if mask.any() else None
        out[b] = v
        if v is not None and weights.get(b, 0) > 0:
            num += v * weights[b]
            den += weights[b]
    out["natural"] = num / den if den else None
    return out


def fnum(v, w=8, p=3):
    return f"{v:>{w}.{p}f}" if v is not None else f"{'--':>{w}}"


# --------------------------------------------------------------------------- #
# (B) Oracle router on the positive-controlled rank scores
# --------------------------------------------------------------------------- #
def check_router(
    run_dir: str, full_test: pd.DataFrame, k: int, weights: dict
) -> dict | None:
    llm = read_scores(run_dir, RERANKER)
    if llm is None:
        print("    (B) router: no llm_scores.csv — skipped")
        return None
    # The LLM only scored the stratified rerank subset; restrict the comparison to
    # those users so CF-vs-LLM is apples-to-apples (then reweight to full-test prevalence).
    llm_users = set(llm.user_idx.astype(int))
    test = full_test[full_test.user_idx.astype(int).isin(llm_users)].reset_index(
        drop=True
    )
    llm_hit = hit_array(test, ranked_by_user(llm, k), k)

    cf_hits, cf_over = {}, {}
    for m in CF_MODELS:
        sc = read_scores(run_dir, m)
        if sc is not None:
            cf_hits[m] = hit_array(test, ranked_by_user(sc, k), k)
            cf_over[m] = cf_hits[m].mean()
    if not cf_hits:
        print("    (B) router: no CF scores — skipped")
        return None

    best_cf = max(cf_over, key=cf_over.get)
    oracle = np.maximum(llm_hit, cf_hits[best_cf])  # route llm vs best-CF
    oracle_all = np.maximum.reduce(
        [llm_hit] + list(cf_hits.values())
    )  # llm vs every CF

    s_llm = summarize(test, llm_hit, weights)
    s_cf = summarize(test, cf_hits[best_cf], weights)
    s_or = summarize(test, oracle, weights)
    s_all = summarize(test, oracle_all, weights)

    print(f"    (B) ORACLE ROUTER @{k}   (best-CF = {best_cf})")
    print(f"        {'cell':<12}{'best-CF':>9}{'llm':>9}{'oracle':>9}{'+vs best':>9}")
    for cell in ["overall", "natural"] + SCEN:
        cf, llm, ora = s_cf.get(cell), s_llm.get(cell), s_or.get(cell)
        up = (ora - max(cf or 0, llm or 0)) if ora is not None else None
        print(
            f"        {cell.replace('is_', ''):<12}{fnum(cf, 9)}{fnum(llm, 9)}{fnum(ora, 9)}{fnum(up, 9)}"
        )
    nat_up = (
        (s_or["natural"] - s_cf["natural"])
        if (s_or["natural"] and s_cf["natural"])
        else None
    )
    print(f"        oracle(llm∪allCF) natural = {fnum(s_all['natural'], 0)}")
    verdict = "HEADROOM" if (nat_up and nat_up > 0.02) else "thin"
    print(
        f"        -> routing headroom on natural dist: {fnum(nat_up, 0)}  [{verdict}]"
    )
    return {"router_nat_uplift": nat_up}


# --------------------------------------------------------------------------- #
# (A) Oracle union retrieval coverage + (C) content-vs-CF cold probe
# --------------------------------------------------------------------------- #
def find_retrieval_dir(out_root: str, dataset: str, seed: int) -> str | None:
    cands = glob.glob(os.path.join(out_root, f"{dataset}-retrieval-s{seed}-N*"))
    cands += [os.path.join(out_root, f"{dataset}-top*-s{seed}", "retrieval")]
    for d in cands:
        if glob.glob(os.path.join(d, "*_scores.csv")):
            return d
    return None


FUSION_VARIANTS = [
    ("fusion", "fus-rrf"),
    ("fusion_il", "fus-il"),
    ("cara", "cara"),
]  # (pool, label)


def check_union_and_content(
    retr_dir: str, test: pd.DataFrame, weights: dict
) -> dict | None:
    pools = {}
    for m in set(UNION_SET + CONTENT_RETR + CF_RETR + [f for f, _ in FUSION_VARIANTS]):
        sc = read_scores(retr_dir, m)
        if sc is not None:
            pools[m] = ranked_by_user(sc, None)  # full pool (coverage = membership)
    if len(pools) < 2:
        print(
            f"    (A/C) retrieval pools not found in {retr_dir} (need --write_pools) — skipped"
        )
        return None

    # (A) union coverage (oracle ceiling) + the REAL fusion retrievers
    single = {m: hit_array(test, pools[m], None) for m in pools}
    union_members = [m for m in UNION_SET if m in pools]
    union_hit = np.maximum.reduce([single[m] for m in union_members])
    s_union = summarize(test, union_hit, weights)
    best_single = max(union_members, key=lambda m: single[m].mean())
    s_best = summarize(test, single[best_single], weights)
    fusions = [(f, lbl) for f, lbl in FUSION_VARIANTS if f in single]
    s_fus = {f: summarize(test, single[f], weights) for f, _ in fusions}

    print(
        f"    (A) UNION coverage (oracle ceiling) vs FUSION (real)   (union = {'∪'.join(union_members)})"
    )
    hdr = (
        f"        {'cell':<12}{'best-1':>9}"
        + "".join(f"{lbl:>9}" for _, lbl in fusions)
        + f"{'union':>9}{'+vs best':>9}"
    )
    print(hdr + f"   (best-1={best_single})")
    for cell in ["overall", "natural"] + SCEN:
        b, u = s_best.get(cell), s_union.get(cell)
        up = (u - b) if (u is not None and b is not None) else None
        line = f"        {cell.replace('is_', ''):<12}{fnum(b, 9)}"
        line += "".join(fnum(s_fus[f].get(cell), 9) for f, _ in fusions)
        line += fnum(u, 9) + fnum(up, 9)
        print(line)
    cold_up = None
    if (
        s_union.get("is_item_cold") is not None
        and s_best.get("is_item_cold") is not None
    ):
        cold_up = s_union["is_item_cold"] - s_best["is_item_cold"]
    print(
        f"        -> item_cold coverage uplift from union (ceiling): {fnum(cold_up, 0)}"
    )

    # Report each real fusion; track the best (by realized fraction of the ceiling).
    fusion_realized = fusion_cold = None
    bn, un = s_best.get("natural"), s_union.get("natural")
    for f, lbl in fusions:
        sf = s_fus[f]
        real = (
            (sf.get("natural") - bn) / (un - bn)
            if None not in (bn, sf.get("natural"), un) and (un - bn) > 1e-9
            else None
        )
        cold = (
            sf.get("is_item_cold") - s_best.get("is_item_cold")
            if None not in (sf.get("is_item_cold"), s_best.get("is_item_cold"))
            else None
        )
        print(
            f"        -> {lbl}: realizes {(f'{real:.0%}' if real is not None else '--')} of ceiling on "
            f"natural; item_cold uplift {fnum(cold, 0)}"
        )
        if real is not None and (fusion_realized is None or real > fusion_realized):
            fusion_realized, fusion_cold = real, cold

    # (C) content vs CF on cold cells
    content = [m for m in CONTENT_RETR if m in pools]
    cf = [m for m in CF_RETR if m in pools]
    gap = None
    if content and cf:
        print("    (C) CONTENT vs CF coverage on cold cells:")
        print(f"        {'cell':<12}" + "".join(f"{m[:8]:>9}" for m in content + cf))
        for cell in ["is_item_cold", "is_long_tail"]:
            vals = {
                m: summarize(test, single[m], weights).get(cell) for m in content + cf
            }
            print(
                f"        {cell.replace('is_', ''):<12}"
                + "".join(fnum(vals[m], 9) for m in content + cf)
            )
            if cell == "is_item_cold":
                cmax = max((vals[m] or 0) for m in content)
                fmax = max((vals[m] or 0) for m in cf)
                gap = cmax - fmax
        v = "GREEN (content bridges cold)" if (gap and gap > 0.02) else "thin"
        print(f"        -> content advantage on item_cold: {fnum(gap, 0)}  [{v}]")
    return {
        "union_cold_uplift": cold_up,
        "content_cold_gap": gap,
        "fusion_realized": fusion_realized,
        "fusion_cold": fusion_cold,
    }


# --------------------------------------------------------------------------- #
def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_glob",
        default="outputs/*-top200-s42",
        help="glob of positive-controlled rank run dirs (for the router check)",
    )
    ap.add_argument(
        "--data_dir",
        default="data",
        help="root holding processed/<ds>/s<seed>/test.csv",
    )
    ap.add_argument(
        "--out_root", default="outputs", help="root to search for retrieval pools"
    )
    ap.add_argument(
        "--k", type=int, default=10, help="depth for the oracle router (recall@k)"
    )
    args = ap.parse_args()

    runs = sorted(
        d
        for d in glob.glob(args.run_glob)
        if os.path.isdir(d)
        and not re.search(r"-h\d+$", os.path.basename(d.rstrip("/")))
        and "-retrieval-" not in os.path.basename(d)
    )
    if not runs:
        print(f"no run dirs matched {args.run_glob!r}")
        return

    summary = []
    for run_dir in runs:
        name = os.path.basename(run_dir.rstrip("/"))
        dataset, seed = parse_run(name)
        if dataset is None:
            continue
        test = load_test(args.data_dir, dataset, seed)
        if test is None:
            print(f"\n### {name}: test.csv not found under {args.data_dir} — skipped")
            continue
        weights = partition_weights(test)

        print("\n" + "=" * 80)
        print(f"### {name}   (eval rows={len(test):,})")
        print("=" * 80)

        rec = {"name": name}
        r = check_router(run_dir, test, args.k, weights)
        if r:
            rec.update(r)
        retr_dir = find_retrieval_dir(args.out_root, dataset, seed)
        if retr_dir:
            print(f"    (A/C) retrieval pools from: {retr_dir}")
            c = check_union_and_content(retr_dir, test, weights)
            if c:
                rec.update(c)
        else:
            print(
                "    (A/C) no retrieval pool dir found — run run_retrieval.py --write_pools"
            )
        summary.append(rec)

    print("\n" + "=" * 80)
    print(
        "CROSS-DATASET SUMMARY  (uplift = oracle - best baseline; >0.02 = worth building)"
    )
    print("=" * 80)
    print(
        f"{'run':<34}{'router_nat':>11}{'union_cold':>11}{'content_cold':>13}"
        f"{'fus_real%':>10}{'fus_cold':>9}"
    )
    for r in summary:
        fr = r.get("fusion_realized")
        print(
            f"{r['name']:<34}{fnum(r.get('router_nat_uplift'), 11)}"
            f"{fnum(r.get('union_cold_uplift'), 11)}{fnum(r.get('content_cold_gap'), 13)}"
            f"{(f'{fr:>9.0%}' if fr is not None else f'{chr(45) * 2:>9}') + ' '}{fnum(r.get('fusion_cold'), 9)}"
        )
    print("\nRead:")
    print("  router_nat   > 0.02  => build selective/routed LLM reranking (Idea 2)")
    print("  union_cold   > 0.02  => UNION ceiling worth chasing (oracle upper bound)")
    print(
        "  content_cold > 0.02  => content embeddings bridge cold items (Idea 1 mechanism)"
    )
    print(
        "  fus_real%            => % of the union ceiling a REAL RRF fusion realizes (want high)"
    )
    print(
        "  fus_cold     > 0.02  => the REAL fusion lifts item_cold above best single — the method works"
    )
    print(
        "  all ~0               => no headroom even at the oracle; deepen the negative instead"
    )


if __name__ == "__main__":
    main()
