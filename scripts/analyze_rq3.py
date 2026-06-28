#!/usr/bin/env python3
"""analyze_rq3.py — settle RQ3 ("does graph evidence help, or is it a tail prior?")
from existing outputs/, with NO new model runs.

Two free, decisive tests on the score/metric files already on disk:

  (A) Tail-bonus decomposition.
      The GALA variants share one prompt and differ only in which signals are on:
        gala            = evidence + cooccur + tail-bonus
        gala_no_tail    = evidence + cooccur            (i.e. tail bonus OFF == bonus 0)
        gala_no_evidence= nothing  (== plain llm baseline)
      So:  evidence effect = R(gala_no_tail) - R(llm)      (graph evidence's own value)
           tail-prior effect = R(gala) - R(gala_no_tail)   (the +tail_bonus heuristic)
      If the evidence effect is <= 0 while the tail-prior effect carries the gain,
      "graph-aware evidence" is inert and the win is a tuned tail prior.

  (B) Natural-distribution reweighting.
      The rerank eval set is stratified to OVER-sample cold/tail. scenario_counts.csv
      gives the natural test prevalence. We reweight each model's per-scenario Recall@10
      by natural prevalence to estimate performance on realistic traffic. If GALA's
      aggregate win disappears (or flips) under natural weights, the headline was a
      composition artifact.

      Caveat: the four scenarios (user_cold / item_cold / long_tail / warm) are NOT a
      clean partition (an item can be item_cold AND long_tail), so the natural estimate
      is weighted-average, not exact. The per-scenario prevalences are printed so the
      story stays transparent regardless.

Usage:
  python scripts/analyze_rq3.py                       # default: *-top200-s42 (no hist)
  python scripts/analyze_rq3.py --glob 'outputs/*-top200-s42'
  python scripts/analyze_rq3.py --metric ndcg@10
Pure stdlib (no pandas) so it runs anywhere the outputs/ tree is.
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import re

logger = logging.getLogger(__name__)

SCEN = ["is_item_cold", "is_user_cold", "is_long_tail", "is_warm"]
CF = ["lightgcn", "popularity", "bpr", "itemknn"]


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def overall(rows, metric):
    return {
        r["model"]: float(r[metric]) for r in rows if r.get(metric) not in (None, "")
    }


def subgroup(rows, metric):
    out = {}
    for r in rows:
        try:
            out[(r["model"], r["scenario"])] = float(r[metric])
        except (KeyError, ValueError):
            pass
    return out


def natural_weights(counts_rows):
    w = {r["scenario"]: float(r["num_test_cases"]) for r in counts_rows}
    tot = sum(w.get(s, 0.0) for s in SCEN)
    return {s: (w.get(s, 0.0) / tot if tot else 0.0) for s in SCEN}, w


def natural_recall(model, sub, weights):
    """Weighted-average per-scenario recall by natural prevalence (estimate)."""
    num = 0.0
    den = 0.0
    for s in SCEN:
        v = sub.get((model, s))
        if v is not None and weights[s] > 0:
            num += v * weights[s]
            den += weights[s]
    return num / den if den else None


def fnum(v, w=8, p=3):
    return f"{v:>{w}.{p}f}" if v is not None else f"{'--':>{w}}"


def analyze_run(path, metric):
    name = os.path.basename(path.rstrip("/"))
    ov = overall(read_csv(os.path.join(path, "all_model_metrics.csv")), metric)
    sub = subgroup(
        read_csv(os.path.join(path, "all_model_subgroup_metrics.csv")), metric
    )
    counts = read_csv(os.path.join(path, "scenario_counts.csv"))
    if not ov or not counts:
        print(f"\n### {name}: missing metric/scenario files — skipped")
        return None
    wts, raw = natural_weights(counts)

    print("\n" + "=" * 84)
    print(f"### {name}   (metric = {metric})")
    print("=" * 84)

    # Natural prevalence
    tot = sum(raw.get(s, 0.0) for s in SCEN)
    print(
        "natural test prevalence:  "
        + "  ".join(
            f"{s.replace('is_', ''): <9}{(raw.get(s, 0) / tot if tot else 0):5.1%} (n={int(raw.get(s, 0))})"
            for s in SCEN
        )
    )

    # (A) decomposition
    llm = ov.get("llm")
    gnt = ov.get("gala_no_tail")
    gal = ov.get("gala")
    ev_eff = (gnt - llm) if (gnt is not None and llm is not None) else None
    tail_eff = (gal - gnt) if (gal is not None and gnt is not None) else None
    print("\n(A) decomposition (stratified overall):")
    print(f"    llm baseline            {fnum(llm)}")
    print(
        f"    gala_no_tail (evidence) {fnum(gnt)}   evidence effect vs baseline = {fnum(ev_eff, 7)}"
    )
    print(
        f"    gala (evidence + tail)  {fnum(gal)}   tail-prior effect           = {fnum(tail_eff, 7)}"
    )

    # (B) natural reweighting
    best_cf_strat = max(
        ((m, ov[m]) for m in CF if m in ov), key=lambda x: x[1], default=(None, None)
    )
    nat = {m: natural_recall(m, sub, wts) for m in ["llm", "gala_no_tail", "gala"] + CF}
    best_cf_nat = max(
        ((m, nat[m]) for m in CF if nat.get(m) is not None),
        key=lambda x: x[1],
        default=(None, None),
    )
    print("\n(B) natural-distribution-reweighted vs stratified-reported:")
    print(f"    {'model':<16}{'stratified':>12}{'natural':>12}")
    for m in ["llm", "gala", best_cf_strat[0]]:
        if m:
            print(f"    {m:<16}{fnum(ov.get(m), 12)}{fnum(nat.get(m), 12)}")
    print(f"    best CF = {best_cf_strat[0]}")

    # per-scenario GALA vs baseline vs best-CF
    print("\n    per-scenario Recall (gala / llm / best-CF):")
    for s in SCEN:
        g = sub.get(("gala", s))
        llm_val = sub.get(("llm", s))
        cfv = max((sub.get((m, s), -1) for m in CF), default=-1)
        cfv = cfv if cfv >= 0 else None
        print(
            f"      {s.replace('is_', ''):<11}{fnum(g)} /{fnum(llm_val)} /{fnum(cfv)}   (prevalence {wts[s]:.1%})"
        )

    # verdict
    print("\n    VERDICT:")
    if ev_eff is not None:
        print(
            f"      - graph evidence alone {'HELPS' if ev_eff > 0 else 'does NOT help'} "
            f"(evidence effect {ev_eff:+.3f})"
        )
    if ev_eff is not None and tail_eff is not None and tail_eff != 0:
        share = (
            tail_eff / (tail_eff + max(ev_eff, 0))
            if (tail_eff + max(ev_eff, 0))
            else 1.0
        )
        print(
            f"      - tail prior accounts for ~{share:.0%} of the GALA delta over baseline"
        )
    gn, ln, cn = nat.get("gala"), nat.get("llm"), best_cf_nat[1]
    if gn is not None and ln is not None:
        verdict = (
            "net-positive"
            if (gn > ln and (cn is None or gn >= cn))
            else "NOT net-positive"
        )
        print(
            f"      - on natural distribution GALA is {verdict} "
            f"(gala {fnum(gn, 0)} vs llm {fnum(ln, 0)} vs best-CF {fnum(cn, 0)})"
        )
    return {
        "name": name,
        "ev_eff": ev_eff,
        "tail_eff": tail_eff,
        "gala_nat": nat.get("gala"),
        "llm_nat": nat.get("llm"),
        "cf_nat": best_cf_nat[1],
    }


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        default="outputs/*-top200-s42",
        help="glob of run dirs to analyze (default main-table runs)",
    )
    ap.add_argument("--metric", default="recall@10")
    args = ap.parse_args()

    runs = sorted(
        d
        for d in glob.glob(args.glob)
        if os.path.isdir(d)
        and not re.search(r"-h\d+$", os.path.basename(d.rstrip("/")))
    )
    if not runs:
        print(f"no run dirs matched {args.glob!r}")
        return
    summ = [r for d in runs if (r := analyze_run(d, args.metric))]

    print("\n" + "=" * 84)
    print("CROSS-DATASET SUMMARY")
    print("=" * 84)
    print(
        f"{'run':<34}{'evid.eff':>10}{'tail.eff':>10}{'gala_nat':>10}{'llm_nat':>10}{'cf_nat':>9}"
    )
    for r in summ:
        print(
            f"{r['name']:<34}{fnum(r['ev_eff'], 10)}{fnum(r['tail_eff'], 10)}"
            f"{fnum(r['gala_nat'], 10)}{fnum(r['llm_nat'], 10)}{fnum(r['cf_nat'], 9)}"
        )
    print(
        "\nRead: evid.eff <= 0 everywhere  => graph evidence inert (RQ3 negative as built)."
    )
    print(
        "      gala_nat <= llm_nat/cf_nat  => GALA win was a stratification artifact."
    )


if __name__ == "__main__":
    main()
