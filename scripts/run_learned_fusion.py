"""LHF — Learned Hybrid Fusion: a practical retrieval-side fusion baseline.

Trains a lightweight learning-to-rank model over the UNION pool of the base
retrievers and re-ranks it to top-N, then compares its coverage@N to (i) the
best single retriever, (ii) the oracle union ceiling, and (iii) heuristic
fusions (RRF, CARA).

POST-HOC & GPU-FREE: reads the component ``{retriever}_scores.csv`` pools that
``run_retrieval.py`` already wrote. Features are retriever ranks/scores plus
user/item metadata. Labels:

  --label_source test_xfold   (default) GroupKFold over test users: train on 4/5,
      predict the held-out 1/5, rotate. No per-user leakage.
  --label_source valid        train on a VALIDATION retrieval dir (gold = valid
      positives), apply to the test union pool.

Ablation flags:
  --rank_only             keep only rank features
  --score_only            keep only score features
  --no_metadata           drop ALL user/item metadata, keep only retriever features
  --no_cold_metadata      drop item_new + item_pop/logpop only
  --no_text_retrievers    exclude text-based retrievers (sbert, tfidf) from union
  --no_cf_retrievers      exclude CF-based retrievers (lightgcn, bpr, sasrec) from union
  --regime_weights W      coverage-aware training: upweight positives by regime
                          (e.g. "item_new=5,item_cold=3,long_tail=2")

Usage:
  python scripts/run_learned_fusion.py --dataset yelp-Philadelphia-Restaurants \
      --seed 42 --data_dir data --write

CPU-only (post-hoc analysis on existing pool files).
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

# Base retrievers whose pools feed the union (exclude derived fusions cara/fusion/fusion_il).
BASE_RETRIEVERS = [
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
TEXT_RETRIEVERS = {"sbert", "tfidf"}
CF_RETRIEVERS = {"lightgcn", "bpr", "sasrec"}
SCENARIOS = ["is_item_new", "is_item_cold", "is_long_tail", "is_user_cold", "is_warm"]


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


def _load_pool_scores(rdir: str, retrievers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for r in retrievers:
        p = os.path.join(rdir, f"{r}_scores.csv")
        if os.path.exists(p):
            out[r] = pd.read_csv(p)
    return out


def _build_score_lookup(
    raw_pools: dict[str, pd.DataFrame],
) -> dict[str, dict[int, dict[int, float]]]:
    """Build {retriever: {user: {item: score}}} lookup avoiding tuple-key overhead."""
    out = {}
    for r, df in raw_pools.items():
        by_user: dict[int, dict[int, float]] = {}
        for u, grp in df.groupby("user_idx", sort=False):
            by_user[int(u)] = dict(
                zip(grp.item_idx.astype(int), grp.score.astype(float))
            )
        out[r] = by_user
    return out


def _build_features(
    users,
    pools,
    retr,
    N,
    hist_len,
    item_pop,
    item_new,
    item_textlen,
    user_cold_set,
    gold,
    score_lookup=None,
):
    """One row per (user, candidate-in-union-pool). Returns X (float32 DataFrame), y, groups, ui.

    Memory-efficient: builds per-user float32 numpy chunks instead of Python dicts,
    avoiding the ~3KB-per-row overhead that causes OOM on large datasets. Peak memory
    is ~2x the final feature matrix size rather than 10x.
    """
    big = float(N + 1)
    n_retr = len(retr)
    has_scores = score_lookup is not None
    # Column layout (all float32):
    #   0..n_retr-1         rank_*        (default = N+1)
    #   n_retr..2n_retr-1   has_*         (default = 0)
    #   2n_retr..3n_retr-1  score_*       (default = 0)
    #   3n_retr+0           n_hits
    #   3n_retr+1           min_rank
    #   3n_retr+2           mean_rank
    #   3n_retr+3           max_score
    #   3n_retr+4           mean_score
    #   3n_retr+5..+10      user_hist_len, user_cold, item_pop, item_logpop, item_new, item_textlen
    BASE = 3 * n_retr
    n_feat = BASE + 11

    x_chunks: list[np.ndarray] = []
    all_u: list[int] = []
    all_i: list[int] = []
    all_y: list[int] = []

    for u in users:
        g = gold.get(u)
        hl = float(hist_len.get(u, 0))
        uc = 1.0 if u in user_cold_set else 0.0
        sc_u = {
            ri: (score_lookup[r].get(u, {}) if has_scores else {})
            for ri, r in enumerate(retr)
        }

        # Build union: item -> {retr_idx: rank}
        union: dict[int, dict[int, int]] = {}
        for ri, r in enumerate(retr):
            for rk, it in enumerate(pools[r].get(u, [])[:N], 1):
                if it not in union:
                    union[it] = {}
                union[it][ri] = rk

        n_u = len(union)
        if n_u == 0:
            continue

        arr = np.empty((n_u, n_feat), dtype=np.float32)
        arr[:, :n_retr] = big  # rank default
        arr[:, n_retr:BASE] = 0.0  # has_ and score_ default

        for row, (it, rank_map) in enumerate(union.items()):
            all_u.append(u)
            all_i.append(it)
            all_y.append(1 if it == g else 0)

            rk_min = big
            rk_sum = 0.0
            sc_max = 0.0
            sc_sum = 0.0
            n_present = 0
            for ri, rk in rank_map.items():
                arr[row, ri] = rk
                arr[row, n_retr + ri] = 1.0
                sc = float(sc_u[ri].get(it, 0.0))
                arr[row, 2 * n_retr + ri] = sc
                rk_min = min(rk_min, rk)
                rk_sum += rk
                sc_max = max(sc_max, sc)
                sc_sum += sc
                n_present += 1

            arr[row, BASE] = n_present
            arr[row, BASE + 1] = rk_min if n_present else big
            arr[row, BASE + 2] = rk_sum / n_present if n_present else big
            arr[row, BASE + 3] = sc_max
            arr[row, BASE + 4] = sc_sum / n_present if n_present else 0.0
            pop = item_pop.get(it, 0)
            arr[row, BASE + 5] = hl
            arr[row, BASE + 6] = uc
            arr[row, BASE + 7] = pop
            arr[row, BASE + 8] = np.log1p(pop)
            arr[row, BASE + 9] = 0.0 if pop > 0 else 1.0  # is_new indicator
            arr[row, BASE + 10] = item_textlen.get(it, 0)

        x_chunks.append(arr)

    X_np = (
        np.concatenate(x_chunks, axis=0)
        if x_chunks
        else np.empty((0, n_feat), dtype=np.float32)
    )
    del x_chunks

    col_names = (
        [f"rank_{r}" for r in retr]
        + [f"has_{r}" for r in retr]
        + [f"score_{r}" for r in retr]
        + [
            "n_hits",
            "min_rank",
            "mean_rank",
            "max_score",
            "mean_score",
            "user_hist_len",
            "user_cold",
            "item_pop",
            "item_logpop",
            "item_new",
            "item_textlen",
        ]
    )
    X = pd.DataFrame(X_np, columns=col_names)
    del X_np  # DataFrame shares the buffer; explicit del prevents accidental double-ref

    y = np.array(all_y, dtype=np.int8)
    groups = np.array(all_u, dtype=np.int32)
    ui = list(zip(all_u, all_i))
    return X, y, groups, ui


def _coverage(topn_by_user, gold, test, users):
    """Coverage@N overall + per primary scenario."""
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
                (float(np.mean([hit[u] for u in us])), len(us))
                if us
                else (float("nan"), 0)
            )
    return out


def _ablation_tag(args) -> str:
    """Build a short tag for the ablation mode (for output filenames and log lines)."""
    parts = []
    if args.no_metadata:
        parts.append("nometa")
    if args.no_cold_metadata:
        parts.append("nocold")
    if args.rank_only:
        parts.append("rankonly")
    if args.score_only:
        parts.append("scoreonly")
    if args.no_text_retrievers:
        parts.append("notxt")
    if args.no_cf_retrievers:
        parts.append("nocf")
    if args.regime_weights:
        # Encode weights into tag so each variant gets a distinct filename.
        abbrev = {
            "item_new": "n",
            "item_cold": "c",
            "long_tail": "t",
            "user_cold": "u",
            "warm": "w",
        }
        segs = []
        for part in args.regime_weights.split(","):
            if "=" in part:
                k, v = part.strip().split("=", 1)
                segs.append(f"{abbrev.get(k.strip(), k.strip())}{int(float(v))}")
        parts.append("regwt_" + "".join(segs) if segs else "regwt")
    return "_".join(parts) if parts else ""


def _parse_regime_weights(s: str) -> dict[str, float]:
    """Parse 'item_new=5,item_cold=3,long_tail=2' into a dict."""
    if not s:
        return {}
    out = {}
    for part in s.split(","):
        k, v = part.strip().split("=")
        key = k.strip() if k.strip().startswith("is_") else f"is_{k.strip()}"
        out[key] = float(v.strip())
    return out


def _build_sample_weights(y, groups, ui, test, regime_weights):
    """Build per-sample weights that upweight positives from cold regimes."""
    if not regime_weights:
        return None
    flags = {}
    for col in SCENARIOS:
        if col in test.columns:
            flags[col] = dict(zip(test.user_idx.astype(int), test[col].astype(bool)))
    w = np.ones(len(y), dtype=np.float64)
    for i, (u, _it) in enumerate(ui):
        if y[i] == 0:
            continue
        for scen, mult in regime_weights.items():
            if scen in flags and flags[scen].get(u, False):
                w[i] = mult
                break
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--N", type=int, default=200, help="pool / top-N budget")
    ap.add_argument("--retrieval_dir", default=None)
    ap.add_argument(
        "--valid_retrieval_dir", default=None, help="for --label_source valid"
    )
    ap.add_argument("--retrievers", default=",".join(BASE_RETRIEVERS))
    ap.add_argument(
        "--label_source", choices=["test_xfold", "valid"], default="test_xfold"
    )
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument(
        "--max_train_users",
        type=int,
        default=12000,
        help="Cap on validation users used to TRAIN the fusion (--label_source valid). "
        "0 = no cap.",
    )
    ap.add_argument("--model", choices=["hgb", "logreg"], default="hgb")
    ap.add_argument(
        "--no_metadata",
        action="store_true",
        help="Ablation: drop ALL user/item metadata, keep only retriever features.",
    )
    ap.add_argument(
        "--no_cold_metadata",
        action="store_true",
        help="Ablation: drop item_new + item_pop/logpop only (cold-start signal test).",
    )
    ap.add_argument(
        "--rank_only",
        action="store_true",
        help="Ablation: keep only rank features (rank_*, has_*, n_hits, min/mean_rank).",
    )
    ap.add_argument(
        "--score_only",
        action="store_true",
        help="Ablation: keep only score features (score_*, max/mean_score).",
    )
    ap.add_argument(
        "--no_text_retrievers",
        action="store_true",
        help="Ablation: exclude text retrievers (sbert, tfidf) from the union.",
    )
    ap.add_argument(
        "--no_cf_retrievers",
        action="store_true",
        help="Ablation: exclude CF retrievers (lightgcn, bpr, sasrec) from the union.",
    )
    ap.add_argument(
        "--regime_weights",
        default="",
        help="Coverage-aware training: upweight positives by regime. "
        "Format: 'item_new=5,item_cold=3,long_tail=2'. Default: no weighting.",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="Write lhf_scores.csv into the retrieval dir.",
    )
    args = ap.parse_args()

    rdir = args.retrieval_dir or os.path.join(
        args.output_dir, f"{args.dataset}-retrieval-s{args.seed}-N{args.N}"
    )
    sp = os.path.join(args.data_dir, "processed", args.dataset, f"s{args.seed}")
    retr_names = [r.strip() for r in args.retrievers.split(",") if r.strip()]

    if args.no_text_retrievers:
        excluded = TEXT_RETRIEVERS & set(retr_names)
        retr_names = [r for r in retr_names if r not in TEXT_RETRIEVERS]
        logger.info("--no_text_retrievers: excluded %s", excluded)
    if args.no_cf_retrievers:
        excluded = CF_RETRIEVERS & set(retr_names)
        retr_names = [r for r in retr_names if r not in CF_RETRIEVERS]
        logger.info("--no_cf_retrievers: excluded %s", excluded)

    test = pd.read_csv(os.path.join(sp, "test.csv"))
    train = pd.read_csv(os.path.join(sp, "train.csv"))
    items = pd.read_csv(os.path.join(sp, "items_mapped.csv"))
    gold = dict(zip(test.user_idx.astype(int), test.item_idx.astype(int)))
    hist_len = train.groupby("user_idx").size().to_dict()
    item_pop = train.groupby("item_idx").size().to_dict()
    item_new = {int(i): (item_pop.get(int(i), 0) == 0) for i in items.item_idx}
    item_textlen = items.set_index("item_idx")["text"].astype(str).str.len().to_dict()
    user_cold_set = (
        {
            int(u)
            for u in test.loc[test.get("is_user_cold", False).astype(bool), "user_idx"]
        }
        if "is_user_cold" in test.columns
        else set()
    )

    pools = _load_pools(rdir, retr_names)
    if len(pools) < 2:
        logger.warning("need >=2 component pools in %s; found %s", rdir, list(pools))
        return
    retr = list(pools)

    raw_pools = _load_pool_scores(rdir, retr)
    score_lookup = _build_score_lookup(raw_pools)
    del raw_pools

    users = sorted({u for r in retr for u in pools[r]})
    abl_tag = _ablation_tag(args)
    logger.info(
        "%s | dir=%s | retrievers=%s | users=%d | N=%s%s",
        args.dataset,
        rdir,
        retr,
        len(users),
        args.N,
        f" | ablation={abl_tag}" if abl_tag else "",
    )

    X, y, groups, ui = _build_features(
        users,
        pools,
        retr,
        args.N,
        hist_len,
        item_pop,
        item_new,
        item_textlen,
        user_cold_set,
        gold,
        score_lookup=score_lookup,
    )

    META_COLS = [
        "user_hist_len",
        "user_cold",
        "item_pop",
        "item_logpop",
        "item_new",
        "item_textlen",
    ]
    COLD_META_COLS = ["item_new", "item_pop", "item_logpop"]
    RANK_COLS = [
        c
        for c in X.columns
        if c.startswith("rank_")
        or c.startswith("has_")
        or c in ("n_hits", "min_rank", "mean_rank")
    ]
    SCORE_COLS = [
        c
        for c in X.columns
        if c.startswith("score_") or c in ("max_score", "mean_score")
    ]

    if args.rank_only:
        X = X[RANK_COLS]
        logger.info("--rank_only ABLATION: %d rank features only", X.shape[1])
    elif args.score_only:
        X = X[SCORE_COLS]
        logger.info("--score_only ABLATION: %d score features only", X.shape[1])
    elif args.no_metadata:
        X = X.drop(columns=[c for c in META_COLS if c in X.columns])
        logger.info(
            "--no_metadata ABLATION: dropped user/item metadata -> %d features",
            X.shape[1],
        )
    elif args.no_cold_metadata:
        X = X.drop(columns=[c for c in COLD_META_COLS if c in X.columns])
        logger.info(
            "--no_cold_metadata ABLATION: dropped item_new/pop -> %d features",
            X.shape[1],
        )

    regime_weights = _parse_regime_weights(args.regime_weights)
    sample_weights = _build_sample_weights(y, groups, ui, test, regime_weights)
    if regime_weights:
        logger.info("--regime_weights: %s", regime_weights)

    logger.info(
        "union-pool rows=%d | positives(gold in union)=%d (%.2f%% of rows) | features=%d",
        len(X),
        int(y.sum()),
        y.mean() * 100,
        X.shape[1],
    )

    is_pipeline = args.model == "logreg"

    def _new_model():
        if args.model == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            return make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            )
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=args.seed,
        )

    def _fit_model(m, X_tr, y_tr, sw=None):
        if sw is not None:
            if is_pipeline:
                m.fit(X_tr, y_tr, logisticregression__sample_weight=sw)
            else:
                m.fit(X_tr, y_tr, sample_weight=sw)
        else:
            m.fit(X_tr, y_tr)

    # Convert to numpy once; use array indexing in folds to avoid DataFrame copy overhead.
    # float32 halves memory vs float64; HGB accepts float32 natively.
    X_arr = X.to_numpy(dtype=np.float32)

    pred = np.zeros(len(X_arr), dtype=np.float32)
    if args.label_source == "test_xfold":
        from sklearn.model_selection import GroupKFold

        gkf = GroupKFold(n_splits=args.n_folds)
        for fold, (tr, te) in enumerate(gkf.split(X_arr, y, groups)):
            m = _new_model()
            sw = sample_weights[tr] if sample_weights is not None else None
            _fit_model(m, X_arr[tr], y[tr], sw)
            pred[te] = m.predict_proba(X_arr[te])[:, 1]
            logger.info("fold %d: train=%d test=%d pos_tr=%d", fold, len(tr), len(te), int(y[tr].sum()))
    else:  # valid
        vdir = args.valid_retrieval_dir or os.path.join(
            args.output_dir, f"{args.dataset}-retrieval-valid-s{args.seed}-N{args.N}"
        )
        if not os.path.isdir(vdir):
            logger.warning(
                "--label_source valid needs validation pools at %s "
                "(run run_retrieval.py on the valid split first). Aborting.",
                vdir,
            )
            return
        valid = pd.read_csv(os.path.join(sp, "valid.csv"))
        vgold = dict(zip(valid.user_idx.astype(int), valid.item_idx.astype(int)))
        vpools = _load_pools(vdir, retr)
        vraw = _load_pool_scores(vdir, retr)
        vscores = _build_score_lookup(vraw)
        del vraw
        vusers = sorted({u for r in vpools for u in vpools[r]})
        if args.max_train_users and len(vusers) > args.max_train_users:
            rng2 = np.random.default_rng(args.seed)
            vusers = sorted(
                int(u)
                for u in rng2.choice(vusers, size=args.max_train_users, replace=False)
            )
            logger.info("capped valid training users -> %d (--max_train_users)", len(vusers))
        Xv, yv, gv, uiv = _build_features(
            vusers,
            vpools,
            list(vpools),
            args.N,
            hist_len,
            item_pop,
            item_new,
            item_textlen,
            user_cold_set,
            vgold,
            score_lookup=vscores,
        )
        Xv_arr = Xv.reindex(columns=X.columns, fill_value=args.N + 1).to_numpy(
            dtype=np.float32
        )
        del Xv
        m = _new_model()
        vsw = None
        if sample_weights is not None:
            vsw = _build_sample_weights(
                yv,
                gv,
                uiv,
                valid if "is_item_new" in valid.columns else test,
                regime_weights,
            )
        _fit_model(m, Xv_arr, yv, vsw)
        del Xv_arr
        pred = m.predict_proba(X_arr)[:, 1]
        logger.info("trained on valid rows=%d pos=%d", len(yv), int(yv.sum()))

    # Re-rank union pool by predicted score, take top-N.
    df = pd.DataFrame(
        {"user": [u for u, _ in ui], "item": [i for _, i in ui], "p": pred}
    )
    lhf_top = {
        int(u): g.sort_values("p", ascending=False)["item"]
        .astype(int)
        .head(args.N)
        .tolist()
        for u, g in df.groupby("user")
    }

    # ---- Comparison: coverage@N ----
    union_top = {
        u: list({i for r in retr for i in pools[r].get(u, [])[: args.N]}) for u in users
    }
    cov = {
        "LHF (learned)": _coverage(lhf_top, gold, test, users),
        "union (oracle)": _coverage(union_top, gold, test, users),
    }
    for r in retr:
        cov[f"single:{r}"] = _coverage(
            {u: pools[r].get(u, [])[: args.N] for u in users}, gold, test, users
        )
    for extra in ["fusion", "cara"]:
        p = os.path.join(rdir, f"{extra}_scores.csv")
        if os.path.exists(p):
            cov[f"heuristic:{extra}"] = _coverage(
                _ranked(pd.read_csv(p)), gold, test, users
            )

    best_single = max((cov[k]["overall"], k) for k in cov if k.startswith("single:"))
    print("\n=== coverage@%d (overall + per primary scenario; (rate,n)) ===" % args.N)
    order = ["LHF (learned)", "union (oracle)", best_single[1]] + [
        k for k in cov if k.startswith("heuristic:")
    ]
    cols = ["overall", "item_new", "item_cold", "long_tail", "user_cold", "warm"]
    print(f"{'method':<22}" + "".join(f"{c:>12}" for c in cols))
    for k in order:
        row = cov[k]
        cells = [f"{row['overall']:.4f}"] + [
            (f"{row[c][0]:.3f}" if isinstance(row.get(c), tuple) and row[c][1] else "-")
            for c in cols[1:]
        ]
        print(f"{k:<22}" + "".join(f"{c:>12}" for c in cells))
    lhf_o, uni_o, bs_o = (
        cov["LHF (learned)"]["overall"],
        cov["union (oracle)"]["overall"],
        best_single[0],
    )
    realized = (lhf_o - bs_o) / (uni_o - bs_o) if uni_o > bs_o else float("nan")
    print(
        f"\n[LHF]{' [' + abl_tag + ']' if abl_tag else ''} best-single={bs_o:.4f}  LHF={lhf_o:.4f}  "
        f"oracle-union={uni_o:.4f}  -> LHF realizes {realized * 100:.1f}% of the oracle headroom"
    )

    # Recall@K curve: LHF ranking at K=10,50,100,200,500.
    recall_ks = [10, 50, 100, 200, 500]
    print("\n=== Recall@K curve (candidate-generation ranking) ===")
    header = f"{'method':<22}" + "".join(f"{'R@' + str(k):>10}" for k in recall_ks)
    print(header)
    for label, topn in [
        ("LHF", lhf_top),
        ("oracle-union", union_top),
        (
            best_single[1].replace("single:", "best:"),
            {
                u: pools[best_single[1].replace("single:", "")].get(u, [])[: args.N]
                for u in users
            },
        ),
    ]:
        cells = []
        for k in recall_ks:
            trunc = {u: lst[:k] for u, lst in topn.items()}
            c = _coverage(trunc, gold, test, users)
            cells.append(f"{c['overall']:.4f}")
        print(f"{label:<22}" + "".join(f"{c:>10}" for c in cells))
    for extra in ["fusion", "cara"]:
        p = os.path.join(rdir, f"{extra}_scores.csv")
        if os.path.exists(p):
            ext_ranked = _ranked(pd.read_csv(p))
            cells = []
            for k in recall_ks:
                trunc = {u: ext_ranked.get(u, [])[:k] for u in users}
                c = _coverage(trunc, gold, test, users)
                cells.append(f"{c['overall']:.4f}")
            print(f"{extra:<22}" + "".join(f"{c:>10}" for c in cells))

    # Regret to oracle.
    print("\n=== Regret to oracle (oracle_coverage - method_coverage) ===")
    print(f"{'method':<22}{'regret':>10}{'regret%':>10}")
    for k in order:
        reg = uni_o - cov[k]["overall"]
        regpct = reg / uni_o * 100 if uni_o > 0 else float("nan")
        print(f"{k:<22}{reg:>10.4f}{regpct:>9.1f}%")

    if args.write:
        rows_u, rows_i, rows_s = [], [], []
        for u, arr in lhf_top.items():
            n = len(arr)
            rows_u += [u] * n
            rows_i += arr
            rows_s += list(range(n, 0, -1))
        if abl_tag:
            out_name = f"lhf_{abl_tag}_scores.csv"
        else:
            out_name = "lhf_scores.csv"
        outp = os.path.join(rdir, out_name)
        pd.DataFrame({"user_idx": rows_u, "item_idx": rows_i, "score": rows_s}).to_csv(
            outp, index=False
        )
        logger.info("wrote %s", outp)


if __name__ == "__main__":
    main()
