"""Retrieval-realistic coverage experiment (RQ2 + the candidate-expansion pillar).

The main pipeline uses a ``positive_controlled`` candidate pool: the gold test
item is *always* present, which isolates reranker quality but makes retrieval /
candidate-expansion failures invisible by construction.

This experiment removes that guarantee. Each retriever proposes its own top-N
from the **full item catalogue** (minus the user's seen items), with NO gold
guarantee. We then measure oracle coverage — how often the gold test item lands
in the retrieved pool — overall and per cold/tail scenario. This is the only
protocol under which graph-based candidate expansion can demonstrate value, and
it directly answers RQ2 ("is the failure retrieval or rerank?").

Retrievers compared
-------------------
  popularity      non-personalised floor (global frequency order)
  itemknn         cosine item-item CF
  bpr             BPR-MF dot-product ranking
  lightgcn        LightGCN dot-product ranking (graph CF)
  graph_cooccur   candidate EXPANSION: union of co-occurrence-graph neighbours
                  of the user's history items (non-parametric graph)
  graph_emb       candidate EXPANSION: union of LightGCN-embedding nearest
                  neighbours of the user's history items (learned graph)

The two ``graph_*`` retrievers are the previously-untested "graph candidate
expansion" module: seeded from the user's own interaction history rather than a
single global ranking. Comparing their cold/tail coverage against the CF
retrievers tells us whether graphs help at *retrieval* (a defensible positive)
or not (a stronger negative that generalises the RQ3 result).

Outputs use the SAME schema as the rank experiments (each retriever is scored as a
full-catalogue ranker, so `recall@N` = pool coverage and `recall@k` = oracle coverage
at depth k). Written to a separate run dir; the existing split is read-only:
  all_model_metrics.csv           per retriever: recall@k / ndcg@k / mrr@k + exposure
  all_model_subgroup_metrics.csv  per retriever x cold/tail scenario
  all_model_metrics_ci.csv        bootstrap CIs per retriever
  retrieval_report.md             human-readable summary
  <retriever>_scores.csv          (optional, --write_pools) the realistic pools
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from coldtail.metrics import bootstrap_ranking_ci, evaluate_rankings, subgroup_metrics
from coldtail.recommenders.base import build_user_item_matrix, infer_shape
from coldtail.utils import get_device

logger = logging.getLogger(__name__)

DEFAULT_RETRIEVERS = [
    "popularity",
    "itemknn",
    "tfidf",
    "sbert",
    "markov",
    "sasrec",
    "bpr",
    "lightgcn",
    "graph_cooccur",
    "graph_emb",
    "fusion",  # RRF fusion — MUST be after its components (reuses their top-N)
    "fusion_il",  # interleave (round-robin) fusion — also after its components
    "cara",  # Coverage-Aware Retrieval Allocation (regime-aware budgets) — last
]

# Stronger off-the-shelf dense retrievers (revision B2): name -> (model ref, doc prefix).
# NOT in DEFAULT_RETRIEVERS (they are large); run explicitly, e.g. RETRIEVERS=bge_large,e5_large.
# They test whether a *stronger* content encoder narrows the coverage bottleneck — answering
# "is retrieval the structural bottleneck, or just our retriever being weak?".
#
# Prefer a local model cache directory (set via COLDTAIL_STRONG_ENCODERS_DIR env var),
# falling back to the HuggingFace Hub repo id when models aren't pre-downloaded.
MODELS_DIR = os.environ.get("COLDTAIL_STRONG_ENCODERS_DIR", None)


def _model_ref(repo: str, local_name: str) -> str:
    if MODELS_DIR is None:
        return repo
    local = os.path.join(MODELS_DIR, local_name)
    return local if os.path.isdir(local) else repo


DENSE_MODELS = {
    "bge_large": (
        _model_ref("BAAI/bge-large-en-v1.5", "bge-large-en-v1.5"),
        "",
    ),  # upgrade over bge-base `sbert`
    "bge_m3": (_model_ref("BAAI/bge-m3", "bge-m3"), ""),  # 568M multilingual; no prefix
    "e5_large": (
        _model_ref("intfloat/e5-large-v2", "e5-large-v2"),
        "passage: ",
    ),  # diff strong family; prefix
}

# Components fused by the "fusion" retriever (reciprocal-rank fusion). Mirrors the
# oracle UNION_SET in analyze_oracle_ceiling.py so the realised-vs-ceiling gap is
# a clean comparison.
FUSION_COMPONENTS = ["lightgcn", "itemknn", "sbert", "popularity"]

# CARA: per-regime budget *fractions* over retrievers (sum≈1 each). The regime is
# gated by the serving-time-observable user_cold flag (no test labels): cold users
# get a content-heavy pool (semantic retrievers reach their items), warm users a
# CF-heavy pool. Budgets can be validation-tuned; these are sensible defaults.
CARA_PROFILES = {
    "cold": {
        "sbert": 0.35,
        "tfidf": 0.20,
        "itemknn": 0.20,
        "lightgcn": 0.15,
        "popularity": 0.10,
    },
    "warm": {
        "lightgcn": 0.60,
        "itemknn": 0.15,
        "popularity": 0.10,
        "sbert": 0.10,
        "tfidf": 0.05,
    },
}


# ---------------------------------------------------------------------------
# Per-user seen / history helpers
# ---------------------------------------------------------------------------
def _seen_sets(
    train: pd.DataFrame, valid: pd.DataFrame | None = None
) -> dict[int, set[int]]:
    """Items each user has already interacted with (train + valid). These are
    excluded from every retriever's pool so coverage measures genuinely *new*
    retrieval, matching the main pipeline's exclusion of seen items.
    """
    seen = (
        train.groupby("user_idx")["item_idx"]
        .apply(lambda s: set(map(int, s)))
        .to_dict()
    )
    if valid is not None and len(valid):
        for u, grp in valid.groupby("user_idx"):
            seen.setdefault(int(u), set()).update(int(x) for x in grp["item_idx"])
    return seen


def _recent_history(train: pd.DataFrame, window: int) -> dict[int, list[int]]:
    """Last ``window`` train items per user (chronological), used to seed the
    graph-expansion retrievers.
    """
    return (
        train.sort_values("timestamp")
        .groupby("user_idx")["item_idx"]
        .apply(lambda s: [int(x) for x in s][-window:])
        .to_dict()
    )


# ---------------------------------------------------------------------------
# Per-retriever top-N over the full catalogue (gold NOT injected)
# ---------------------------------------------------------------------------
def _topn_popularity(train, shape, eval_users, seen, N) -> dict[int, np.ndarray]:
    counts = (
        train.groupby("item_idx")
        .size()
        .reindex(range(shape.n_items), fill_value=0)
        .to_numpy()
    )
    order = np.argsort(-counts, kind="stable")  # best (most popular) first
    out: dict[int, np.ndarray] = {}
    for u in eval_users:
        s = seen.get(u)
        if s:
            # Among the first N+|seen| popular items, at most |seen| are removed,
            # so >= N remain — a cheap exact slice instead of scanning the catalogue.
            buf = order[: N + len(s)]
            picked = buf[~np.isin(buf, list(s))][:N]
        else:
            picked = order[:N]
        out[u] = picked.astype(np.int64)
    return out


def _topn_itemknn(train, shape, eval_users, seen, N, topk_sim) -> dict[int, np.ndarray]:
    from coldtail.recommenders.itemknn import fit_itemknn_sim

    ui = build_user_item_matrix(train, shape).astype(np.float32)
    sim = fit_itemknn_sim(train, shape, topk_sim)
    user_scores = (ui @ sim).tocsr()
    K = min(N, shape.n_items)
    out: dict[int, np.ndarray] = {}
    for u in eval_users:
        row = np.asarray(user_scores[u].todense()).ravel()
        s = seen.get(u)
        if s:
            row[list(s)] = -np.inf
        if len(row) > K:
            idx = np.argpartition(-row, K - 1)[:K]
            idx = idx[np.argsort(-row[idx], kind="stable")]
        else:
            idx = np.argsort(-row, kind="stable")
        out[u] = idx.astype(np.int64)
    return out


def _topn_from_dense(row, seen_u, K, n_items) -> np.ndarray:
    """Top-K item indices from a dense per-user score row (excl. seen)."""
    if seen_u:
        row[list(seen_u)] = -np.inf
    if n_items > K:
        idx = np.argpartition(-row, K - 1)[:K]
        idx = idx[np.argsort(-row[idx], kind="stable")]
    else:
        idx = np.argsort(-row, kind="stable")
    return idx.astype(np.int64)


def _topn_tfidf(
    user_profile, item_vec, eval_users, seen, N, batch=512
) -> dict[int, np.ndarray]:
    """Top-N by TF-IDF cosine over the full catalogue (sparse, user-batched)."""
    n_items = item_vec.shape[0]
    K = min(N, n_items)
    item_vec_t = item_vec.T.tocsr()  # vocab x n_items
    eu = np.asarray(eval_users, dtype=np.int64)
    out: dict[int, np.ndarray] = {}
    for start in range(0, len(eu), batch):
        bu = eu[start : start + batch]
        dense = np.asarray((user_profile[bu] @ item_vec_t).todense())  # (b, n_items)
        for r, u in enumerate(bu):
            out[int(u)] = _topn_from_dense(dense[r], seen.get(int(u)), K, n_items)
    return out


def _topn_markov(
    trans, last_item, eval_users, seen, N, n_items
) -> dict[int, np.ndarray]:
    """Top-N by first-order transition prob from the user's last item."""
    K = min(N, n_items)
    out: dict[int, np.ndarray] = {}
    for u in eval_users:
        li = last_item.get(u)
        if li is None:
            out[u] = np.empty(0, dtype=np.int64)
            continue
        row = np.asarray(trans[li].todense()).ravel().astype(np.float64)
        out[u] = _topn_from_dense(row, seen.get(u), K, n_items)
    return out


def _topn_sasrec(
    model, seqs, max_seq_len, eval_users, seen, N, device, batch=128
) -> dict[int, np.ndarray]:
    """Top-N by SASRec sequence hidden-state vs all item embeddings."""
    import torch

    from coldtail.recommenders.sasrec import _pad_seq

    item_mat = model.item_emb.weight[1:]  # (n_items, dim); row j == item_idx j
    n_items = item_mat.shape[0]
    K = min(N, n_items)
    eu = np.asarray(eval_users, dtype=np.int64)
    out: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, len(eu), batch):
            bu = eu[start : start + batch]
            seq_t = torch.tensor(
                [_pad_seq(seqs.get(int(u), []), max_seq_len) for u in bu],
                dtype=torch.long,
                device=device,
            )
            scores = model.forward(seq_t) @ item_mat.t()  # (b, n_items)
            for i, u in enumerate(bu):
                s = seen.get(int(u))
                if s:
                    scores[
                        i, torch.as_tensor(list(s), device=device, dtype=torch.long)
                    ] = float("-inf")
            topi = torch.topk(scores, K, dim=1).indices.cpu().numpy()
            for i, u in enumerate(bu):
                out[int(u)] = topi[i].astype(np.int64)
    return out


def _topn_fusion(
    topn_by_retriever, components, eval_users, N, mode="rrf", k_rrf=60
) -> dict[int, np.ndarray]:
    """Fuse several retrievers' top-N lists into a single top-N (a *real*,
    non-oracle multi-signal retriever — its coverage vs the oracle union tells how
    much complementarity one fused ranking realises). Components already excluded
    seen items, so the fused pool does too.

    mode:
      'rrf'        reciprocal-rank fusion: score = sum_r 1/(k_rrf + rank_r), then
                   take top-N. Re-scores globally, so weak retrievers can demote a
                   strong retriever's top picks (dilution).
      'interleave' round-robin slot allocation: take rank-0 from each retriever,
                   then rank-1, ... (dedup), until N. PRESERVES each retriever's
                   top items, so it keeps the dominant retriever's warm picks while
                   adding the content retriever's cold picks — no demotion.
    """
    out: dict[int, np.ndarray] = {}
    for u in eval_users:
        lists = [topn_by_retriever.get(c, {}).get(u, []) for c in components]
        if mode == "interleave":
            picked: list[int] = []
            seen_i: set[int] = set()
            for rank in range(max((len(lst) for lst in lists), default=0)):
                for lst in lists:
                    if rank < len(lst):
                        it = int(lst[rank])
                        if it not in seen_i:
                            seen_i.add(it)
                            picked.append(it)
                if len(picked) >= N:
                    break
            out[u] = np.array(picked[:N], dtype=np.int64)
        else:
            agg: dict[int, float] = defaultdict(float)
            for lst in lists:
                for rank, item in enumerate(lst):
                    agg[int(item)] += 1.0 / (k_rrf + rank + 1)
            fused = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
            out[u] = np.array([it for it, _ in fused[:N]], dtype=np.int64)
    return out


def _topn_cara(
    topn_by_retriever, profiles, user_cold_set, eval_users, N
) -> dict[int, np.ndarray]:
    """Coverage-Aware Retrieval Allocation: per user, pick a regime (cold/warm) from
    the serving-time user_cold flag, allocate the N budget across retrievers by the
    regime profile (top-b_r from each, highest budget first, dedup), and pad if short.
    A real, regime-aware allocator — its coverage vs the oracle union and vs uniform
    fusion tells whether allocating by regime captures more of the complementarity.
    """
    out: dict[int, np.ndarray] = {}
    for u in eval_users:
        prof = profiles["cold"] if u in user_cold_set else profiles["warm"]
        order = sorted(prof, key=lambda r: -prof[r])
        picked: list[int] = []
        seen_i: set[int] = set()
        for r in order:
            b = max(1, int(round(prof[r] * N)))
            for it in topn_by_retriever.get(r, {}).get(u, [])[:b]:
                it = int(it)
                if it not in seen_i:
                    seen_i.add(it)
                    picked.append(it)
                    if len(picked) >= N:
                        break
            if len(picked) >= N:
                break
        if len(picked) < N:  # pad from the regime's retrievers beyond their budget
            for r in order:
                for it in topn_by_retriever.get(r, {}).get(u, []):
                    it = int(it)
                    if it not in seen_i:
                        seen_i.add(it)
                        picked.append(it)
                        if len(picked) >= N:
                            break
                if len(picked) >= N:
                    break
        out[u] = np.array(picked[:N], dtype=np.int64)
    return out


def _topn_embedding(U, V, eval_users, seen, N, batch=256) -> dict[int, np.ndarray]:
    """Top-N by user-item dot product over the full catalogue (BPR / LightGCN)."""
    import torch

    device = U.device
    n_items = V.shape[0]
    K = min(N, n_items)
    Vt = V.t().contiguous()
    eu = np.asarray(eval_users, dtype=np.int64)
    out: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, len(eu), batch):
            bu = eu[start : start + batch]
            scores = U[torch.as_tensor(bu, device=device)] @ Vt  # (b, n_items)
            for i, u in enumerate(bu):
                s = seen.get(int(u))
                if s:
                    scores[
                        i, torch.as_tensor(list(s), device=device, dtype=torch.long)
                    ] = float("-inf")
            topi = torch.topk(scores, K, dim=1).indices.cpu().numpy()
            for i, u in enumerate(bu):
                out[int(u)] = topi[i].astype(np.int64)
    return out


def _build_weighted_cooccur(
    train, window=50, min_cooccur=2
) -> dict[int, dict[int, int]]:
    """Symmetric item co-occurrence counts (capped to last ``window`` items/user)."""
    pair: dict[tuple[int, int], int] = defaultdict(int)
    for _, grp in train.sort_values("timestamp").groupby("user_idx")["item_idx"]:
        its = sorted({int(x) for x in list(grp)[-window:]})
        for a, b in combinations(its, 2):
            pair[(a, b)] += 1
    co: dict[int, dict[int, int]] = defaultdict(dict)
    for (a, b), c in pair.items():
        if c >= min_cooccur:
            co[a][b] = c
            co[b][a] = c
    return co


def _embedding_item_knn(
    V, item_ids, k=50, batch=512
) -> dict[int, list[tuple[int, float]]]:
    """Cosine top-k neighbours over the full catalogue, only for ``item_ids``."""
    import torch
    import torch.nn.functional as F

    Vn = F.normalize(V, dim=1)
    ids = np.asarray(sorted({int(x) for x in item_ids}), dtype=np.int64)
    out: dict[int, list[tuple[int, float]]] = {}
    K = min(k + 1, V.shape[0])
    with torch.no_grad():
        for start in range(0, len(ids), batch):
            b = ids[start : start + batch]
            sims = Vn[torch.as_tensor(b, device=V.device)] @ Vn.t()  # (b, n_items)
            for i, it in enumerate(b):
                sims[i, int(it)] = float("-inf")  # exclude self
            topv, topi = torch.topk(sims, K, dim=1)
            tv, ti = topv.cpu().numpy(), topi.cpu().numpy()
            for i, it in enumerate(b):
                out[int(it)] = list(zip(ti[i].tolist(), tv[i].tolist()))[:k]
    return out


def _topn_graph_expansion(
    history, neighbours, eval_users, seen, N
) -> dict[int, np.ndarray]:
    """Candidate expansion: aggregate weighted neighbours of a user's history
    items into a per-user candidate score, then take top-N (gold not injected).

    ``neighbours`` maps item -> iterable of (neighbour_item, weight).
    """
    out: dict[int, np.ndarray] = {}
    for u in eval_users:
        sc: dict[int, float] = defaultdict(float)
        for it in history.get(u, []):
            for nb, w in neighbours.get(it, ()):  # type: ignore[union-attr]
                sc[nb] += float(w)
        s = seen.get(u, set())
        ranked = sorted(((v, k) for k, v in sc.items() if k not in s), reverse=True)
        out[u] = np.array([k for _, k in ranked[:N]], dtype=np.int64)
    return out


# ---------------------------------------------------------------------------
# Coverage scoring
# ---------------------------------------------------------------------------
def _pool_df(topn: dict[int, np.ndarray]) -> pd.DataFrame:
    """Flatten {user: [items best->worst]} into a scored candidates frame with a
    descending ``score`` so the ranking metrics reproduce the retriever's order.

    The result is a drop-in for ``evaluate_rankings`` / ``subgroup_metrics`` —
    the same functions the rank experiments use — so a retriever's full-catalogue
    ``recall@k`` (gold not injected) equals its oracle coverage at depth k, and
    ``recall@N`` equals ``pool_coverage``.
    """
    users, items, scores = [], [], []
    for u, arr in topn.items():
        n = len(arr)
        users.extend([u] * n)
        items.extend(int(x) for x in arr)
        scores.extend(range(n, 0, -1))  # rank n .. 1, descending
    return pd.DataFrame({"user_idx": users, "item_idx": items, "score": scores})


def run_retrieval_coverage(
    processed_dir: Path,
    out_dir: Path,
    cfg: dict,
    retrieval_n: int = 200,
    retrievers: list[str] | None = None,
    write_pools: bool = False,
    eval_split: str = "test",
) -> Path:
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    retrievers = retrievers or DEFAULT_RETRIEVERS

    train = pd.read_csv(processed_dir / "train.csv")
    valid = (
        pd.read_csv(processed_dir / "valid.csv")
        if (processed_dir / "valid.csv").exists()
        else None
    )
    test = pd.read_csv(processed_dir / "test.csv")
    items = pd.read_csv(processed_dir / "items_mapped.csv")
    item_stats = pd.read_csv(processed_dir / "item_stats.csv")
    is_tail = item_stats.set_index("item_idx")["is_tail"]

    # Which split is the prediction target. eval_split="valid" builds pools whose gold
    # is the validation positive (for training a learned fusion without test leakage);
    # in that case only TRAIN items are masked as "seen" (valid items are the targets).
    if eval_split == "valid":
        if valid is None:
            raise FileNotFoundError(
                f"eval_split='valid' but no valid.csv at {processed_dir}"
            )
        test = valid  # `test` below is the gold/eval frame throughout
        seen = _seen_sets(train, None)
    else:
        seen = _seen_sets(train, valid)

    shape = infer_shape(train, None, items)  # full catalogue
    eval_users = sorted(int(u) for u in test.user_idx.unique())
    k_list = [
        k
        for k in cfg.get("metrics", {}).get("k_list", [5, 10, 20, 49])
        if k <= retrieval_n
    ]
    if retrieval_n not in k_list:
        k_list = sorted(set(k_list + [retrieval_n]))

    logger.info(
        "[retrieval] dataset shape | n_users=%d n_items=%d eval_users=%d N=%d retrievers=%s",
        shape.n_users,
        shape.n_items,
        len(eval_users),
        retrieval_n,
        retrievers,
    )

    # Shared trained artefacts (train each model at most once).
    tcfg = cfg.get("training", {})
    device = get_device(tcfg.get("device", "auto"))
    dim = tcfg.get("embedding_dim", 64)
    seed = cfg.get("seed", 42)
    lgcn_V = None  # LightGCN item embeddings, reused by graph_emb

    pools: dict[str, pd.DataFrame] = {}
    topn_by_retriever: dict[
        str, dict[int, np.ndarray]
    ] = {}  # raw lists, reused by fusion

    for name in retrievers:
        logger.info("[retrieval] building pool | retriever=%s", name)
        if name == "popularity":
            topn = _topn_popularity(train, shape, eval_users, seen, retrieval_n)

        elif name == "itemknn":
            topk_sim = cfg.get("itemknn", {}).get("topk_sim", 100)
            topn = _topn_itemknn(train, shape, eval_users, seen, retrieval_n, topk_sim)

        elif name == "sbert":
            import torch

            from coldtail.recommenders.sbert_retrieval import build_sbert_embeddings

            sb = cfg.get("sbert", {})
            user_prof, item_emb = build_sbert_embeddings(
                train,
                items,
                shape,
                model_name=sb.get("model_name", "BAAI/bge-base-en-v1.5"),
                max_hist=sb.get("max_hist", 10),
                device=device,
                batch_size=sb.get("batch_size", 256),
            )
            topn = _topn_embedding(
                torch.as_tensor(user_prof, device=device),
                torch.as_tensor(item_emb, device=device),
                eval_users,
                seen,
                retrieval_n,
            )

        elif name in DENSE_MODELS:
            # Stronger off-the-shelf dense retriever (B2): same pipeline as `sbert`,
            # different (larger) encoder + optional doc prefix.
            import torch

            from coldtail.recommenders.sbert_retrieval import build_sbert_embeddings

            model_id, doc_prefix = DENSE_MODELS[name]
            sb = cfg.get("sbert", {})
            user_prof, item_emb = build_sbert_embeddings(
                train,
                items,
                shape,
                model_name=model_id,
                max_hist=sb.get("max_hist", 10),
                device=device,
                batch_size=sb.get("batch_size", 256),
                doc_prefix=doc_prefix,
            )
            topn = _topn_embedding(
                torch.as_tensor(user_prof, device=device),
                torch.as_tensor(item_emb, device=device),
                eval_users,
                seen,
                retrieval_n,
            )

        elif name == "tfidf":
            from coldtail.recommenders.content_tfidf import build_tfidf_vectors

            user_prof, item_vec = build_tfidf_vectors(train, items, shape, max_hist=10)
            topn = _topn_tfidf(user_prof, item_vec, eval_users, seen, retrieval_n)

        elif name == "markov":
            from coldtail.recommenders.markov import build_markov

            trans, last_item = build_markov(train, shape)
            topn = _topn_markov(
                trans, last_item, eval_users, seen, retrieval_n, shape.n_items
            )

        elif name == "sasrec":
            from coldtail.recommenders.sasrec import train_sasrec

            sas = cfg.get("sasrec", {})
            model, seqs = train_sasrec(
                train,
                shape,
                dim=dim,
                max_seq_len=sas.get("max_seq_len", 50),
                n_heads=sas.get("n_heads", 2),
                n_layers=sas.get("n_layers", 2),
                dropout=sas.get("dropout", 0.2),
                epochs=tcfg.get("epochs_sasrec", 100),
                batch_size=min(tcfg.get("batch_size", 4096), 256),
                lr=tcfg.get("lr", 1e-3),
                weight_decay=tcfg.get("weight_decay", 1e-6),
                device=device,
                seed=seed,
            )
            topn = _topn_sasrec(
                model,
                seqs,
                sas.get("max_seq_len", 50),
                eval_users,
                seen,
                retrieval_n,
                device,
            )

        elif name == "bpr":
            import torch

            from coldtail.recommenders.bpr import train_bpr

            model = train_bpr(
                train,
                shape,
                dim=dim,
                epochs=tcfg.get("epochs_bpr", 20),
                batch_size=tcfg.get("batch_size", 4096),
                lr=tcfg.get("lr", 1e-3),
                weight_decay=tcfg.get("weight_decay", 1e-6),
                device=device,
                seed=seed,
            )
            with torch.no_grad():
                topn = _topn_embedding(
                    model.user_emb.weight.detach(),
                    model.item_emb.weight.detach(),
                    eval_users,
                    seen,
                    retrieval_n,
                )

        elif name in ("lightgcn", "graph_emb"):
            import torch

            from coldtail.recommenders.lightgcn import train_lightgcn

            if lgcn_V is None:
                model, adj = train_lightgcn(
                    train,
                    shape,
                    dim=dim,
                    epochs=tcfg.get("epochs_lightgcn", 20),
                    batch_size=tcfg.get("batch_size", 4096),
                    lr=tcfg.get("lr", 1e-3),
                    weight_decay=tcfg.get("weight_decay", 1e-6),
                    n_layers=cfg.get("lightgcn", {}).get("n_layers", 2),
                    device=device,
                    seed=seed,
                )
                with torch.no_grad():
                    lgcn_U, lgcn_V = model.propagate(adj)
                    lgcn_U, lgcn_V = lgcn_U.detach(), lgcn_V.detach()
            if name == "lightgcn":
                topn = _topn_embedding(lgcn_U, lgcn_V, eval_users, seen, retrieval_n)
            else:  # graph_emb: expand from history via embedding kNN
                history = _recent_history(
                    train, window=cfg.get("graph_expansion", {}).get("hist_window", 50)
                )
                hist_items = {it for u in eval_users for it in history.get(u, [])}
                knn = _embedding_item_knn(
                    lgcn_V, hist_items, k=cfg.get("graph_expansion", {}).get("knn", 50)
                )
                topn = _topn_graph_expansion(
                    history, knn, eval_users, seen, retrieval_n
                )

        elif name == "graph_cooccur":
            history = _recent_history(
                train, window=cfg.get("graph_expansion", {}).get("hist_window", 50)
            )
            cooccur = _build_weighted_cooccur(
                train,
                window=cfg.get("graph_expansion", {}).get("hist_window", 50),
                min_cooccur=cfg.get("graph_expansion", {}).get("min_cooccur", 2),
            )
            neighbours = {it: list(nbrs.items()) for it, nbrs in cooccur.items()}
            topn = _topn_graph_expansion(
                history, neighbours, eval_users, seen, retrieval_n
            )

        elif name == "two_tower":
            from coldtail.recommenders.two_tower import train_two_tower, two_tower_topn

            tt = cfg.get("two_tower", {})
            sb = cfg.get("sbert", {})
            model_name = tt.get(
                "model_name", sb.get("model_name", "BAAI/bge-base-en-v1.5")
            )
            tt_model, user_prof, item_emb = train_two_tower(
                train,
                items,
                shape,
                model_name=model_name,
                proj_dim=tt.get("proj_dim", 128),
                max_hist=tt.get("max_hist", sb.get("max_hist", 10)),
                epochs=tt.get("epochs", 10),
                batch_size=tt.get("batch_size", 512),
                lr=tt.get("lr", 1e-4),
                neg_per_pos=tt.get("neg_per_pos", 15),
                device=device,
                seed=seed,
                encoder_batch_size=sb.get("batch_size", 256),
            )
            topn = two_tower_topn(
                tt_model,
                user_prof,
                item_emb,
                eval_users,
                seen,
                retrieval_n,
                device=device,
            )

        elif name in ("fusion", "fusion_il"):
            fcfg = cfg.get("fusion", {})
            components = fcfg.get("components", FUSION_COMPONENTS)
            avail = [c for c in components if c in topn_by_retriever]
            if not avail:
                logger.warning(
                    "[retrieval] %s: none of its components %s ran before it — skipped "
                    "(fusion must come after its components)",
                    name,
                    components,
                )
                continue
            if len(avail) < len(components):
                logger.warning(
                    "[retrieval] %s: missing components %s; using %s",
                    name,
                    [c for c in components if c not in avail],
                    avail,
                )
            topn = _topn_fusion(
                topn_by_retriever,
                avail,
                eval_users,
                retrieval_n,
                mode=("interleave" if name == "fusion_il" else "rrf"),
                k_rrf=fcfg.get("k_rrf", 60),
            )

        elif name == "cara":
            profiles = cfg.get("cara", {}).get("profiles", CARA_PROFILES)
            comps = {r for prof in profiles.values() for r in prof}
            if sum(c in topn_by_retriever for c in comps) < 2:
                logger.warning(
                    "[retrieval] cara: <2 of its components %s ran before it — skipped",
                    comps,
                )
                continue
            user_cold = (
                {
                    int(u)
                    for u in test.loc[test["is_user_cold"].astype(bool), "user_idx"]
                }
                if "is_user_cold" in test.columns
                else set()
            )
            logger.info(
                "[retrieval] cara | user_cold=%d/%d eval users",
                len(user_cold & set(eval_users)),
                len(eval_users),
            )
            topn = _topn_cara(
                topn_by_retriever, profiles, user_cold, eval_users, retrieval_n
            )

        else:
            logger.warning("[retrieval] unknown retriever '%s' — skipped", name)
            continue

        topn_by_retriever[name] = topn
        pool = _pool_df(topn)
        pools[name] = pool
        if write_pools:
            pool.to_csv(out_dir / f"{name}_scores.csv", index=False)

    # ----- Metrics, in the SAME schema as the rank experiments -----
    # Each retriever is evaluated as a full-catalogue ranker, so recall@k = oracle
    # coverage at depth k and recall@N = pool_coverage. Files mirror run_dataset.py:
    #   all_model_metrics.csv / all_model_subgroup_metrics.csv / all_model_metrics_ci.csv
    metrics_rows, subgroup_frames, ci_rows = [], [], []
    n_boot = int(cfg.get("metrics", {}).get("n_boot", 1000))
    for name, pool in pools.items():
        met = evaluate_rankings(pool, test, k_list, is_tail, top_candidates=retrieval_n)
        met.update(model=name, eval_users=len(eval_users), eval_user_coverage=1.0)
        metrics_rows.append(met)

        sub = subgroup_metrics(pool, test, k_list, top_candidates=retrieval_n)
        if len(sub):
            sub["model"] = name
            subgroup_frames.append(sub)

        ci = bootstrap_ranking_ci(
            pool, test, k_list, top_candidates=retrieval_n, n_boot=n_boot, seed=seed
        )
        ci["model"] = name
        ci_rows.append(ci)
        logger.info(
            "[retrieval] scored | %s | recall@%d(coverage)=%.4f",
            name,
            retrieval_n,
            met.get(f"recall@{retrieval_n}", float("nan")),
        )

    overall = pd.DataFrame(metrics_rows)
    ci_df = pd.DataFrame(ci_rows)
    by_scen = (
        pd.concat(subgroup_frames, ignore_index=True)
        if subgroup_frames
        else pd.DataFrame()
    )

    # Auto-merge: preserve existing rows for retrievers NOT run this time (so an
    # incremental add — e.g. RETRIEVERS=bge_large — appends without clobbering the
    # full sweep). Rows for retrievers run this time overwrite their prior values.
    def _merge(new: pd.DataFrame, path: Path) -> pd.DataFrame:
        if path.exists() and len(new) and "model" in new.columns:
            old = pd.read_csv(path)
            if "model" in old.columns:
                old = old[~old["model"].isin(set(new["model"]))]
                new = pd.concat([old, new], ignore_index=True)
        return new

    _merge(overall, out_dir / "all_model_metrics.csv").to_csv(
        out_dir / "all_model_metrics.csv", index=False
    )
    _merge(ci_df, out_dir / "all_model_metrics_ci.csv").to_csv(
        out_dir / "all_model_metrics_ci.csv", index=False
    )
    if len(by_scen):
        _merge(by_scen, out_dir / "all_model_subgroup_metrics.csv").to_csv(
            out_dir / "all_model_subgroup_metrics.csv", index=False
        )

    report_path = _write_report(
        out_dir, overall, by_scen, retrieval_n, shape, len(eval_users)
    )
    logger.info("[retrieval] done | report=%s", report_path)
    return report_path


def _md_table(df: pd.DataFrame) -> str:
    """Markdown table, degrading to plain text if `tabulate` is unavailable."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def _recall_view(df: pd.DataFrame, k_list, lead: list[str]) -> pd.DataFrame:
    """Select the lead columns + available recall@k columns for a compact view."""
    cols = [c for c in lead if c in df.columns] + [
        f"recall@{k}" for k in k_list if f"recall@{k}" in df.columns
    ]
    return df[cols]


def _write_report(out_dir, overall, by_scen, N, shape, n_eval) -> Path:
    k_list = sorted(
        int(c.split("@")[1]) for c in overall.columns if c.startswith("recall@")
    )
    lines = [
        "# Retrieval-realistic coverage",
        "",
        f"- catalogue items: **{shape.n_items:,}**, eval users: **{n_eval:,}**, pool size N = **{N}**",
        "- pool is **not** positive-controlled: gold item present only if the retriever surfaces it.",
        "- metrics use the SAME schema as the rank experiments (`recall@k`/`ndcg@k`/`mrr@k`);",
        f"  here each retriever is a full-catalogue ranker, so **`recall@{N}` = pool coverage**",
        "  and `recall@k` = oracle coverage at depth k. Full columns in `all_model_metrics.csv`.",
        "",
        "## Overall (recall@k)",
        "",
        _md_table(_recall_view(overall, k_list, ["model", "eval_users"])),
    ]
    if len(by_scen):
        lines += ["", "## By scenario (recall@k)", ""]
        for scen, grp in by_scen.groupby("scenario"):
            lines += [
                f"### {scen}",
                "",
                _md_table(_recall_view(grp, k_list, ["model", "num_users"])),
                "",
            ]
    lines += [
        "## Reading the result",
        "",
        f"- `recall@{N}` is the retrieval ceiling for the top-{N} reranking runs: the reranker",
        "  cannot recover any gold item the retriever failed to surface.",
        "- If `graph_cooccur` / `graph_emb` lift cold/tail recall above the CF retrievers, graph",
        "  signal helps at **retrieval** (a defensible positive contribution).",
        "- If they do not, the negative result generalises across both graph pillars",
        "  (prompt-evidence in RQ3 *and* candidate-expansion here).",
        "",
    ]
    path = out_dir / "retrieval_report.md"
    path.write_text("\n".join(lines))
    return path
