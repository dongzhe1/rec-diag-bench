"""Prompt / scoring-variant sanity check for the LLM reranker.

Compares, on the same small subset, common reranking variants and reports
recall@k / ndcg@k to test whether the main results are robust to the choice of
prompt or scoring method.

Variants (``--variants``, comma-separated):
  logprob  pointwise yes/no log-prob (default method)
  score    pointwise 0-100 integer score (generate mode)
  listwise one prompt lists the top-M candidates; the LLM returns an ordered list
  pairwise bubble-style ranking from O(M) sampled pairwise comparisons (expensive)

Pools (``--pool``): ``positive`` (positive-controlled candidates.csv) or
``e2e:<retriever>`` (a realistic pool from the retrieval dir).

GPU required. Keep ``--n_users`` small (100-200) and ``--top_m`` modest (50).

Usage:
  python scripts/run_rerank_variants.py --dataset amazon-videogames --seed 42 \
      --config configs/hpc.yaml --variants logprob,score,listwise \
      --n_users 150 --top_m 50
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from coldtail.config import load_config
from coldtail.experiments.run_llm_rerank import (
    _INSTRUCTION_GENERATE,
    _INSTRUCTION_LOGPROB,
    _answer_token_ids,
    _apply_chat_template,
    _build_baseline_prompt,
    _logodds_to_prob,
    _score_candidates,
)
from coldtail.metrics import evaluate_rankings
from coldtail.utils import seed_everything

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _load_model(model_name, load_in_4bit):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, fix_mistral_regex=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    kw: dict = dict(
        device_map="auto" if device == "cuda" else None, trust_remote_code=True
    )
    if load_in_4bit and device == "cuda":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kw["torch_dtype"] = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    if device == "cpu":
        model.to(device)
    model.eval()
    return model, tok, device


def _pointwise(
    model, tok, cand_sub, mode, is_qwen3, infer_kwargs, hist_text_fn, item_text
):
    instr = _INSTRUCTION_LOGPROB if mode == "logprob" else _INSTRUCTION_GENERATE

    def build(idx, row):
        return _build_baseline_prompt(
            tok,
            hist_text_fn(int(row.user_idx)),
            item_text.get(int(row.item_idx), ""),
            is_qwen3,
            instr,
        )

    kw = dict(infer_kwargs)
    kw["scoring_mode"] = mode
    scores, *_ = _score_candidates(
        model, tok, cand_sub, build, label=f"variant:{mode}", **kw
    )
    if mode == "logprob":
        scores = _logodds_to_prob(scores)
    return cand_sub[["user_idx", "item_idx"]].assign(score=scores)


def _listwise(
    model, tok, cand_sub, is_qwen3, device, k, max_length, hist_text_fn, item_text
):
    """One prompt per user lists the candidate pool; the LLM returns an ordered id list."""
    import torch

    rows = []
    users = cand_sub.user_idx.drop_duplicates().tolist()
    for u in users:
        items = cand_sub.loc[cand_sub.user_idx == u, "item_idx"].astype(int).tolist()
        listing = "\n".join(
            f"{j}: {item_text.get(it, '')[:120]}" for j, it in enumerate(items)
        )
        content = (
            "You are a recommender-system reranker.\n\nUser history:\n"
            f"{hist_text_fn(int(u))}\n\nCandidate items (number: description):\n{listing}\n\n"
            f"Return the {k} item numbers the user is most likely to like, best first, "
            "as a comma-separated list of numbers only. No explanation."
        )
        prompt = _apply_chat_template(tok, content, is_qwen3)
        inp = tok(
            [prompt], return_tensors="pt", truncation=True, max_length=max_length
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=4 * k + 16,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen = tok.decode(out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True)
        order = [int(x) for x in re.findall(r"\d+", gen) if int(x) < len(items)]
        seen, ranked = set(), []
        for j in order:
            if j not in seen:
                seen.add(j)
                ranked.append(items[j])
        ranked += [
            it for j, it in enumerate(items) if j not in seen
        ]  # un-listed items keep pool order
        n = len(ranked)
        rows += [
            {"user_idx": int(u), "item_idx": it, "score": float(n - r)}
            for r, it in enumerate(ranked)
        ]
    return pd.DataFrame(rows)


def _pairwise(
    model,
    tok,
    cand_sub,
    is_qwen3,
    device,
    max_length,
    hist_text_fn,
    item_text,
    n_pairs,
    seed,
):
    """Rank by win-count over n_pairs random pairs per user (Yes='first is better')."""
    import torch

    rng = np.random.default_rng(seed)
    rows = []
    for u in cand_sub.user_idx.drop_duplicates().tolist():
        items = cand_sub.loc[cand_sub.user_idx == u, "item_idx"].astype(int).tolist()
        wins = dict.fromkeys(items, 0)
        if len(items) < 2:
            rows += [{"user_idx": int(u), "item_idx": it, "score": 0.0} for it in items]
            continue
        pairs = [
            tuple(rng.choice(items, size=2, replace=False)) for _ in range(n_pairs)
        ]
        prompts, order = [], []
        for a, b in pairs:
            content = (
                "You are a recommender-system reranker.\n\nUser history:\n"
                f"{hist_text_fn(int(u))}\n\nItem A: {item_text.get(a, '')[:120]}\n"
                f"Item B: {item_text.get(b, '')[:120]}\n\n"
                'Which item will the user prefer? Answer with only "A" or "B".'
            )
            prompts.append(_apply_chat_template(tok, content, is_qwen3))
            order.append((a, b))
        inp = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=2, do_sample=False, pad_token_id=tok.pad_token_id
            )
        for seq, (a, b) in zip(out, order):
            ans = (
                tok.decode(seq[inp["input_ids"].shape[1] :], skip_special_tokens=True)
                .strip()
                .upper()
            )
            wins[a if ans.startswith("A") else b] += 1
        rows += [
            {"user_idx": int(u), "item_idx": it, "score": float(w)}
            for it, w in wins.items()
        ]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--config", default="configs/local.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", default="logprob,score,listwise")
    ap.add_argument(
        "--pool", default="positive", help="'positive' or 'e2e:<retriever>'"
    )
    ap.add_argument("--retrieval_dir", default=None)
    ap.add_argument("--n_users", type=int, default=150)
    ap.add_argument(
        "--top_m", type=int, default=50, help="candidates per user fed to the reranker"
    )
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--n_pairs", type=int, default=60, help="pairwise comparisons per user"
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    seed_everything(args.seed)
    rcfg = cfg.get("rerank", {})
    sp = Path(args.data_dir).resolve() / "processed" / args.dataset / f"s{args.seed}"
    out = Path(args.output_dir).resolve() / f"{args.dataset}-rerankvar-s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(sp / "test.csv")
    train = pd.read_csv(sp / "train.csv")
    items = pd.read_csv(sp / "items_mapped.csv")
    item_stats = pd.read_csv(sp / "item_stats.csv")
    item_text = items.set_index("item_idx")["text"].fillna("").astype(str).to_dict()
    hist_map = (
        train.sort_values("timestamp")
        .groupby("user_idx")["item_idx"]
        .apply(lambda s: s.astype(int).tolist())
        .to_dict()
    )
    hist_window = cfg.get("max_history") or 10

    def hist_text_fn(u):
        return " ; ".join(
            item_text.get(i, "") for i in hist_map.get(u, [])[-hist_window:]
        )

    # candidate pool
    if args.pool.startswith("e2e:"):
        retr = args.pool.split(":", 1)[1]
        rdir = (
            Path(args.retrieval_dir)
            if args.retrieval_dir
            else Path(args.output_dir).resolve()
            / f"{args.dataset}-retrieval-s{args.seed}-N200"
        )
        cand = pd.read_csv(rdir / f"{retr}_scores.csv")[["user_idx", "item_idx"]]
    else:
        cand = pd.read_csv(sp / "candidates.csv")[["user_idx", "item_idx"]]

    rng = np.random.default_rng(args.seed)
    pool_users = cand.user_idx.unique()
    users = set(
        rng.choice(
            pool_users, size=min(args.n_users, len(pool_users)), replace=False
        ).tolist()
    )
    cand = (
        cand[cand.user_idx.isin(users)]
        .groupby("user_idx")
        .head(args.top_m)
        .reset_index(drop=True)
    )
    test_sub = test[test.user_idx.isin(users)].copy()
    logger.info(
        "[variants] users=%d candidate_pairs=%d top_m=%d pool=%s",
        len(users),
        len(cand),
        args.top_m,
        args.pool,
    )

    model_name = rcfg.get("llm_model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    model, tok, device = _load_model(model_name, rcfg.get("llm_load_in_4bit", False))
    is_qwen3 = Path(model_name).name.lower().startswith("qwen3")
    yes_ids = _answer_token_ids(tok, ["Yes", "yes", "YES"])
    no_ids = _answer_token_ids(tok, ["No", "no", "NO"])
    ov = set(yes_ids) & set(no_ids)
    yes_ids = [i for i in yes_ids if i not in ov]
    no_ids = [i for i in no_ids if i not in ov]
    max_length = rcfg.get("llm_max_length", 1024)
    infer_kwargs = dict(
        yes_ids=yes_ids,
        no_ids=no_ids,
        batch_size=rcfg.get("llm_batch_size", 8),
        max_length=max_length,
        max_new_tokens=8,
        log_every=rcfg.get("llm_log_every", 10),
        device=device,
    )
    is_tail = item_stats.set_index("item_idx")["is_tail"]

    summary = []
    for v in [x.strip() for x in args.variants.split(",") if x.strip()]:
        logger.info("[variants] === %s ===", v)
        if v in ("logprob", "score"):
            scored = _pointwise(
                model, tok, cand, v, is_qwen3, infer_kwargs, hist_text_fn, item_text
            )
        elif v == "listwise":
            scored = _listwise(
                model,
                tok,
                cand,
                is_qwen3,
                device,
                args.k,
                max_length,
                hist_text_fn,
                item_text,
            )
        elif v == "pairwise":
            scored = _pairwise(
                model,
                tok,
                cand,
                is_qwen3,
                device,
                max_length,
                hist_text_fn,
                item_text,
                args.n_pairs,
                args.seed,
            )
        else:
            logger.warning("unknown variant %s — skipped", v)
            continue
        scored.to_csv(out / f"{v}_scores.csv", index=False)
        m = evaluate_rankings(
            scored,
            test_sub,
            cfg["metrics"]["k_list"],
            is_tail,
            top_candidates=args.top_m,
        )
        m["variant"] = v
        summary.append(m)
        logger.info(
            "[variants] %s | recall@%d=%.4f ndcg@%d=%.4f",
            v,
            args.k,
            m.get(f"recall@{args.k}", float("nan")),
            args.k,
            m.get(f"ndcg@{args.k}", float("nan")),
        )

    sm = pd.DataFrame(summary)
    sm.to_csv(out / "variants_summary.csv", index=False)
    cols = ["variant"] + [f"recall@{args.k}", f"ndcg@{args.k}", f"mrr@{args.k}"]
    print(
        f"\n=== prompt/scoring variants ({args.dataset} | {len(users)} users | pool={args.pool}) ==="
    )
    print(sm[[c for c in cols if c in sm.columns]].to_string(index=False))
    print(f"\nwrote {out}/variants_summary.csv")


if __name__ == "__main__":
    main()
