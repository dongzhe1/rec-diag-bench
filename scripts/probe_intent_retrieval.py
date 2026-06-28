"""Feasibility probe for Direction 1: LLM-generated INTENT as a retrieval query for cold-start.

Hypothesis: the LLM's semantic strength is wasted at reranking (coverage-capped), but useful
at RETRIEVAL. Instead of retrieving from the user's history alone, we ask an LLM to generate a
few short "next-intent" phrases from the history, embed them with a dense encoder, and retrieve
cold/new items by those phrases. We then test whether this lifts coverage@200 over history-only
dense retrieval, ESPECIALLY on item_cold / item_new (the regimes where history retrieval fails).

This is a GO/NO-GO probe, not a finished method. If intent retrieval lifts cold/new coverage,
Direction 1 has signal; if not, switch directions before investing weeks.

GPU/LLM required (generation + catalogue embedding). Keep --n_users small for a quick read.

Usage (cluster):
  python scripts/probe_intent_retrieval.py --dataset yelp-Philadelphia-Restaurants \
      --n_users 200 --n_intents 5 \
      --encoder BAAI/bge-large-en-v1.5 \
      --llm Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCEN = ["is_item_new", "is_item_cold", "is_long_tail", "is_user_cold", "is_warm"]


def _load_encoder(path_or_id, device):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(path_or_id, device=device)


def _load_llm(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if dev == "cuda" else torch.float32),
        device_map=("auto" if dev == "cuda" else None),
    )
    if dev == "cpu":
        model.to(dev)
    model.eval()
    return model, tok, dev


def _gen_intents(
    model, tok, dev, hist_texts, n_intents, is_qwen3, batch=8, max_new=160
):
    import torch

    prompts = []
    for h in hist_texts:
        content = (
            f"A user has recently interacted with these items:\n{h}\n\n"
            f"List {n_intents} short phrases (3 to 8 words each) describing other items this "
            f"user is likely to want next. Make the phrases specific and varied. "
            f"Output one phrase per line, with no numbering or extra text."
        )
        msgs = [{"role": "user", "content": content}]
        if tok.chat_template:
            kw = dict(tokenize=False, add_generation_prompt=True)
            if is_qwen3:
                kw["enable_thinking"] = False
            prompts.append(tok.apply_chat_template(msgs, **kw))
        else:
            prompts.append(content)
    out = []
    for i in range(0, len(prompts), batch):
        b = prompts[i : i + batch]
        inp = tok(
            b, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(model.device)
        with torch.no_grad():
            g = model.generate(
                **inp,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        plen = inp["input_ids"].shape[1]
        for seq in g:
            txt = tok.decode(seq[plen:], skip_special_tokens=True)
            lines = [
                line.strip(" -*\t.0123456789)") for line in txt.splitlines() if line.strip()
            ]
            lines = [line for line in lines if 2 <= len(line) <= 80]
            out.append(lines[:n_intents])
    return out


def _topn_excl(sims, seen, N):
    """top-N item indices by sims (1-D), excluding seen."""
    order = np.argsort(-sims)
    res = []
    for it in order:
        if it in seen:
            continue
        res.append(int(it))
        if len(res) >= N:
            break
    return res


def _quota_union(hist_list, intent_list, N, q):
    """Pool of N items: up to (N-q) best history items + up to q best intent items (dedup; backfill
    from history if intent is short). Guarantees intent contributes regardless of similarity scale.
    """
    out, seen = [], set()
    for it in hist_list[: max(N - q, 0)]:
        if it not in seen:
            seen.add(it)
            out.append(it)
    for it in intent_list:
        if len(out) >= N:
            break
        if it not in seen:
            seen.add(it)
            out.append(it)
    for it in hist_list:  # backfill if still short (intent had few unique items)
        if len(out) >= N:
            break
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out[:N]


def _coverage(topn_by_user, gold, flags, users):
    hit = {u: (gold.get(u) in topn_by_user.get(u, [])) for u in users}
    out = {"overall": float(np.mean([hit[u] for u in users]))}
    for c in SCEN:
        us = [u for u in users if flags.get(c, {}).get(u, False)]
        out[c.replace("is_", "")] = (
            (float(np.mean([hit[u] for u in us])), len(us)) if us else (float("nan"), 0)
        )
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument(
        "--n_users", type=int, default=200, help="random test users to probe"
    )
    ap.add_argument("--n_intents", type=int, default=5)
    ap.add_argument("--max_hist", type=int, default=10)
    ap.add_argument("--N", type=int, default=200, help="pool size for coverage@N")
    ap.add_argument(
        "--intent_quota",
        type=int,
        default=-1,
        help="slots in the N-pool reserved for LLM-intent items in the combined retriever "
        "(-1 => N//2). The rest go to history. Replaces the old per-item max fusion.",
    )
    ap.add_argument(
        "--encoder", default=os.environ.get("PROBE_ENCODER", "BAAI/bge-large-en-v1.5")
    )
    ap.add_argument("--llm", default=os.environ.get("PROBE_LLM", "Qwen/Qwen3-8B"))
    ap.add_argument("--enc_batch", type=int, default=256)
    args = ap.parse_args()

    sp = Path(args.data_dir) / "processed" / args.dataset / f"s{args.seed}"
    train = pd.read_csv(sp / "train.csv")
    test = pd.read_csv(sp / "test.csv")
    items = pd.read_csv(sp / "items_mapped.csv")
    n_items = int(items.item_idx.max()) + 1
    item_text = items.set_index("item_idx")["text"].fillna("").astype(str).to_dict()
    gold = dict(zip(test.user_idx.astype(int), test.item_idx.astype(int)))
    flags = {
        c: dict(zip(test.user_idx.astype(int), test[c].astype(bool)))
        for c in SCEN
        if c in test.columns
    }
    hist = (
        train.sort_values("timestamp")
        .groupby("user_idx")["item_idx"]
        .apply(lambda s: [int(x) for x in s])
        .to_dict()
    )
    seen = {u: set(v) for u, v in hist.items()}

    rng = np.random.default_rng(args.seed)
    cand_users = [u for u in test.user_idx.astype(int).unique() if u in hist]
    users = sorted(
        rng.choice(
            cand_users, size=min(args.n_users, len(cand_users)), replace=False
        ).tolist()
    )
    print(
        f"[probe] {args.dataset} | users={len(users)} | n_items={n_items} | "
        f"encoder={Path(args.encoder).name} | llm={Path(args.llm).name} | n_intents={args.n_intents}"
    )

    import torch

    enc = _load_encoder(args.encoder, "cuda" if torch.cuda.is_available() else "cpu")
    texts = [item_text.get(i, "") for i in range(n_items)]
    print(f"[probe] embedding {n_items:,} item texts...")
    item_emb = enc.encode(
        texts,
        batch_size=args.enc_batch,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    # ---- baseline: history-mean dense retrieval ----
    hist_top = {}
    for u in users:
        h = hist[u][-args.max_hist :]
        q = item_emb[h].mean(0)
        q = q / (np.linalg.norm(q) + 1e-9)
        hist_top[u] = _topn_excl(item_emb @ q, seen[u], args.N)

    # ---- LLM intent generation ----
    model, tok, dev = _load_llm(args.llm)
    is_qwen3 = Path(args.llm).name.lower().startswith("qwen3")
    hist_texts = [
        " ; ".join(item_text.get(i, "") for i in hist[u][-args.max_hist :])
        for u in users
    ]
    print("[probe] generating intents...")
    intents = _gen_intents(model, tok, dev, hist_texts, args.n_intents, is_qwen3)
    # show a couple of examples for sanity
    for u, it in list(zip(users, intents))[:3]:
        print(f"  user {u} hist: {hist_texts[users.index(u)][:90]}")
        print(f"    intents: {it}")

    # ---- intent + combined retrieval ----
    # Combined uses a QUOTA UNION of the two ranked pools (not per-item max, which lets the
    # higher-magnitude history similarities bury intent's item_new picks). We reserve `intent_quota`
    # of the N slots for intent's top items and fill the rest from history (backfilling either side).
    q = args.intent_quota if args.intent_quota >= 0 else args.N // 2
    intent_top, comb_top = {}, {}
    for u, phrases in zip(users, intents):
        if not phrases:
            intent_top[u] = []
            comb_top[u] = hist_top[u]
            continue
        pe = enc.encode(
            phrases,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        sims = (item_emb @ pe.T).max(axis=1)  # best similarity to any intent phrase
        intent_top[u] = _topn_excl(sims, seen[u], args.N)
        comb_top[u] = _quota_union(hist_top[u], intent_top[u], args.N, q)

    cov_h = _coverage(hist_top, gold, flags, users)
    cov_i = _coverage(intent_top, gold, flags, users)
    cov_c = _coverage(comb_top, gold, flags, users)

    cols = ["overall", "item_new", "item_cold", "long_tail", "user_cold", "warm"]

    def fmt(cov):
        cells = [f"{cov['overall']:.4f}"]
        for c in cols[1:]:
            v = cov.get(c)
            cells.append(f"{v[0]:.3f}" if isinstance(v, tuple) and v[1] else "-")
        return cells

    print(f"\n=== coverage@{args.N} (probe; {len(users)} users) ===")
    print(f"{'method':<22}" + "".join(f"{c:>11}" for c in cols))
    for name, cov in [
        ("history-only (base)", cov_h),
        ("LLM-intent only", cov_i),
        ("history + intent", cov_c),
    ]:
        print(f"{name:<22}" + "".join(f"{x:>11}" for x in fmt(cov)))

    dn = (
        cov_c["item_new"][0] - cov_h["item_new"][0]
        if isinstance(cov_h.get("item_new"), tuple)
        else float("nan")
    )
    dc = (
        cov_c["item_cold"][0] - cov_h["item_cold"][0]
        if isinstance(cov_h.get("item_cold"), tuple)
        else float("nan")
    )
    do = cov_c["overall"] - cov_h["overall"]
    print(
        f"\n[VERDICT] combined vs history-only:  overall {do:+.4f} | "
        f"item_new {dn:+.4f} | item_cold {dc:+.4f}"
    )
    print(
        "  GO if item_new / item_cold coverage rises meaningfully (intent reaches items history can't)."
    )
    print(
        "  NO-GO if flat/negative -> the LLM intent does not add reachable coverage; switch direction."
    )


if __name__ == "__main__":
    main()
