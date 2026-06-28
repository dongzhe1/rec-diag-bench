"""Unified LLM reranker: baseline + GALA variants with single model load."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from coldtail.metrics import (
    evaluate_rankings,
    limit_candidates_per_user,
    subgroup_metrics,
)

from .common import _init_nvml, _log_cpu_stats, _log_gpu_info, _log_gpu_stats
from .gala_evidence import build_graph_evidence

logger = logging.getLogger(__name__)


_INSTRUCTION_GENERATE = (
    "Score how likely the user will like the candidate from 0 to 100.\n"
    "Return only one integer. No explanation."
)
_INSTRUCTION_LOGPROB = (
    'Will the user like this candidate item? Answer with only "Yes" or "No".'
)

_BASELINE_TEMPLATE = """\
You are a recommender-system reranker.

User history:
{history}

Candidate item:
{candidate}

{instruction}"""

_GALA_TEMPLATE = """\
You are a recommender-system reranker.

User history:
{history}

Candidate item:
{candidate}
{evidence_block}
{instruction}"""

_EVIDENCE_BLOCK = """
Graph evidence:
{lines}"""


GALA_VARIANTS = [
    {"name": "gala", "use_evidence": True, "use_cooccur": True, "use_tail": True},
    {
        "name": "gala_no_evidence",
        "use_evidence": False,
        "use_cooccur": False,
        "use_tail": False,
    },
    {
        "name": "gala_no_cooccur",
        "use_evidence": True,
        "use_cooccur": False,
        "use_tail": True,
    },
    {
        "name": "gala_no_tail",
        "use_evidence": True,
        "use_cooccur": True,
        "use_tail": False,
    },
]


def _parse_score_or_none(text: str) -> float | None:
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    if not nums:
        return None
    return max(0.0, min(100.0, float(nums[0])))


def parse_score(text: str) -> float:
    val = _parse_score_or_none(text)
    return 0.0 if val is None else val


def _answer_token_ids(tokenizer, words: list[str]) -> list[int]:
    ids: list[int] = []
    for w in words:
        for variant in (w, " " + w):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if enc:
                ids.append(enc[0])
    return sorted(set(ids))


def _apply_chat_template(tokenizer, user_content: str, is_qwen3: bool) -> str:
    has_template = getattr(tokenizer, "chat_template", None) is not None
    if has_template:
        kwargs: dict = dict(tokenize=False, add_generation_prompt=True)
        if is_qwen3:
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            **kwargs,
        )
    return user_content


def _build_baseline_prompt(
    tokenizer,
    hist_text: str,
    cand_text: str,
    is_qwen3: bool,
    instruction: str,
) -> str:
    content = _BASELINE_TEMPLATE.format(
        history=hist_text,
        candidate=cand_text,
        instruction=instruction,
    )
    return _apply_chat_template(tokenizer, content, is_qwen3)


def _evidence_lines(row, use_cooccur: bool, use_tail: bool) -> str:
    lines = []
    gcn_pct = float(row.graph_score_pct)
    sem_pct = float(row.semantic_score_pct)

    lines.append(
        f"- Collaborative signal (LightGCN): top {100 - gcn_pct:.0f}% "
        f"of candidates for this user."
    )
    lines.append(
        f"- Semantic similarity (TF-IDF): top {100 - sem_pct:.0f}% "
        f"of candidates for this user."
    )

    if use_cooccur:
        strength = int(row.cooccur_strength)
        if strength > 0:
            lines.append(
                f"- Co-occurrence: this item co-occurred with "
                f"{row.hist_overlap_text} from your history "
                f"({strength} shared item{'s' if strength > 1 else ''})."
            )
        else:
            lines.append("- Co-occurrence: no direct co-occurrence with your history.")

    if use_tail:
        pop_pct = float(row.popularity_pct)
        if row.is_tail:
            lines.append(
                f"- Popularity: long-tail item (bottom {pop_pct:.0f}% by interaction count)."
            )
        else:
            lines.append(
                f"- Popularity: popular item (top {100 - pop_pct:.0f}% by interaction count)."
            )

    return "\n".join(lines)


def _build_gala_prompt(
    tokenizer,
    hist_text: str,
    cand_text: str,
    evidence_row,
    is_qwen3: bool,
    use_evidence: bool,
    use_cooccur: bool,
    use_tail: bool,
    instruction: str,
) -> str:
    if use_evidence and evidence_row is not None:
        ev = _evidence_lines(evidence_row, use_cooccur=use_cooccur, use_tail=use_tail)
        evidence_block = _EVIDENCE_BLOCK.format(lines=ev)
    else:
        evidence_block = ""

    content = _GALA_TEMPLATE.format(
        history=hist_text,
        candidate=cand_text,
        evidence_block=evidence_block,
        instruction=instruction,
    )
    return _apply_chat_template(tokenizer, content, is_qwen3)


def _score_candidates(
    model,
    tokenizer,
    cand_sub: pd.DataFrame,
    build_prompt: Callable[[int, object], str],
    *,
    scoring_mode: str,
    yes_ids: list[int],
    no_ids: list[int],
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    log_every: int,
    device: str,
    label: str,
) -> tuple[np.ndarray, float, int, int]:
    """Run batched inference; returns (scores, elapsed_s, n_issue, n_cands)."""
    import torch

    num_cands = len(cand_sub)
    num_batches = (num_cands + batch_size - 1) // batch_size
    all_scores = np.empty(num_cands, dtype=np.float32)
    cursor = 0
    n_issue = 0

    logger.info(
        f"[{label}] scoring | num_candidates={num_cands:,} | num_batches={num_batches:,}"
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()

    for batch_idx in tqdm(range(num_batches), desc=f"{label} Reranking"):
        lo = batch_idx * batch_size
        hi = min(lo + batch_size, num_cands)

        batch_prompts = []
        for k, row in enumerate(cand_sub.iloc[lo:hi].itertuples(index=False)):
            batch_prompts.append(build_prompt(lo + k, row))

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        if scoring_mode == "logprob":
            with torch.no_grad():
                try:
                    outputs = model(**inputs, logits_to_keep=1)
                except TypeError as e:
                    if "logits_to_keep" not in str(e):
                        raise
                    outputs = model(**inputs)

                last_logits = outputs.logits[:, -1, :].float()
                yes_score = torch.logsumexp(last_logits[:, yes_ids], dim=-1)
                no_score = torch.logsumexp(last_logits[:, no_ids], dim=-1)
                batch_scores = (yes_score - no_score).cpu().numpy()

                log_z = torch.logsumexp(last_logits, dim=-1)
                answer_mass = (
                    (torch.exp(yes_score - log_z) + torch.exp(no_score - log_z))
                    .cpu()
                    .numpy()
                )
                n_issue += int((answer_mass < 0.5).sum())

            for s in batch_scores:
                all_scores[cursor] = s
                cursor += 1

            del (
                outputs,
                last_logits,
                yes_score,
                no_score,
                log_z,
                answer_mass,
                batch_scores,
            )
        else:
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            for seq in out_ids:
                gen = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                val = _parse_score_or_none(gen)
                if val is None:
                    n_issue += 1
                    val = 0.0
                all_scores[cursor] = val
                cursor += 1
            del out_ids

        del inputs, batch_prompts

        if device == "cuda" and batch_idx % log_every == 0:
            elapsed = time.perf_counter() - t_start
            throughput = cursor / elapsed if elapsed > 0 else 0
            logger.info(
                f"[{label}] batch {batch_idx:>5d}/{num_batches} | "
                f"progress={cursor:,}/{num_cands:,} | throughput={throughput:.1f}pairs/s"
            )
            _log_gpu_stats(f"batch_{batch_idx}")
            _log_cpu_stats(f"batch_{batch_idx}")

    total_time = time.perf_counter() - t_start
    logger.info(
        f"[{label}] inference complete | num_pairs={num_cands:,} | "
        f"time={total_time:.0f}s | throughput={num_cands / total_time:.1f}pairs/s"
    )
    return all_scores, total_time, n_issue, num_cands


def _log_scoring_diagnostic(
    scoring_mode: str, n_issue: int, n_total: int, label: str
) -> None:
    pct = 100 * n_issue / max(n_total, 1)
    if scoring_mode == "generate":
        log_fn = logger.warning if pct > 2 else logger.info
        log_fn(
            f"[{label}] parse-failure rate: {n_issue:,}/{n_total:,} ({pct:.2f}%) — scored 0.0"
        )
    else:
        log_fn = logger.warning if pct > 5 else logger.info
        log_fn(
            f"[{label}] weak Yes/No answer rate (mass<0.5): {n_issue:,}/{n_total:,} ({pct:.2f}%)"
        )


def _logodds_to_prob(scores: np.ndarray) -> np.ndarray:
    """Map log-odds to P(Yes) in [0, 100]."""
    return (100.0 / (1.0 + np.exp(-scores))).astype(np.float32)


def _save_and_eval(
    scores: np.ndarray,
    cand_sub: pd.DataFrame,
    test_sub: pd.DataFrame,
    item_stats: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
    model_label: str,
    top_candidates: int | None,
) -> None:
    scored = cand_sub[["user_idx", "item_idx"]].assign(score=scores)
    scored.to_csv(out_dir / f"{model_label}_scores.csv", index=False)
    logger.info(f"[Output] scores saved | {model_label}_scores.csv")

    is_tail = item_stats.set_index("item_idx")["is_tail"]
    metrics = evaluate_rankings(
        scored,
        test_sub,
        cfg["metrics"]["k_list"],
        is_tail,
        top_candidates=top_candidates,
    )
    metrics["model"] = model_label
    pd.DataFrame([metrics]).to_csv(out_dir / f"{model_label}_metrics.csv", index=False)

    sub = subgroup_metrics(
        scored, test_sub, cfg["metrics"]["k_list"], top_candidates=top_candidates
    )
    if len(sub):
        sub["model"] = model_label
        sub.to_csv(out_dir / f"{model_label}_subgroup_metrics.csv", index=False)


def run_llm_rerank(
    processed_dir: Path | str,
    out_dir: Path | str,
    cfg: dict,
    top_candidates: int | None = None,
    rerank_users_path: Path | str | None = None,
    timing_collector=None,
    *,
    run_baseline: bool = True,
    gala_variants: list[str] | None = None,
    candidates_override: pd.DataFrame | None = None,
) -> None:
    """Run baseline LLM + GALA variants with a single model load.

    candidates_override: if given (columns user_idx,item_idx), the LLM reranks this
    pool instead of the positive-controlled candidates.csv — used for end-to-end
    retrieve->rerank evaluation, where the gold item may be absent from the pool.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    p = Path(processed_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rerank_cfg = cfg.get("rerank", {})
    model_name = rerank_cfg.get("llm_model_name", "Qwen/Qwen2.5-1.5B-Instruct")
    sample_users = rerank_cfg.get("sample_users", 50)
    load_in_4bit = rerank_cfg.get("llm_load_in_4bit", False)
    batch_size = rerank_cfg.get("llm_batch_size", 8)
    log_every = rerank_cfg.get("llm_log_every", 10)
    max_length = rerank_cfg.get("llm_max_length", 1024)
    scoring_mode = rerank_cfg.get("llm_scoring_mode", "logprob")
    tail_bonus = rerank_cfg.get("gala_tail_bonus", 5.0)
    hist_window = (
        cfg.get("max_history") or 10
    )  # from --max_history CLI arg; default last 10
    logger.info(
        f"[LLM] history window per user = last {hist_window} interactions (max_history={cfg.get('max_history')})"
    )
    if scoring_mode not in ("logprob", "generate"):
        raise ValueError(
            f"llm_scoring_mode must be 'logprob' or 'generate', got {scoring_mode!r}"
        )
    instruction = (
        _INSTRUCTION_LOGPROB if scoring_mode == "logprob" else _INSTRUCTION_GENERATE
    )

    # Resolve GALA variant list
    if gala_variants is None:
        gala_to_run = list(GALA_VARIANTS)
    elif len(gala_variants) == 0:
        gala_to_run = []
    else:
        gala_to_run = [v for v in GALA_VARIANTS if v["name"] in gala_variants]

    _log_gpu_info()
    nvml_ok = _init_nvml()

    model_basename = Path(model_name).name.lower()
    is_qwen3 = model_basename.startswith("qwen3")
    max_new_tokens = 8

    logger.info(
        f"[LLM] loading reranker | model={model_name} | "
        f"scoring_mode={scoring_mode} | batch_size={batch_size} | "
        f"run_baseline={run_baseline} | gala_variants={[v['name'] for v in gala_to_run]}"
    )

    train = pd.read_csv(p / "train.csv")
    test = pd.read_csv(p / "test.csv")
    if candidates_override is not None:
        candidates = candidates_override[["user_idx", "item_idx"]].copy()
        logger.info(
            f"[LLM] using candidate override pool | rows={len(candidates):,} (end-to-end retrieve->rerank)"
        )
    else:
        candidates = pd.read_csv(p / "candidates.csv")
        candidates = limit_candidates_per_user(candidates, top_candidates)
    items = pd.read_csv(p / "items_mapped.csv")
    item_stats = pd.read_csv(p / "item_stats.csv")

    if rerank_users_path is not None and Path(rerank_users_path).exists():
        users = pd.read_csv(rerank_users_path)["user_idx"].tolist()
        logger.info(f"[Data] using stratified rerank subset | num_users={len(users):,}")
    else:
        users = test.user_idx.drop_duplicates().head(sample_users).tolist()
        logger.info(
            f"[Data] using head({sample_users}) fallback | num_users={len(users):,}"
        )

    test_sub = test[test.user_idx.isin(users)].copy()
    cand_sub = candidates[candidates.user_idx.isin(users)].copy().reset_index(drop=True)

    item_text_map = items.set_index("item_idx")["text"].fillna("").astype(str).to_dict()
    user_hist_map: dict[int, list[int]] = (
        train.sort_values("timestamp")
        .groupby("user_idx")["item_idx"]
        .apply(lambda s: s.astype(int).tolist())
        .to_dict()
    )

    logger.info(
        f"[Data] subset | num_users={len(users):,} | test_rows={len(test_sub):,} | "
        f"candidate_pairs={len(cand_sub):,}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[Model] device={device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, fix_mistral_regex=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    yes_ids: list[int] = []
    no_ids: list[int] = []
    if scoring_mode == "logprob":
        yes_ids = _answer_token_ids(tokenizer, ["Yes", "yes", "YES"])
        no_ids = _answer_token_ids(tokenizer, ["No", "no", "NO"])
        overlap = set(yes_ids) & set(no_ids)
        yes_ids = [i for i in yes_ids if i not in overlap]
        no_ids = [i for i in no_ids if i not in overlap]
        if not yes_ids or not no_ids:
            raise RuntimeError(
                f"Could not resolve distinct Yes/No token ids for {model_name}; "
                "use llm_scoring_mode='generate' for this tokenizer."
            )
        logger.info(f"[LLM] logprob scoring | yes_ids={yes_ids} | no_ids={no_ids}")

    load_kwargs: dict = dict(
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if load_in_4bit and device == "cuda":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = (
            torch.bfloat16 if device == "cuda" else torch.float32
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if device == "cpu":
        model.to(device)
    model.eval()

    if hasattr(model, "hf_device_map"):
        logger.info(f"[Model] hf_device_map={model.hf_device_map}")

    if device == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        _log_gpu_stats("model_loaded")
        _log_cpu_stats("model_loaded")

    infer_kwargs = dict(
        scoring_mode=scoring_mode,
        yes_ids=yes_ids,
        no_ids=no_ids,
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        log_every=log_every,
        device=device,
    )

    if run_baseline:

        def _baseline_prompt(idx: int, row) -> str:
            hist = user_hist_map.get(int(row.user_idx), [])[-hist_window:]
            hist_text = " ; ".join(item_text_map.get(int(i), "") for i in hist)
            cand_text = item_text_map.get(int(row.item_idx), "")
            return _build_baseline_prompt(
                tokenizer, hist_text, cand_text, is_qwen3, instruction
            )

        scores, elapsed, n_issue, n_total = _score_candidates(
            model,
            tokenizer,
            cand_sub,
            _baseline_prompt,
            label="LLM",
            **infer_kwargs,
        )
        _log_scoring_diagnostic(scoring_mode, n_issue, n_total, "LLM")
        if scoring_mode == "logprob":
            scores = _logodds_to_prob(scores)

        _save_and_eval(
            scores, cand_sub, test_sub, item_stats, cfg, out, "llm", top_candidates
        )

        if timing_collector is not None:
            gpu_peak = None
            if device == "cuda":
                gpu_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            timing_collector.record("llm", n_total, elapsed, gpu_peak)

    if gala_to_run:
        any_needs_evidence = any(v["use_evidence"] for v in gala_to_run)
        if any_needs_evidence:
            logger.info("[GALA] Building graph evidence...")
            t0 = time.perf_counter()
            evidence_df = build_graph_evidence(
                cand_sub,
                train,
                items,
                item_stats,
                scores_dir=out,
            )
            logger.info(
                f"[GALA] Graph evidence built in {time.perf_counter() - t0:.1f}s"
            )
        else:
            evidence_df = None

        for variant in gala_to_run:
            vname = variant["name"]
            v_evidence = variant["use_evidence"]
            v_cooccur = variant["use_cooccur"]
            v_tail = variant["use_tail"]

            logger.info(
                f"[GALA] === {vname} === evidence={v_evidence} cooccur={v_cooccur} tail={v_tail}"
            )

            def _gala_prompt(
                idx: int, row, _ev=v_evidence, _co=v_cooccur, _ta=v_tail
            ) -> str:
                hist = user_hist_map.get(int(row.user_idx), [])[-hist_window:]
                hist_text = " ; ".join(item_text_map.get(int(i), "") for i in hist)
                cand_text = item_text_map.get(int(row.item_idx), "")
                ev_row = evidence_df.iloc[idx] if evidence_df is not None else None
                return _build_gala_prompt(
                    tokenizer,
                    hist_text,
                    cand_text,
                    ev_row,
                    is_qwen3=is_qwen3,
                    use_evidence=_ev,
                    use_cooccur=_co,
                    use_tail=_ta,
                    instruction=instruction,
                )

            scores, elapsed, n_issue, n_total = _score_candidates(
                model,
                tokenizer,
                cand_sub,
                _gala_prompt,
                label=f"GALA {vname}",
                **infer_kwargs,
            )
            _log_scoring_diagnostic(scoring_mode, n_issue, n_total, f"GALA {vname}")
            if scoring_mode == "logprob":
                scores = _logodds_to_prob(scores)

            if v_tail and evidence_df is not None:
                is_tail_arr = evidence_df["is_tail"].values.astype(float)
                scores = scores + (is_tail_arr * tail_bonus).astype(np.float32)
                logger.info(
                    f"[GALA {vname}] tail bonus +{tail_bonus} applied to {int(is_tail_arr.sum()):,} items"
                )

            _save_and_eval(
                scores, cand_sub, test_sub, item_stats, cfg, out, vname, top_candidates
            )

            if timing_collector is not None:
                gpu_peak = None
                if device == "cuda":
                    gpu_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
                timing_collector.record(vname, n_total, elapsed, gpu_peak)

    if device == "cuda":
        _log_gpu_stats("pipeline_complete")
        _log_cpu_stats("pipeline_complete")

    if nvml_ok:
        try:
            import pynvml

            pynvml.nvmlShutdown()
        except Exception:
            pass

    logger.info("[LLM] done")
