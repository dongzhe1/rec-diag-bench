"""Rerank candidates using a cross-encoder or bi-encoder model.

Supports SentenceTransformers CrossEncoder, HuggingFace sequence-classification
models (e.g. Nemotron-NAS), batched inference with BF16 autocast, and
optional torch.compile for GPU acceleration.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from coldtail.metrics import (
    evaluate_rankings,
    limit_candidates_per_user,
    subgroup_metrics,
)

from .common import _init_nvml, _log_cpu_stats, _log_gpu_info, _log_gpu_stats

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)


def build_user_hist(
    train: pd.DataFrame,
    items: pd.DataFrame,
    max_hist: int = 10,
) -> tuple[dict, pd.Series]:
    item_text = items.set_index("item_idx")["text"].fillna("").astype(str)

    last_n = (
        train.sort_values("timestamp")
        .groupby("user_idx", sort=False)
        .tail(max_hist)
        .groupby("user_idx")["item_idx"]
        .agg(list)
        .reset_index()
        .rename(columns={"item_idx": "hist_idxs"})
    )
    last_n["hist_text"] = last_n["hist_idxs"].apply(
        lambda idxs: " ; ".join(item_text.get(i, "") for i in idxs)
    )
    user_hist = last_n.set_index("user_idx")["hist_text"].to_dict()
    return user_hist, item_text


class _PairDataset:
    def __init__(
        self,
        candidates: pd.DataFrame,
        user_hist: dict,
        item_text: pd.Series,
        model_type: str = "cross_encoder",
    ):
        self.candidates = candidates.reset_index(drop=True)
        self.user_hist = user_hist
        self.item_text = item_text
        self.model_type = model_type

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, idx: int):
        row = self.candidates.iloc[idx]
        doc = self.item_text.get(row["item_idx"], "")
        hist = self.user_hist.get(row["user_idx"], "")
        if self.model_type == "nemotron":
            return (f"question:Recommend items similar to: {hist} \n \n passage:{doc}",)
        return "Recommend items similar to: " + hist, doc


class _CollateFn:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        if len(batch[0]) == 1:
            # Nemotron: single string, tokenize directly
            texts = [item[0] for item in batch]
            return self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        # Cross-encoder: (query, doc) pair
        queries, docs = zip(*batch)
        return self.tokenizer(
            list(queries),
            list(docs),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


def _predict_fast(
    model,
    candidates: pd.DataFrame,
    user_hist: dict,
    item_text: pd.Series,
    batch_size: int = 512,
    num_workers: int = 0,
    max_length: int = 512,
    log_interval: int = 10,
    model_type: str = "cross_encoder",
) -> np.ndarray:
    """Batched cross-encoder inference; returns score array."""
    import torch
    from torch.utils.data import DataLoader

    is_cuda = next(model.model.parameters()).is_cuda
    if is_cuda:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)

    dataset = _PairDataset(candidates, user_hist, item_text, model_type=model_type)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=is_cuda,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_CollateFn(model.tokenizer, max_length),
    )

    num_batches = len(loader)
    num_pairs = len(dataset)
    all_scores = np.empty(num_pairs, dtype=np.float32)

    is_cuda = next(model.model.parameters()).is_cuda
    model.model.eval()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    _log_gpu_stats("inference_start")
    _log_cpu_stats("inference_start")
    t_start = time.perf_counter()

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16) if is_cuda else torch.no_grad()
    )
    with torch.no_grad(), autocast_ctx:
        cursor = 0
        for batch_idx, batch in enumerate(loader):
            if is_cuda:
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            logits = model.model(**batch).logits

            raw = logits.float().cpu()
            if raw.dim() == 1:
                scores_batch = raw.numpy()
            elif raw.dim() == 2 and raw.shape[-1] == 1:
                scores_batch = raw.squeeze(-1).numpy()
            elif raw.dim() == 2:
                scores_batch = raw[:, 1].numpy()
            else:
                scores_batch = raw[:, 0, 0].numpy()

            lo = cursor
            hi = cursor + len(scores_batch)
            all_scores[lo:hi] = scores_batch
            cursor = hi

            del logits, batch

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == num_batches:
                elapsed = time.perf_counter() - t_start
                throughput = cursor / elapsed
                logger.info(
                    f"[Inference] batch {batch_idx + 1:,}/{num_batches:,} | "
                    f"progress={cursor:,}/{num_pairs:,} | throughput={throughput:,.0f}pairs/s"
                )
                _log_gpu_stats(f"batch_{batch_idx + 1}")
                _log_cpu_stats(f"batch_{batch_idx + 1}")

    total_time = time.perf_counter() - t_start
    logger.info(
        f"[Inference] complete | num_pairs={num_pairs:,} | "
        f"time={total_time:.1f}s | throughput={num_pairs / total_time:,.0f}pairs/s"
    )
    _log_gpu_stats("inference_complete")
    _log_cpu_stats("inference_complete")

    return all_scores


def run_cross_encoder_rerank(
    processed_dir: Path | str,
    out_dir: Path | str,
    cfg: dict,
    top_candidates: int | None = None,
    rerank_users_path: Path | str | None = None,
    timing_collector=None,
) -> None:

    import torch
    from sentence_transformers import CrossEncoder
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    processed_dir_path = Path(processed_dir)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rerank_cfg = cfg.get("rerank", {})
    model_name = rerank_cfg.get("cross_encoder_model", "BAAI/bge-reranker-base")
    model_type = rerank_cfg.get("cross_encoder_model_type", "cross_encoder")
    sample_users = rerank_cfg.get("sample_users", 200)
    batch_size = rerank_cfg.get("cross_encoder_batch_size", 512)
    num_workers = rerank_cfg.get("cross_encoder_num_workers", 0)
    max_length = rerank_cfg.get("cross_encoder_max_length", 512)
    use_compile = rerank_cfg.get("cross_encoder_compile", True)
    log_interval = rerank_cfg.get("cross_encoder_log_interval", 10)
    chunk_size = rerank_cfg.get("cross_encoder_chunk_size", 200_000)

    _init_nvml()
    _log_gpu_info()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"[Config] model={model_name} | type={model_type} | device={device} | "
        f"batch_size={batch_size} | num_workers={num_workers}"
    )

    t0 = time.perf_counter()
    if model_type == "nemotron":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        hf_model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            .eval()
            .to(device)
        )

        if hf_model.config.pad_token_id is None:
            hf_model.config.pad_token_id = tokenizer.eos_token_id

        class _NemotronWrapper:
            def __init__(self, m, tok):
                self.model = m
                self.tokenizer = tok

        model = _NemotronWrapper(hf_model, tokenizer)
        logger.info(f"[Model] nemotron loaded | time={time.perf_counter() - t0:.1f}s")

    else:
        model = CrossEncoder(model_name, device=device)
        if device == "cuda":
            model.model = model.model.to(torch.bfloat16)
            logger.info("[Model] cast to BF16")
        logger.info(
            f"[Model] cross_encoder loaded | time={time.perf_counter() - t0:.1f}s"
        )

    _log_gpu_stats("model_loaded")
    _log_cpu_stats("model_loaded")

    if device == "cuda" and use_compile:
        logger.info("[Model] compiling with torch.compile...")
        t0 = time.perf_counter()
        model.model = torch.compile(model.model, mode="reduce-overhead")
        logger.info(f"[Model] compile done | time={time.perf_counter() - t0:.1f}s")

    logger.info("[Data] loading CSVs...")
    t0 = time.perf_counter()
    train = pd.read_csv(processed_dir_path / "train.csv")
    test = pd.read_csv(processed_dir_path / "test.csv")
    candidates = pd.read_csv(processed_dir_path / "candidates.csv")
    candidates = limit_candidates_per_user(candidates, top_candidates)
    items = pd.read_csv(processed_dir_path / "items_mapped.csv")
    item_stats = pd.read_csv(processed_dir_path / "item_stats.csv")
    logger.info(f"[Data] CSVs loaded | time={time.perf_counter() - t0:.1f}s")

    if rerank_users_path is not None and Path(rerank_users_path).exists():
        users = pd.read_csv(rerank_users_path)["user_idx"].tolist()
        logger.info(f"[Data] using stratified rerank subset | num_users={len(users):,}")
    else:
        users = test.user_idx.drop_duplicates().head(sample_users).tolist()
        logger.info(
            f"[Data] using head({sample_users}) fallback | num_users={len(users):,}"
        )

    test_sub = test[test.user_idx.isin(users)].copy()
    cand_sub = candidates[candidates.user_idx.isin(users)].copy()
    logger.info(
        f"[Data] subset | num_users={len(users):,} | test_rows={len(test_sub):,} | "
        f"candidate_pairs={len(cand_sub):,}"
    )

    logger.info("[Data] building user history lookups...")
    t0 = time.perf_counter()
    hist_window = (
        cfg.get("max_history") or 10
    )  # from --max_history CLI arg; default last 10
    logger.info(
        f"[CE] history window per user = last {hist_window} interactions (max_history={cfg.get('max_history')})"
    )
    user_hist, item_text = build_user_hist(train, items, max_hist=hist_window)
    logger.info(f"[Data] history built | time={time.perf_counter() - t0:.1f}s")

    num_pairs = len(cand_sub)
    effective_chunk_size = chunk_size or num_pairs
    num_chunks = (num_pairs + effective_chunk_size - 1) // effective_chunk_size

    all_chunk_scores: list[np.ndarray] = []
    t_infer_start = time.perf_counter()

    for chunk_idx in range(num_chunks):
        lo = chunk_idx * effective_chunk_size
        hi = min(lo + effective_chunk_size, num_pairs)
        chunk = cand_sub.iloc[lo:hi]

        if num_chunks > 1:
            logger.info(
                f"[Chunk] {chunk_idx + 1}/{num_chunks} | rows {lo:,}-{hi:,} of {num_pairs:,}"
            )
            _log_cpu_stats(f"chunk_{chunk_idx + 1}")

        if device == "cuda":
            chunk_scores = _predict_fast(
                model,
                chunk,
                user_hist,
                item_text,
                batch_size=batch_size,
                num_workers=num_workers,
                max_length=max_length,
                log_interval=log_interval,
                model_type=model_type,
            )
        else:
            if chunk_idx == 0:
                logger.info("[Inference] CUDA unavailable | falling back to CPU")
            if model_type == "nemotron":
                chunk_scores = _predict_fast(
                    model,
                    chunk,
                    user_hist,
                    item_text,
                    batch_size=batch_size,
                    num_workers=0,
                    max_length=max_length,
                    log_interval=log_interval,
                    model_type=model_type,
                )
            else:
                cand_texts = chunk["item_idx"].map(item_text).fillna("")
                hist_texts = chunk["user_idx"].map(user_hist).fillna("")
                pairs = list(
                    zip("Recommend items similar to: " + hist_texts, cand_texts)
                )
                chunk_scores = model.predict(
                    pairs, batch_size=batch_size, show_progress_bar=True
                )
                del pairs

        all_chunk_scores.append(chunk_scores)
        del chunk, chunk_scores

    scores = np.concatenate(all_chunk_scores)
    del all_chunk_scores

    total_infer = time.perf_counter() - t_infer_start
    logger.info(
        f"[Inference] total | num_pairs={num_pairs:,} | time={total_infer:.1f}s | "
        f"throughput={num_pairs / total_infer:,.0f}pairs/s"
    )

    if timing_collector is not None:
        import torch as _torch

        gpu_peak = None
        if device == "cuda":
            gpu_peak = _torch.cuda.max_memory_allocated() / 1024 / 1024
        timing_collector.record("cross_encoder", num_pairs, total_infer, gpu_peak)

    assert len(scores) == len(cand_sub), (
        f"Score count mismatch: {len(scores)} vs {len(cand_sub)}"
    )
    scored = cand_sub[["user_idx", "item_idx"]].assign(score=scores)
    scored.to_csv(output_path / "cross_encoder_scores.csv", index=False)
    logger.info(
        f"[Output] scores saved | path={output_path / 'cross_encoder_scores.csv'}"
    )

    is_tail = item_stats.set_index("item_idx")["is_tail"]
    metrics = evaluate_rankings(
        scored,
        test_sub,
        cfg["metrics"]["k_list"],
        is_tail,
        top_candidates=top_candidates,
    )
    metrics["model"] = "cross_encoder"
    pd.DataFrame([metrics]).to_csv(
        output_path / "cross_encoder_metrics.csv", index=False
    )

    sub = subgroup_metrics(
        scored, test_sub, cfg["metrics"]["k_list"], top_candidates=top_candidates
    )
    if len(sub):
        sub["model"] = "cross_encoder"
        sub.to_csv(output_path / "cross_encoder_subgroup_metrics.csv", index=False)

    _log_gpu_stats("pipeline_complete")
    _log_cpu_stats("pipeline_complete")
    logger.info("[CrossEncoder] done")
