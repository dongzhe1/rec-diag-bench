"""GALA ablation wrapper — delegates to run_llm_rerank."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_gala_ablations(
    processed_dir: Path | str,
    out_dir: Path | str,
    cfg: dict,
    top_candidates: int | None = None,
    variants: list[str] | None = None,
    rerank_users_path: Path | str | None = None,
    timing_collector=None,
) -> None:
    """Run GALA ablation variants via run_llm_rerank."""
    from .run_llm_rerank import run_llm_rerank

    run_llm_rerank(
        processed_dir,
        out_dir,
        cfg,
        top_candidates=top_candidates,
        rerank_users_path=rerank_users_path,
        timing_collector=timing_collector,
        run_baseline=False,
        gala_variants=variants,
    )
