"""Second-LLM generality runs (Qwen3-32B + Llama-3.3-70B-Instruct).

Runs the LLM reranker for two additional models beyond the main Qwen3-8B run,
hardcoded here (not read from any YAML config):

  * Qwen3-32B            — same family, 4x scale (bf16, no quantisation).
  * Llama-3.3-70B-Instruct — different family and larger scale (nf4 4-bit).

Models are loaded from ``$COLDTAIL_MODEL_DIR`` (falls back to HuggingFace Hub).
The split is reused read-only (build it with ``run_dataset.py --split_only``).

Results are written into the same base run dir as the main pipeline, with a
per-model suffix so the base ``llm_*`` files (Qwen3-8B) are never overwritten.

Usage:
    python scripts/run_second_llms.py --dataset ml-20m --seed 42 --top_k 200

GPU required (runs additional LLMs on the reranking task).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from coldtail.experiments.run_llm_rerank import run_llm_rerank
from coldtail.experiments.timing import TimingCollector
from coldtail.utils import seed_everything

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Hardcoded LLM configs (intentionally not loaded from any YAML).
MODEL_DIR = Path(os.environ.get("COLDTAIL_MODEL_DIR", os.path.expanduser("~/models")))

K_LIST = [5, 10, 20, 49]  # match the main-table rank runs (hpc.yaml)
MAX_LENGTH = 1024
SCORING_MODE = "logprob"  # keep identical to the Qwen3-8B main run
# Set True to also run the GALA variants for these models (≈5x the LLM cost,
# expensive for the 70B). Default: baseline LLM only for generality comparison.
RUN_GALA = False

MODELS = [
    {
        "suffix": "llama-70B",
        # files: llm-llama-70B_*.csv ; model col: llm-llama-70B
        "subdir": "Llama-3.3-70B-Instruct",
        "load_in_4bit": True,  # nf4 ~35GB fits one H100 80GB
        "batch_size": 8,
    },
    {
        "suffix": "qwen-32B",  # files: llm-32B_*.csv ; model col: llm-32B
        "subdir": "Qwen3-32B",
        "load_in_4bit": False,  # bf16 ~64GB fits one H100 80GB; clean scale point
        # batch 4 (not 8): run_llm_rerank forces the math SDP kernel (flash/
        # mem-efficient attention disabled), which materialises the full
        # batch*heads*seq*seq score matrix. On top of 64GB of bf16 weights that
        # pushes batch 8 toward the 80GB edge; batch 4 keeps a safe margin.
        "batch_size": 4,
    },
]


def _build_cfg(model_spec: dict, seed: int) -> dict:
    """Self-contained cfg for run_llm_rerank — only the sections it reads."""
    return {
        "seed": seed,
        # no "max_history" key -> reranker uses its default last-10 window,
        # matching the main-table Qwen3-8B run.
        "metrics": {"k_list": K_LIST},
        "rerank": {
            "llm_model_name": str(MODEL_DIR / model_spec["subdir"]),
            "llm_load_in_4bit": model_spec["load_in_4bit"],
            "llm_batch_size": model_spec["batch_size"],
            "llm_max_length": MAX_LENGTH,
            "llm_scoring_mode": SCORING_MODE,
            "llm_log_every": 10,
            "sample_users": 400,  # fallback only; rerank_eval_users.csv is used
        },
    }


# File-type suffixes run_llm_rerank emits, longest first (so a "*_subgroup_metrics.csv"
# is not mistaken for "*_metrics.csv").
_FILE_TYPES = ("_subgroup_metrics.csv", "_metrics.csv", "_scores.csv")


def _emit_into_base(work_dir: Path, base_dir: Path, suffix: str) -> None:
    """Move run_llm_rerank's outputs from the temp dir into the base run dir,
    inserting the per-model suffix and relabelling the model column.

    e.g. llm_scores.csv -> base_dir/llm-32B_scores.csv (model col -> 'llm-32B').
    Handles GALA variant files too if RUN_GALA is enabled (gala_metrics.csv ->
    gala-32B_metrics.csv).
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(work_dir.glob("*.csv")):
        ftype = next((t for t in _FILE_TYPES if f.name.endswith(t)), None)
        if ftype is None:
            shutil.move(str(f), str(base_dir / f"{f.stem}-{suffix}.csv"))
            continue
        model_part = f.name[: -len(ftype)]
        new_model = f"{model_part}-{suffix}"
        dst = base_dir / f"{new_model}{ftype}"
        if ftype == "_scores.csv":
            shutil.move(str(f), str(dst))
        else:
            df = pd.read_csv(f)
            if "model" in df.columns:
                df["model"] = new_model
            df.to_csv(dst, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Second-LLM generality runs (Qwen3-32B + Llama-3.3-70B)"
    )
    ap.add_argument(
        "--dataset", required=True, help="Dataset name (must already be split)"
    )
    ap.add_argument(
        "--data_dir", default="data", help="Root dir for data (raw/processed)"
    )
    ap.add_argument("--output_dir", default="outputs", help="Output root")
    ap.add_argument("--seed", type=int, default=42, help="Seed of the split to reuse")
    ap.add_argument("--top_k", type=int, default=200, help="Per-user reranking budget")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    processed_dir = data_dir / "processed" / args.dataset / f"s{args.seed}"
    rerank_users = processed_dir / "rerank_eval_users.csv"

    for f in (
        "train.csv",
        "test.csv",
        "candidates.csv",
        "items_mapped.csv",
        "item_stats.csv",
    ):
        if not (processed_dir / f).exists():
            raise FileNotFoundError(
                f"Split artifact {f} missing at {processed_dir}. Build it first:\n"
                f"  python scripts/run_dataset.py --dataset {args.dataset} "
                f"--data_dir {args.data_dir} --config configs/hpc.yaml "
                f"--top_k 500 --seed {args.seed} --split_only --force_split"
            )

    seed_everything(args.seed)
    gala_variants = None if RUN_GALA else []
    base_dir = (
        Path(args.output_dir).resolve() / f"{args.dataset}-top{args.top_k}-s{args.seed}"
    )

    for spec in MODELS:
        suffix = spec["suffix"]
        model_path = MODEL_DIR / spec["subdir"]
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. Download {spec['subdir']} "
                f"into {MODEL_DIR} (or set $COLDTAIL_MODEL_DIR)."
            )

        # Run into a temp subdir, then move outputs into the base dir with the
        # per-model suffix so the base llm_* (8B) files are never clobbered.
        work_dir = base_dir / f"_tmp_{suffix}"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        logger.info(
            "[second-llm] %s | model=%s | 4bit=%s | batch=%d | base=%s",
            suffix,
            model_path,
            spec["load_in_4bit"],
            spec["batch_size"],
            base_dir,
        )

        timing = TimingCollector()
        run_llm_rerank(
            processed_dir=processed_dir,
            out_dir=work_dir,
            cfg=_build_cfg(spec, args.seed),
            top_candidates=args.top_k,
            rerank_users_path=rerank_users if rerank_users.exists() else None,
            timing_collector=timing,
            run_baseline=True,
            gala_variants=gala_variants,
        )
        timing.save(work_dir / "timing_summary.csv")
        _emit_into_base(work_dir, base_dir, suffix)
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.info(
            "[second-llm] %s done | files=%s/llm-%s_*.csv", suffix, base_dir, suffix
        )

    logger.info("[second-llm] all models complete for %s", args.dataset)


if __name__ == "__main__":
    main()
