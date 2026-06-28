"""Per-model timing collector for latency, throughput, and GPU memory."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class TimingCollector:
    def __init__(self):
        self.records: list[dict] = []

    def record(
        self,
        model: str,
        num_pairs: int,
        elapsed_seconds: float,
        peak_gpu_memory_mb: float | None = None,
    ) -> None:
        throughput = num_pairs / elapsed_seconds if elapsed_seconds > 0 else 0.0
        rec = {
            "model": model,
            "num_pairs": num_pairs,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "throughput_pairs_per_sec": round(throughput, 1),
            "peak_gpu_memory_mb": round(peak_gpu_memory_mb, 1)
            if peak_gpu_memory_mb is not None
            else None,
        }
        self.records.append(rec)
        logger.info(
            f"[Timing] {model} | pairs={num_pairs:,} | time={elapsed_seconds:.1f}s | "
            f"throughput={throughput:,.0f}pairs/s"
            + (f" | gpu_peak={peak_gpu_memory_mb:.0f}MB" if peak_gpu_memory_mb else "")
        )

    def save(self, path: str | Path) -> None:
        if not self.records:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.records).to_csv(path, index=False)
        logger.info(f"[Timing] saved {len(self.records)} records to {path}")
