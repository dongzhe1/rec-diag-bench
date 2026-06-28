"""GPU/CPU resource monitoring utilities for experiment logging."""

import logging

import psutil

logger = logging.getLogger(__name__)

_nvml_handles: list = []


def _init_nvml() -> bool:
    global _nvml_handles
    try:
        import pynvml

        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()
        _nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(num_gpus)]
        logger.info(f"[GPU] pynvml initialized | num_gpus={num_gpus}")
        return True
    except Exception as e:
        logger.warning(f"[GPU] pynvml unavailable | error={e}")
        return False


def _log_gpu_info() -> None:
    import torch

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.warning("[GPU] no CUDA devices detected | running on CPU")
        return

    for idx in range(num_gpus):
        props = torch.cuda.get_device_properties(idx)
        total_vram_gb = props.total_memory / 1024**3
        logger.info(
            f"[GPU {idx}] {props.name} | "
            f"vram_total={total_vram_gb:.1f}GB | "
            f"cuda_capability={props.major}.{props.minor}"
        )


def _gpu_stats(device_idx: int = 0) -> dict | None:
    import torch

    if not torch.cuda.is_available() or device_idx >= torch.cuda.device_count():
        return None

    allocated = torch.cuda.memory_allocated(device_idx) / 1024**3
    peak = torch.cuda.max_memory_allocated(device_idx) / 1024**3
    total = torch.cuda.get_device_properties(device_idx).total_memory / 1024**3

    gpu_util = -1
    if _nvml_handles:
        try:
            import pynvml

            rates = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handles[device_idx])
            gpu_util = rates.gpu
        except Exception:
            pass

    return dict(
        vram_allocated_gb=allocated,
        vram_peak_gb=peak,
        vram_total_gb=total,
        vram_used_pct=allocated / total * 100,
        gpu_util_pct=gpu_util,
    )


def _log_gpu_stats(label: str, device_idx: int = 0) -> None:
    stats = _gpu_stats(device_idx)
    if stats is None:
        logger.debug(f"[GPU] {label} | cuda unavailable")
        return

    util_str = (
        f"{stats['gpu_util_pct']:>3.0f}%" if stats["gpu_util_pct"] >= 0 else "n/a"
    )
    logger.info(
        f"[GPU] {label} | "
        f"vram_used={stats['vram_allocated_gb']:.2f}GB | "
        f"vram_peak={stats['vram_peak_gb']:.2f}GB | "
        f"vram_total={stats['vram_total_gb']:.1f}GB | "
        f"vram_util={stats['vram_used_pct']:.1f}% | "
        f"gpu_util={util_str}"
    )


def _log_cpu_stats(label: str = "") -> None:
    cpu_pct = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    ram_used_gb = vm.used / 1024**3
    ram_total_gb = vm.total / 1024**3

    suffix = f" | {label}" if label else ""
    logger.info(
        f"[CPU]{suffix} | "
        f"cpu_util={cpu_pct:.1f}% | "
        f"ram_used={ram_used_gb:.2f}GB/{ram_total_gb:.1f}GB | "
        f"ram_util={vm.percent:.1f}%"
    )
