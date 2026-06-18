"""Detect hardware and derive safe training defaults."""
from __future__ import annotations

import os


def detect_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


def default_n_envs() -> int:
    """CPU count minus 2 for the main process and OS headroom."""
    return max(1, (os.cpu_count() or 4) - 2)


def default_batch_size(device: str) -> int:
    if device not in ("cuda", "mps"):
        return 512
    if device == "mps":
        return 512
    try:
        import torch
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb >= 16:
            return 4096
        if vram_gb >= 8:
            return 2048
        return 1024
    except Exception:
        return 512


def gpu_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return f"{props.name} ({props.total_memory / 1e9:.1f} GB VRAM)"
        if torch.backends.mps.is_available():
            return "Apple MPS"
        return "no GPU"
    except Exception:
        return "unknown"


def summary() -> str:
    device = detect_device()
    n = default_n_envs()
    bs = default_batch_size(device)
    gpu = gpu_info() if device in ("cuda", "mps") else "none"
    return f"device={device}  n_envs={n}  batch_size={bs}  gpu={gpu}"
