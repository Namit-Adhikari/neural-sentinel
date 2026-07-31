"""GPU/TPU detection and fallback utility for Neural Sentinel.

Every notebook and agent should call ``get_device()`` rather than
hard-coding ``torch.device("cuda")`` or ``"cpu"``.  This ensures Kaggle
free-tier CPU kernels never fail due to missing accelerators.

Usage::

    from src.utils.device_utils import get_device
    device = get_device()
    model = model.to(device)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_device() -> "torch.device":  # type: ignore[name-defined]  # noqa: F821
    """Return the best available PyTorch device with a CPU fallback.

    Priority order:
    1. CUDA GPU (if ``torch.cuda.is_available()``)
    2. Apple MPS (if ``torch.backends.mps.is_available()``)
    3. CPU (always available)

    Returns:
        A ``torch.device`` object pointing at the best available device.

    Raises:
        ImportError: If PyTorch is not installed in the current environment.

    Example::

        device = get_device()   # torch.device('cuda:0') or 'cpu'
        tensor = torch.zeros(3).to(device)
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for device detection. "
            "Install it with: pip install torch==2.5.1"
        ) from exc

    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        logger.info(
            "GPU detected: %s (%.1f GB VRAM)",
            props.name,
            props.total_memory / 1024 ** 3,
        )
        return device

    # Apple Silicon MPS backend
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Apple MPS device detected.")
        return device

    # CPU fallback — always available
    device = torch.device("cpu")
    logger.info("No GPU detected — running on CPU.")
    return device


def log_device_info() -> str:
    """Log and return a human-readable summary of the available compute device.

    Returns:
        String summary of the device (e.g., ``"CUDA GPU: Tesla T4 (15.8 GB)"``).
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "PyTorch not installed — device info unavailable."

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        summary = f"CUDA GPU: {props.name} ({props.total_memory / 1024 ** 3:.1f} GB)"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        summary = "Apple MPS (unified memory)"
    else:
        summary = "CPU only"

    logger.info("Compute device: %s", summary)
    return summary
