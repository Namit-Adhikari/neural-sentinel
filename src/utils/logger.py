"""Structured logging setup for the Neural Sentinel project.

Provides :func:`setup_logger` and a module-level :data:`get_logger` helper so
all library code can request a named, consistently-formatted logger without
duplicating the handler and formatter configuration.  The formatter emits plain
``LEVEL | name | message`` lines — no ANSI colours, no centering, safe for
small terminals.

Usage::

    from src.utils.logger import get_logger
    logger = get_logger("neural_sentinel.cleaning")
    logger.info("Loaded %d rows", 100_000)

The package root logger is wired on first import via :func:`_ensure_root_setup`
so that ``logging.getLogger("neural_sentinel")`` always has a handler attached
with a reasonable default level.  Callers who want DEBUG output should call
:func:`set_level` or configure logging *before* importing library modules.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_DEFAULT_LEVEL: int = logging.INFO
_FORMAT: str = "%(levelname)s | %(name)s | %(message)s"
_DATEFMT: str = "%Y-%m-%d %H:%M:%S"

_root_configured: bool = False


class _PlainFormatter(logging.Formatter):
    """Formatter that deliberately avoids ANSI colour codes."""

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record)


def _ensure_root_setup() -> None:
    """Attach a single StreamHandler to the ``neural_sentinel`` package logger.

    Idempotent: repeated calls are no-ops.  The handler uses ``sys.stderr`` so
    Kaggle and notebook environments preserve logging output separately from
    ``print()`` calls in user code.
    """

    global _root_configured
    if _root_configured:
        return

    root = logging.getLogger("neural_sentinel")
    root.setLevel(_DEFAULT_LEVEL)
    root.propagate = False

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(_DEFAULT_LEVEL)
    handler.setFormatter(_PlainFormatter(_FORMAT, datefmt=_DATEFMT))

    root.handlers.clear()
    root.addHandler(handler)
    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger under the ``neural_sentinel`` namespace.

    Args:
        name: Dotted logger name.  A leading ``"neural_sentinel."`` prefix is
            prepended automatically if missing, so callers can pass
            ``"cleaning"`` instead of ``"neural_sentinel.cleaning"``.

    Returns:
        A :class:`logging.Logger` instance guaranteed to have at least one
        handler on its ancestor chain.
    """

    _ensure_root_setup()
    if not name.startswith("neural_sentinel"):
        name = f"neural_sentinel.{name}"
    return logging.getLogger(name)


def set_level(level: int | str, logger_name: Optional[str] = None) -> None:
    """Adjust the log level for the package root (or a named child).

    Args:
        level: Integer level (``logging.INFO``) or string (``"DEBUG"``).
        logger_name: Optional child logger name.  When ``None`` the package
            root *and* its attached handler are both updated.
    """

    _ensure_root_setup()
    target = logging.getLogger(
        f"neural_sentinel.{logger_name}" if logger_name else "neural_sentinel"
    )
    target.setLevel(level)
    if logger_name is None:
        for handler in target.handlers:
            handler.setLevel(level)


def setup_logger(
    name: str,
    level: int | str = _DEFAULT_LEVEL,
) -> logging.Logger:
    """Alias for :func:`get_logger` with an explicit level override.

    Kept as a thin wrapper so notebooks and scripts can read like::

        logger = setup_logger("phase2_cleaning", level="DEBUG")
    """

    log = get_logger(name)
    log.setLevel(level)
    return log
