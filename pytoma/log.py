from __future__ import annotations
import os, sys


def enabled() -> bool:
    """Return True if debug logging is enabled (via PYTOMA_DEBUG)."""
    return bool(os.environ.get("PYTOMA_DEBUG"))


def debug(*parts: object, tag: str = "core") -> None:
    """
    Lightweight debug logger.
    Usage: debug("message", tag="cli")
    """
    if enabled():
        sys.stderr.write(f"[pytoma:{tag}] " + " ".join(map(str, parts)) + "\n")
