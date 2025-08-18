from __future__ import annotations
import re, unicodedata
from pathlib import Path, PurePosixPath
from typing import List


def line_starts(text: str) -> List[int]:
    """Return character offsets of the start of each line, with a final sentinel."""
    starts = [0]
    acc = 0
    for line in text.splitlines(keepends=True):
        acc += len(line)
        starts.append(acc)
    return starts


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    """Deterministic, accent-insensitive slug."""
    s = unicodedata.normalize("NFKD", title)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = _SLUG_RE.sub("-", s).strip("-")
    return s or "section"


def posix(p: Path | PurePosixPath) -> str:
    return PurePosixPath(str(p)).as_posix()
