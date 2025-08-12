# pytoma/edits.py
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from .ir import Edit


def merge_edits(edits: Iterable[Edit]) -> List[Edit]:
    """
    Policy: outermost wins for nested overlaps; partial overlap -> ValueError.
    Merges independently per file and returns a globally ordered list.
    """
    by_path: Dict[Path, List[Edit]] = defaultdict(list)
    for e in edits:
        by_path[Path(e.path)].append(e)

    merged_all: List[Edit] = []
    for path in sorted(by_path.keys(), key=lambda p: p.as_posix()):
        group = by_path[path]
        ordered = sorted(group, key=lambda e: (e.span[0], -e.span[1]))
        kept: List[Edit] = []
        for e in ordered:
            s, t = e.span
            if kept:
                ks, kt = kept[-1].span
                if s < kt:
                    if t <= kt:
                        continue  # fully contained -> drop current
                    raise ValueError(
                        f"Overlapping edits on {path.as_posix()}: {(ks, kt)} vs {(s, t)}"
                    )
            kept.append(e)
        merged_all.extend(kept)
    return merged_all


def _apply_edits_to_text(text: str, edits: List[Edit]) -> str:
    ordered = sorted(edits, key=lambda e: e.span[0], reverse=True)
    out = text
    last_end = len(text)
    for e in ordered:
        s, t = e.span
        if not (0 <= s <= t <= last_end):
            raise ValueError(f"Invalid or overlapping span: {e.span}")
        out = out[:s] + e.replacement + out[t:]
        last_end = s
    return out


def apply_edits_preview(edits: Iterable[Edit]) -> Dict[Path, str]:
    from collections import defaultdict

    previews: Dict[Path, str] = {}
    by_path: Dict[Path, List[Edit]] = defaultdict(list)
    for e in merge_edits(list(edits)):
        by_path[Path(e.path)].append(e)

    for path, group in by_path.items():
        text = path.read_text(encoding="utf-8")
        previews[path] = _apply_edits_to_text(text, group)
    return previews
