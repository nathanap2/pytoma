import fnmatch
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Dict, List, Tuple
from collections import defaultdict

from .ir import Edit

def iter_files(
    roots: Iterable[Path],
    includes: Iterable[str] = ("**/*",),
    excludes: Iterable[str] = (),
) -> Iterator[Path]:
    """
    File discovery with glob/fnmatch patterns (language-agnostic).
    - 'includes' apply relative to each root.
    - 'excludes' are fnmatch-style patterns on the POSIX *relative* path.
    """
    roots = list(roots)
    for root in roots:
        base = root.resolve()
        for inc in includes:
            for p in base.glob(inc):
                if not p.is_file():
                    continue
                rel = PurePosixPath(p.relative_to(base).as_posix())
                if any(fnmatch.fnmatch(rel, pat) for pat in excludes):
                    continue
                yield p

def merge_edits(edits: Iterable[Edit]) -> List[Edit]:
    """
    Deduplicate and resolve overlaps with a single policy:
      - For nested overlaps, the *outermost* edit wins (children are dropped).
      - For partial overlaps (intersecting but neither contains the other),
        raise ValueError (engines must avoid such conflicts).
    Returns a list of non-overlapping edits (across all files).
    """
    by_path: Dict[Path, List[Edit]] = defaultdict(list)
    for e in edits:
        by_path[Path(e.path)].append(e)

    merged_all: List[Edit] = []
    # deterministic order by path then span
    for path in sorted(by_path.keys(), key=lambda p: p.as_posix()):
        group = by_path[path]
        # Sort: start asc, end desc  => outer segments first for a given start
        ordered = sorted(group, key=lambda e: (e.span[0], -e.span[1]))
        kept: List[Edit] = []
        for e in ordered:
            s, t = e.span
            if kept:
                ks, kt = kept[-1].span
                if s < kt:
                    # overlap with the last kept segment
                    if t <= kt:
                        # fully contained -> drop current (outer wins)
                        continue
                    # partial overlap -> conflict
                    raise ValueError(
                        f"Overlapping edits on {path.as_posix()}: "
                        f"{(ks, kt)} vs {(s, t)}"
                    )
            kept.append(e)
        merged_all.extend(kept)
    return merged_all

def _apply_edits_to_text(text: str, edits: List[Edit]) -> str:
    """
    Apply non-overlapping edits to 'text'.
    Edits MUST target the same file and MUST NOT overlap.
    """
    # sort descending so later offsets remain valid
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


def apply_edits_in_place(edits: Iterable[Edit]) -> None:
    """
    Group per file, read/write as UTF-8. Raise if an overlap is detected.
    """
    # Normalize + merge per file first
    merged = merge_edits(list(edits))
    by_path: Dict[Path, List[Edit]] = defaultdict(list)
    for e in merged:
        by_path[Path(e.path)].append(e)

    for path, group in by_path.items():
        text = path.read_text(encoding="utf-8")
        new = _apply_edits_to_text(text, group)
        path.write_text(new, encoding="utf-8")

def apply_edits_preview(edits: Iterable[Edit]) -> Dict[Path, str]:
    """
    Return a dict {path -> modified_text} without writing to disk.
    Useful to assemble a 'prompt pack' without touching sources.
    """
    merged = merge_edits(list(edits))
    by_path: Dict[Path, List[Edit]] = defaultdict(list)
    for e in merged:
        by_path[Path(e.path)].append(e)

    previews: Dict[Path, str] = {}
    for path, group in by_path.items():
        text = path.read_text(encoding="utf-8")
        previews[path] = _apply_edits_to_text(text, group)
    return previews
