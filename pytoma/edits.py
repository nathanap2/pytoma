# pytoma/edits.py
from collections import defaultdict
from pathlib import Path

from .ir import Edit


from typing import Dict, Iterable, List, Tuple

from .ir import Edit

from .utils import read_text_any


def merge_edits(edits: Iterable[Edit]) -> List[Edit]:
    """
    Merge policy:
      - Per file, merge deletions/modifications (spans with t > s) using the
        "outermost wins" rule (an edit fully contained by another is dropped;
        partial overlap -> error).
      - Insertions (spans with t == s) are processed afterwards: if an insertion
        falls inside a kept deletion span, move it to the right edge of that deletion
        (i.e., to the 'end' position).
      - Returns a list globally ordered by path, then by start offset.
    """
    by_path: Dict[Path, List[Edit]] = defaultdict(list)
    for e in edits:
        by_path[Path(e.path)].append(e)

    merged_all: List[Edit] = []
    for path in sorted(by_path.keys(), key=lambda p: p.as_posix()):
        group = by_path[path]

        # 1) Separate deletions/updates from insertions
        deletes: List[Edit] = []
        inserts: List[Edit] = []
        for e in group:
            s, t = e.span
            if t > s:
                deletes.append(e)
            elif t == s:
                inserts.append(e)
            else:
                raise ValueError(
                    f"Invalid span (end < start) on {path.as_posix()}: {e.span}"
                )

        # 2) Merge deletions/updates: outermost wins
        ordered_del = sorted(deletes, key=lambda e: (e.span[0], -e.span[1]))
        kept: List[Edit] = []
        for e in ordered_del:
            s, t = e.span
            if kept:
                ks, kt = kept[-1].span
                if s < kt:
                    if t <= kt:
                        # fully contained: drop e
                        continue
                    # partial overlap: invalid
                    raise ValueError(
                        f"Overlapping edits on {path.as_posix()}: {(ks, kt)} vs {(s, t)}"
                    )
            kept.append(e)

        # 3) Normalize insertions: push to the right edge if inside a deletion
        norm_inserts: List[Edit] = []
        for ins in sorted(inserts, key=lambda e: e.span[0]):
            p = ins.span[0]
            for d in kept:
                s, t = d.span
                if s <= p <= t:
                    p = t
            if p != ins.span[0]:
                norm_inserts.append(
                    Edit(
                        path=ins.path,
                        span=(p, p),
                        replacement=ins.replacement,
                        comment=ins.comment,
                    )
                )
            else:
                norm_inserts.append(ins)

        # 4) Concatenate (deterministic order within the file)
        merged_all.extend(kept)
        merged_all.extend(norm_inserts)

    # 5) Deterministic global order (by file, then by offset)
    merged_all.sort(key=lambda e: (Path(e.path).as_posix(), e.span[0], e.span[1]))
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
    """
    Apply edits to files, producing a preview string for each path.
    Reads the file content with tolerant encoding detection instead of hardcoded UTF-8.
    """
    from collections import defaultdict

    previews: Dict[Path, str] = {}
    by_path: Dict[Path, List[Edit]] = defaultdict(list)
    for e in merge_edits(list(edits)):
        by_path[Path(e.path)].append(e)

    for path, group in by_path.items():
        text = read_text_any(path)  # tolerant reader
        previews[path] = _apply_edits_to_text(text, group)
    return previews
