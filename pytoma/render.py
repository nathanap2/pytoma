from typing import Callable, List, Optional, Tuple
import re

from .collect import FuncInfo
from .markers import make_omission_line, DEFAULT_OPTIONS

RenderFn = Callable[[FuncInfo, List[str]], str]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_trailing_nl(s: str) -> str:
    return s if s.endswith("\n") else s + "\n"

def _line_indent(s: str) -> str:
    m = re.match(r"[ \t]*", s)
    return m.group(0) if m else ""

def header_exact(func: FuncInfo, src: List[str]) -> str:
    """
    Exact header of a function (without the body): from the first line of the
    decorator (or the 'def') up to the line preceding the first element of the body.
    """
    end_line = (func.body_first_line - 1) if func.body_first_line else func.end[0]
    text = "".join(src[func.deco_start_line - 1 : end_line])
    return _ensure_trailing_nl(text)

def header_one_line(func: FuncInfo, src: List[str]) -> str:
    """
    Compact version of the header (everything on one line), useful for 'sig'.
    Preserves the initial indentation.
    """
    h = header_exact(func, src)
    lines = h.splitlines()
    if not lines:
        return h
    indent = _line_indent(lines[0])
    # Concatenate the lines into a single one, removing indentations
    content = lines[0].rstrip()
    for ln in lines[1:]:
        content += " " + ln.strip()
    # Make sure it ends with ':'
    if not content.rstrip().endswith(":"):
        content = content.rstrip() + ":"
    return _ensure_trailing_nl(content)

def signature_compact(node) -> str:
    """
    Historical compatibility: some older renderers relied on a synthetic
    "compact signature". Here we rely on header_one_line which faithfully
    reflects the source signature.
    """
    # This function is kept for backward compatibility; new code paths
    # use header_one_line(fi, src).
    # We produce a minimal 'def name(...):' when no source is available.
    name = getattr(getattr(node, "name", None), "value", None) or "function"
    return f"def {name}(...):\n"

def compute_body_range(func: FuncInfo) -> Optional[Tuple[int, int]]:
    """
    Return inclusive (start_line, end_line) for the function body.
    """
    if not func.body_first_line:
        return None
    return (func.body_first_line, func.end[0])

def indent_level(line: str) -> int:
    """
    Compute the indentation level (columns) assuming tab=4 spaces.
    """
    level = 0
    for ch in line:
        if ch == " ":
            level += 1
        elif ch == "\t":
            level += 4
        else:
            break
    return level

def slice_with_levels(src: List[str], start: int, end: int, keep_levels: int) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Keep lines whose indentation (column) does not exceed
    base_indent + keep_levels*4; group the omitted spans.

    Returns (kept_lines, omitted_ranges[(a,b)]) with a/b as 1-based indices.
    """
    kept: List[str] = []
    omitted: List[Tuple[int, int]] = []

    # determine the base indent from the first non-empty line
    base = None
    for ln in range(start, end + 1):
        raw = src[ln - 1]
        if raw.strip():
            base = indent_level(raw)
            break
    if base is None:
        return kept, omitted  # empty body / blank lines

    omitting = False
    ostart = None

    def flush(to_line: int) -> None:
        nonlocal omitting, ostart
        if omitting and ostart is not None:
            omitted.append((ostart, to_line - 1))
            omitting, ostart = False, None

    for ln in range(start, end + 1):
        raw = src[ln - 1]
        if not raw.strip():
            # always keep blank lines for readability
            if not omitting:
                kept.append(raw)
            continue
        lvl = indent_level(raw)
        if (lvl - base) <= keep_levels * 4:
            flush(ln)
            kept.append(raw)
        else:
            if not omitting:
                omitting, ostart = True, ln
    flush(end + 1)
    return kept, omitted

# ---------------------------------------------------------------------------
# Destructive editing API (we keep everything then remove)
# ---------------------------------------------------------------------------

def _sig_block(fi: FuncInfo, src: List[str]) -> str:
    # One-line signature + omission marker (instead of "...")
    h = header_one_line(fi, src)
    indent = _line_indent(h)
    rng = compute_body_range(fi)
    if rng:
        a, b = rng
        marker = make_omission_line(
            lang="py",
            a=a,
            b=b,
            indent=indent + "    ",
            opts=DEFAULT_OPTIONS,
            label="body omitted",
        )
        return h + marker
    # Fallback if body cannot be computed
    return h + f"{indent}    ...\n"

def _sigdoc_block(fi: FuncInfo, src: List[str]) -> str:
    h = header_one_line(fi, src)
    indent = _line_indent(h)
    parts = [h]
    if fi.docstring:
        parts.append(_ensure_trailing_nl(fi.docstring))
    else:
        parts.append(f'{indent}    """…"""\n')
    rng = compute_body_range(fi)
    if rng:
        a, b = rng
        parts.append(
            make_omission_line(
                lang="py",
                a=a,
                b=b,
                indent=indent + "    ",
                opts=DEFAULT_OPTIONS,
                label="body omitted",
            )
        )
    else:
        parts.append(f"{indent}    ...\n")
    return "".join(parts)

def _levels_block(fi: FuncInfo, src: List[str], keep_levels: int) -> str:
    parts: List[str] = [header_exact(fi, src)]
    rng = compute_body_range(fi)
    if not rng:
        return "".join(parts)
    kept, omitted = slice_with_levels(src, rng[0], rng[1], keep_levels=keep_levels)
    # docstring if omitted
    if fi.docstring and fi.doc_range and any(a <= fi.doc_range[0] <= b for (a, b) in omitted):
        parts.append(_ensure_trailing_nl(fi.docstring))
    parts.extend(kept)
    base_indent = _line_indent(src[rng[0] - 1]) + " " * 4
    for (a, b) in omitted:
        parts.append(
            make_omission_line(
                lang="py",
                a=a,
                b=b,
                indent=base_indent,
                opts=DEFAULT_OPTIONS,
                label="omitted",
            )
        )
    return "".join(parts)

def _replacement_for_mode(mode: str, fi: FuncInfo, src: List[str]) -> Optional[Tuple[int, int, str]]:
    """
    Compute (start_line, end_line, replacement_text) for a given function.
    start/end are inclusive (1-based).
    """
    if mode == "full":
        return None  # leave untouched
    if mode == "hide":
        # Remove the function entirely from the output
        return (fi.deco_start_line, fi.end[0], "")
    if mode == "sig":
        block = _sig_block(fi, src)
    elif mode == "sig+doc":
        block = _sigdoc_block(fi, src)
    else:
        m = re.match(r"body:levels=(\d+)", mode)
        if not m:
            return None
        block = _levels_block(fi, src, int(m.group(1)))
    return (fi.deco_start_line, fi.end[0], block)

def apply_destructive(src: List[str], funcs: List[FuncInfo], choose_mode_fn) -> str:
    """
    Apply *in-place* replacements to the full source text.

    - src: list of lines (keepends=True)
    - funcs: functions indexed in the file
    - choose_mode_fn: (qualname:str) -> mode:str
    """
    edits: List[Tuple[int, int, str]] = []
    for fi in funcs:
        mode = choose_mode_fn(fi.qualname)
        rep = _replacement_for_mode(mode, fi, src)
        if rep:
            edits.append(rep)
    if not edits:
        return "".join(src)
    # apply from bottom to top to preserve offsets
    edits.sort(key=lambda e: e[0], reverse=True)
    out = src[:]  # copy
    for a, b, block in edits:
        out[a - 1 : b] = [block]
    # 'out' is a list of lines and/or a single large block; normalize
    if isinstance(out, list):
        return "".join(out)
    return out

