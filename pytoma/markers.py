# pytoma/markers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Style = Literal["comment", "scissors", "ellipsis", "box"]

@dataclass(frozen=True)
class MarkerOptions:
    style: Style = "comment"
    show_counts: bool = True
    ascii_only: bool = False
    width: int = 88 # only used for "box"

DEFAULT_OPTIONS = MarkerOptions()

def _comment_wrap(lang: str, text: str, indent: str = "") -> str:
    """
    Return a single commented line adapted to the given 'lang'.
    - py/yaml/toml:    "# text"
    - md:              "<!-- text -->"
    Fallback:          "# text"
    """
    if lang in {"py", "yaml", "yml", "toml"}:
        return f"{indent}# {text}\n"
    if lang == "md":
        return f"{indent}<!-- {text} -->\n"
    return f"{indent}# {text}\n"

def _ellipsis(opts: MarkerOptions) -> str:
    return "..." if opts.ascii_only else "…"

def _scissors(opts: MarkerOptions) -> str:
    return "8<" if opts.ascii_only else "✂"

def _count_text(a: int, b: int, opts: MarkerOptions) -> str:
    n = max(0, b - a + 1)
    if not opts.show_counts:
        return ""
    # ex: " (42 lines)"
    unit = "line" if n == 1 else "lines"
    return f" ({n} {unit})"

def _box_line(width: int, ascii_only: bool) -> str:
    if ascii_only:
        return "-" * width
    return "─" * width

def _box_text(text: str, opts: MarkerOptions, indent: str) -> str:
    """
    Wrap the text inside a small single-line box.
    """
    pad = " "
    inner = f"{pad}{text}{pad}"
    line = _box_line(max(len(inner), min(opts.width, 120)), opts.ascii_only)
    top = indent + line + "\n"
    mid = indent + inner + "\n"
    bot = indent + line + "\n"
    return top + mid + bot

def make_omission_line(
    lang: str,
    a: int,
    b: int,
    *,
    indent: str = "",
    opts: MarkerOptions = DEFAULT_OPTIONS,
    label: str | None = None,
) -> str:
    """
    Build a single-line omission marker (or a light box for style 'box').
    - lang  : 'py', 'md', 'yaml', 'toml', ...
    - a, b  : inclusive line indices omitted, if known
    - indent: indentation to preserve
    - label : additional free text (e.g., "body truncated")
    """

    rng = f"{a}–{b}" if not opts.ascii_only else f"{a}-{b}"
    ell = _ellipsis(opts)
    sc = _scissors(opts)
    extra = _count_text(a, b, opts)

    base = label or "omitted"
    if opts.style == "ellipsis":
        text = f"{ell} lines {rng} {base}{extra} {ell}"
        return _comment_wrap(lang, text, indent)
    if opts.style == "scissors":
        text = f"{sc} lines {rng} {base}{extra} {sc}"
        return _comment_wrap(lang, text, indent)
    if opts.style == "box":
        # The box itself is not commented; we comment the middle line according to the language.
        # Light visual framing
        line = f"lines {rng} {base}{extra}"
        content = _comment_wrap(lang, line, indent)
        # Encadrement visuel léger
        return _box_text(line, opts, indent)
    # default: "comment"
    text = f"{ell} lines {rng} {base}{extra} {ell}"
    return _comment_wrap(lang, text, indent)

