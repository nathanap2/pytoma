from pathlib import Path, PurePosixPath
from typing import List, Tuple, Dict, Optional
import re
import unicodedata

from ..ir import Document, Node, Edit, MD_DOC, MD_HEADING, assign_ids, flatten
from ..policies import Action

from ..markers import DEFAULT_OPTIONS, make_omission_line


# -- Optional dependency: markdown-it-py (CommonMark). Regex fallback if missing.
try:
    from markdown_it import MarkdownIt  # type: ignore
except Exception:  # pragma: no cover
    MarkdownIt = None  # type: ignore

# Old regex (fallback only). Problem: it also matches # inside code fences.
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)


def _line_starts(text: str) -> List[int]:
    """
    Return the character offsets of the start of each line, plus a final sentinel.
    Enables O(1) conversion from line_index -> char_offset.
    """
    starts = [0]
    acc = 0
    for line in text.splitlines(keepends=True):
        acc += len(line)
        starts.append(acc)
    return starts


def _slugify(title: str) -> str:
    """
    Deterministic, accent-insensitive slug.
    """
    s = unicodedata.normalize("NFKD", title)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "section"


def _parse_headings_with_markdown_it(text: str) -> List[Tuple[int, int, str]]:
    """
    Parse headings via markdown-it-py while ignoring those nested in blockquotes.
    Returns [(level, start_line, title)] for top-level headings (outside of > ...).
    """
    assert MarkdownIt is not None
    md = MarkdownIt("commonmark")
    tokens = md.parse(text)

    headings: List[Tuple[int, int, str]] = []
    bq_depth = 0  # blockquote nesting depth

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # Track blockquote nesting
        if tok.type == "blockquote_open":
            bq_depth += 1
        elif tok.type == "blockquote_close":
            if bq_depth > 0:
                bq_depth -= 1

        # Only collect headings when not inside a blockquote
        elif tok.type == "heading_open" and bq_depth == 0:
            level = int(tok.tag[1])  # 'h2' -> 2
            start_line = tok.map[0] if tok.map else 0
            title = ""
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                inline = tokens[i + 1]
                title = inline.content.strip()
            headings.append((level, start_line, title))

        i += 1

    return headings


def _parse_headings_with_regex(text: str) -> List[Tuple[int, int, str]]:
    """
    Minimal fallback (old behavior) based on regex.
    Can mistake '#' inside code for headings.
    """
    ls = _line_starts(text)
    out: List[Tuple[int, int, str]] = []
    for m in _HEADING_RE.finditer(text):
        hashes = m.group(1)
        title = m.group(2).strip()
        level = len(hashes)
        # convert char_start -> line index
        char_start = m.start()
        line_idx = 0
        for i in range(len(ls) - 1):
            if ls[i] <= char_start < ls[i + 1]:
                line_idx = i  # 0-based here
                break
        out.append((level, line_idx, title))
    return out


class MarkdownMinEngine:
    filetypes = {"md"}

    def configure(self, roots: List[Path]) -> None:  # no-op (signature for consistency)
        return

    def parse(self, path: Path, text: str) -> Document:
        """
        Build a flat tree (root -> sections) where each section spans its heading
        and all its content up to the next heading of level <= (or the end of the doc).
        Headings detected via markdown-it-py (robust against code fences).
        """
        posix = PurePosixPath(path.as_posix())
        ls = _line_starts(text)
        n_lines = len(ls) - 1  # number of lines

        # Document root
        root = Node(
            kind=MD_DOC,
            path=posix,
            span=(0, len(text)),
            name=str(posix),
            qual=str(posix),  # allows targeting the entire doc via a match on the path
            meta={},
            children=[],
        )

        # 1) Collect headings (level, start_line, title)
        if MarkdownIt is not None:
            heads_list = _parse_headings_with_markdown_it(text)
        else:
            heads_list = _parse_headings_with_regex(text)

        # 2) Compute section ends: up to the next heading of level <=
        # Work with line indices, then convert to character offsets.
        # heads_enriched: (level, start_line, end_line, title)
        heads_enriched: List[Tuple[int, int, int, str]] = []
        for i, (lvl, start_line, title) in enumerate(heads_list):
            end_line = n_lines
            for j in range(i + 1, len(heads_list)):
                nxt_lvl, nxt_start_line, _ = heads_list[j]
                if nxt_lvl <= lvl:
                    end_line = nxt_start_line
                    break
            heads_enriched.append((lvl, start_line, end_line, title))

        # 3) Create section nodes
        nodes: List[Node] = []
        for lvl, start_line, end_line, title in heads_enriched:
            # Character offsets: take the start of the heading's line,
            # and stop at the start of the next heading's line (or end of text).
            start_char = ls[start_line] if 0 <= start_line < len(ls) else 0
            end_char = ls[end_line] if 0 <= end_line < len(ls) else len(text)

            slug = _slugify(title)
            qual = f"{posix}:{slug}"
            node = Node(
                kind=MD_HEADING,
                path=posix,
                span=(start_char, end_char),
                name=title,
                qual=qual,
                meta={"level": lvl, "slug": slug},
                children=[],
            )
            nodes.append(node)

        # 4) Flat tree (sections as direct children of the root)
        root.children.extend(nodes)
        assign_ids([root])
        flat = flatten([root])

        return Document(path=posix, text=text, roots=[root], nodes=flat)

    def supports(self, action: Action) -> bool:
        return action.kind in {"hide", "full"}

    def render(self, doc: Document, decisions: List[Tuple[Node, Action]]) -> List[Edit]:
        candidates: List[Edit] = []
        for node, action in decisions:
            if action.kind != "hide":
                continue

            def _line_range(span: Tuple[int, int]) -> Tuple[int, int]:
                s, t = span
                before = doc.text[:s]
                omitted = doc.text[s:t]
                start_line = before.count("\n") + 1
                end_line = start_line + omitted.count("\n")
                return start_line, end_line

            if node.kind == MD_DOC:
                n_lines = doc.text.count("\n") + 1
                marker = make_omission_line(
                    lang="md",
                    a=1,
                    b=n_lines,
                    indent="",
                    opts=DEFAULT_OPTIONS,
                    label="document omitted",
                )
                candidates.append(
                    Edit(path=doc.path, span=(0, len(doc.text)), replacement=marker)
                )

            elif node.kind == MD_HEADING:
                a, b = _line_range(node.span)
                label = (
                    f"section {node.name!r} omitted" if node.name else "section omitted"
                )
                marker = make_omission_line(
                    lang="md",
                    a=a,
                    b=b,
                    indent="",
                    opts=DEFAULT_OPTIONS,
                    label=label,
                )
                candidates.append(
                    Edit(path=doc.path, span=node.span, replacement=marker)
                )

        return candidates

def create_engine():
    return MarkdownMinEngine()
