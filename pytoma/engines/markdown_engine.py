from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import List, Tuple

from markdown_it import MarkdownIt

from ..ir import Document, Edit, Node, MD_DOC, MD_HEADING
from ..markers import DEFAULT_OPTIONS, make_omission_line
from ..utils import line_starts, slugify
from ..ir import assign_ids, flatten


def _parse_headings_with_markdown_it(text: str) -> List[Tuple[int, int, str]]:
    md = MarkdownIt("commonmark")
    tokens = md.parse(text)

    headings: List[Tuple[int, int, str]] = []
    bq_depth = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "blockquote_open":
            bq_depth += 1
        elif tok.type == "blockquote_close":
            if bq_depth > 0:
                bq_depth -= 1
        elif tok.type == "heading_open" and bq_depth == 0:
            level = int(tok.tag[1])  # 'h2' -> 2
            start_line = tok.map[0] if tok.map else 0
            title = ""
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                title = tokens[i + 1].content.strip()
            headings.append((level, start_line, title))
        i += 1
    return headings


class MarkdownEngine:
    filetypes = {"md"}

    def configure(self, roots: List[Path]) -> None:
        return

    def parse(self, path: Path, text: str) -> Document:
        posix = PurePosixPath(path.as_posix())
        ls = line_starts(text)
        n_lines = len(ls) - 1

        root = Node(
            kind=MD_DOC,
            path=posix,
            span=(0, len(text)),
            name=str(posix),
            qual=str(posix),
            meta={},
            children=[],
        )

        heads_list = _parse_headings_with_markdown_it(text)

        # Compute section ends: next heading with level <=
        heads_enriched: List[Tuple[int, int, int, str]] = []
        for i, (lvl, start_line, title) in enumerate(heads_list):
            end_line = n_lines
            for j in range(i + 1, len(heads_list)):
                nxt_lvl, nxt_start_line, _ = heads_list[j]
                if nxt_lvl <= lvl:
                    end_line = nxt_start_line
                    break
            heads_enriched.append((lvl, start_line, end_line, title))

        nodes: List[Node] = []
        for lvl, start_line, end_line, title in heads_enriched:
            start_char = ls[start_line] if 0 <= start_line < len(ls) else 0
            end_char = ls[end_line] if 0 <= end_line < len(ls) else len(text)
            slug = slugify(title)
            qual = f"{posix}:{slug}"
            nodes.append(
                Node(
                    kind=MD_HEADING,
                    path=posix,
                    span=(start_char, end_char),
                    name=title,
                    qual=qual,
                    meta={"level": lvl, "slug": slug},
                    children=[],
                )
            )

        root.children.extend(nodes)
        assign_ids([root])
        flat = flatten([root])
        return Document(path=posix, text=text, roots=[root], nodes=flat)

    def supports(self, action) -> bool:
        return action.kind in {"hide", "full"}

    def render(self, doc: Document, decisions: List[tuple[Node, object]]) -> List[Edit]:
        candidates: List[Edit] = []

        def _line_range(span: tuple[int, int]) -> tuple[int, int]:
            s, t = span
            before = doc.text[:s]
            omitted = doc.text[s:t]
            start_line = before.count("\n") + 1
            end_line = start_line + omitted.count("\n")
            return start_line, end_line

        for node, action in decisions:
            if action.kind != "hide":
                continue

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
                    lang="md", a=a, b=b, indent="", opts=DEFAULT_OPTIONS, label=label
                )
                candidates.append(
                    Edit(path=doc.path, span=node.span, replacement=marker)
                )

        return candidates


def create_engine():
    return MarkdownEngine()
