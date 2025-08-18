from pathlib import Path, PurePosixPath
from typing import List, Tuple
import re

from ..ir import Node, Document, assign_ids, flatten, Edit, TOML_DOC, TOML_TABLE
from ..markers import make_omission_line, DEFAULT_OPTIONS


_TBL_RE = re.compile(r"(?m)^[ \t]*\[(?P<name>[^\[\]\n]+)\][ \t]*(?:#.*)?$")
_TBLARR_RE = re.compile(r"(?m)^[ \t]*\[\[(?P<name>[^\[\]\n]+)\]\][ \t]*(?:#.*)?$")
_COMMENT_RE = re.compile(r"^[ \t]*#")
_WHITESPACE = re.compile(r"^[ \t]*$")


def _line_starts(text: str) -> List[int]:
    starts = [0]
    acc = 0
    for line in text.splitlines(keepends=True):
        acc += len(line)
        starts.append(acc)
    return starts


def _has_substantive_text(block: str) -> bool:
    for ln in block.splitlines():
        if _WHITESPACE.match(ln):
            continue
        if _COMMENT_RE.match(ln):
            continue
        return True
    return False


class TomlEngine:

    filetypes = {"toml"}

    def configure(self, roots: List[Path]) -> None:
        return

    def parse(self, path: Path, text: str) -> Document:
        posix = PurePosixPath(path.as_posix())
        ls = _line_starts(text)

        # Racine (document)
        root = Node(
            kind=TOML_DOC,
            path=posix,
            span=(0, len(text)),
            name=str(posix),
            qual=str(posix),
            meta={},
            children=[],
        )

        hits: List[Tuple[int, str, bool]] = []  # (char_start, name, is_array)
        for m in _TBL_RE.finditer(text):
            hits.append((m.start(), m.group("name").strip(), False))
        for m in _TBLARR_RE.finditer(text):
            hits.append((m.start(), m.group("name").strip(), True))
        hits.sort(key=lambda t: t[0])

        nodes: List[Node] = []

        first_start = hits[0][0] if hits else len(text)
        prefix = text[:first_start]
        if _has_substantive_text(prefix):
            nodes.append(
                Node(
                    kind=TOML_TABLE,
                    path=posix,
                    span=(0, first_start),
                    name="(root)",
                    qual=f"{posix}:(root)",
                    meta={"type": "root"},
                    children=[],
                )
            )

        n = len(hits)
        for i, (start_char, name, is_array) in enumerate(hits):
            end_char = hits[i + 1][0] if i + 1 < n else len(text)
            qual_suffix = f"{name}[]" if is_array else name
            nodes.append(
                Node(
                    kind=TOML_TABLE,
                    path=posix,
                    span=(start_char, end_char),
                    name=qual_suffix,
                    qual=f"{posix}:{qual_suffix}",
                    meta={"type": "array" if is_array else "table", "raw": name},
                    children=[],
                )
            )

        root.children.extend(nodes)
        assign_ids([root])
        flat = flatten([root])
        return Document(path=posix, text=text, roots=[root], nodes=flat)

    def supports(self, action) -> bool:
        return action.kind in {"full", "hide"}

    def render(self, doc: Document, decisions: List[Tuple[Node, object]]) -> List[Edit]:
        edits: List[Edit] = []
        for node, action in decisions:
            if action.kind == "full":
                continue
            if action.kind == "hide":
                if node.kind == TOML_DOC:
                    marker = make_omission_line(
                        lang="toml",
                        a=1,
                        b=doc.text.count("\n") + 1,
                        indent="",
                        opts=DEFAULT_OPTIONS,
                        label="document omitted",
                    )
                    edits.append(
                        Edit(path=doc.path, span=(0, len(doc.text)), replacement=marker)
                    )
                    continue

                if node.kind == TOML_TABLE:
                    s, t = node.span
                    before = doc.text[:s]
                    omitted = doc.text[s:t]
                    start_line = before.count("\n") + 1
                    end_line = start_line + omitted.count("\n")
                    label = f"table [{node.meta.get('raw', node.name)}] omitted"
                    marker = make_omission_line(
                        lang="toml",
                        a=start_line,
                        b=end_line,
                        indent="",
                        opts=DEFAULT_OPTIONS,
                        label=label,
                    )
                    edits.append(
                        Edit(path=doc.path, span=node.span, replacement=marker)
                    )
        return edits


def create_engine():
    return TomlEngine()
