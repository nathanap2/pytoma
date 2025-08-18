from __future__ import annotations
from pathlib import Path, PurePosixPath
from typing import List, Tuple, Optional, Dict

import re

from pytoma.ir import Document, Edit, Node, assign_ids, flatten
from pytoma.ir import XML_DOC, XML_ELEMENT
from pytoma.markers import make_omission_line, DEFAULT_OPTIONS
from pytoma.utils import line_starts

# --- mini XML scanner --------------------------------------------------------
_NAME = r"[A-Za-z_:][\w.\-:]*"

RE_COMMENT_OPEN = re.compile(r"<!--")
RE_COMMENT_CLOSE = re.compile(r"-->")
RE_CDATA_OPEN = re.compile(r"<!\[CDATA\[")
RE_CDATA_CLOSE = re.compile(r"\]\]>")
RE_PI = re.compile(r"<\?(?:[^?]|\?[^>])*\?>")  # <? ... ?>
RE_DOCTYPE = re.compile(r"<!DOCTYPE[^>]*>")  # simplified (no internal subset)
RE_OPEN_OR_EMPTY = re.compile(
    rf"<\s*(?P<name>{_NAME})(?P<attrs>(?:\"[^\"]*\"|'[^']*'|[^>])*?)\s*(?P<self>/?)>"
)
RE_CLOSE = re.compile(rf"</\s*(?P<name>{_NAME})\s*>")


def _scan_elements(text: str) -> List[Tuple[int, int, str, bool]]:
    """
    Returns a list of structured events:
      - (pos, end, tagname, is_open) for open/empty/close
    Where:
      * normal open   : is_open=True, end = end of '>' in the start-tag
      * empty <tag/>  : is_open=True then *immediate* synthetic false-close for simplicity
      * close </tag>  : is_open=False, end = end of '>'
    Comments, CDATA, PI, DOCTYPE are properly skipped.
    """
    i, n = 0, len(text)
    events: List[Tuple[int, int, str, bool]] = []

    while i < n:
        ch = text.find("<", i)
        if ch < 0:
            break

        # <!-- comment -->
        if text.startswith("<!--", ch):
            m2 = RE_COMMENT_CLOSE.search(text, ch + 4)
            i = m2.end() if m2 else n
            continue

        # <![CDATA[ ... ]]>
        if text.startswith("<![CDATA[", ch):
            m2 = RE_CDATA_CLOSE.search(text, ch + 9)
            i = m2.end() if m2 else n
            continue

        # <? ... ?>
        m = RE_PI.match(text, ch)
        if m:
            i = m.end()
            continue

        # <!DOCTYPE ...>
        m = RE_DOCTYPE.match(text, ch)
        if m:
            i = m.end()
            continue

        # closing tag
        m = RE_CLOSE.match(text, ch)
        if m:
            name = m.group("name")
            events.append((ch, m.end(), name, False))
            i = m.end()
            continue

        # opening or empty tag
        m = RE_OPEN_OR_EMPTY.match(text, ch)
        if m:
            name = m.group("name")
            is_self = bool(m.group("self"))
            events.append((ch, m.end(), name, True))
            i = m.end()
            if is_self:
                # synthesize an immediate 'close' to simplify stack handling
                events.append((m.end(), m.end(), name, False))
            continue

        # fallback: not a recognized tag, advance by one character
        i = ch + 1

    return events


def _line_range(text: str, span: Tuple[int, int]) -> Tuple[int, int]:
    s, t = span
    before = text[:s]
    omitted = text[s:t]
    start_line = before.count("\n") + 1
    end_line = start_line + omitted.count("\n")
    return start_line, end_line


class XmlMinEngine:
    """
    'min' engine for XML:
      - parse() builds a Document with a tree of elements, each node has a qualname
        of indexed-path type: '/root[1]/item[2]/leaf[1]'.
      - supports(): 'full' and 'hide'
      - render(): replaces a hidden element with an XML comment including line count.
    """

    filetypes = {"xml"}

    def configure(self, roots: List[Path]) -> None:
        return

    def parse(self, path: Path, text: str) -> Document:
        posix = PurePosixPath(path.as_posix())
        ls = line_starts(text)

        root = Node(
            kind=XML_DOC,
            path=posix,
            span=(0, len(text)),
            name=str(posix),
            qual=str(posix),
            meta={},
            children=[],
        )

        # Stack of frames: (node, counters_per_tag: Dict[str, int])
        stack: List[Tuple[Node, Dict[str, int]]] = []

        def _new_index(counters: Dict[str, int], tag: str) -> int:
            counters[tag] = counters.get(tag, 0) + 1
            return counters[tag]

        def _xpath(parent: Optional[Node], counters: Dict[str, int], tag: str) -> str:
            idx = counters[tag]
            if parent is None or not (parent.meta and parent.meta.get("xpath")):
                return f"/{tag}[{idx}]"
            return f"{parent.meta['xpath']}/{tag}[{idx}]"  # type: ignore[index]

        events = _scan_elements(text)

        # Virtual document parent for the root
        doc_counters: Dict[str, int] = {}
        doc_frame = (root, doc_counters)
        stack.append(doc_frame)

        for pos, pos_end, name, is_open in events:
            if is_open:
                parent_node, parent_counters = stack[-1]
                k = _new_index(parent_counters, name)
                xpath = _xpath(
                    parent_node if parent_node is not root else None,
                    parent_counters,
                    name,
                )

                node = Node(
                    kind=XML_ELEMENT,
                    path=posix,
                    span=(pos, pos_end),  # end updated at closing
                    name=f"{name}[{k}]",
                    qual=f"{posix}:{xpath}",
                    meta={"tag": name, "index": k, "xpath": xpath},
                    children=[],
                )
                parent_node.children.append(node)
                # Open a new frame for a normal opening
                if pos != pos_end:  # true open
                    stack.append((node, {}))
            else:
                # closing: pop until finding the current tag (robust against malformed XML)
                # In well-formed XML, this will be the top of the stack.
                j = len(stack) - 1
                while j > 0:
                    cand_node, _ = stack[j]
                    if (
                        cand_node.kind == XML_ELEMENT
                        and cand_node.meta.get("tag") == name
                    ):
                        # finalize the element’s span
                        s, _ = cand_node.span
                        cand_node.span = (s, pos_end)
                        # close the frame
                        stack = stack[:j]
                        break
                    j -= 1
                # otherwise: orphan close -> ignored

        assign_ids([root])
        flat = flatten([root])
        return Document(path=posix, text=text, roots=[root], nodes=flat)

    def supports(self, action) -> bool:
        return action.kind in {"full", "hide"}

    def render(self, doc: Document, decisions: List[Tuple[Node, object]]) -> List[Edit]:
        edits: List[Edit] = []

        for node, action in decisions:
            if action.kind != "hide":
                continue

            if node.kind == XML_DOC:
                marker = make_omission_line(
                    lang="md",  # generates <!-- ... -->, valid in XML
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

            if node.kind == XML_ELEMENT:
                a, b = _line_range(doc.text, node.span)
                tag = (node.meta or {}).get("tag", node.name)  # type: ignore[assignment]
                label = f"element <{tag}> omitted"
                marker = make_omission_line(
                    lang="md",  # XML comment
                    a=a,
                    b=b,
                    indent="",
                    opts=DEFAULT_OPTIONS,
                    label=label,
                )
                edits.append(Edit(path=doc.path, span=node.span, replacement=marker))

        return edits


def create_engine():
    return XmlMinEngine()
