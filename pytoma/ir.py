from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Optional, List, Tuple, Dict, Iterable
import hashlib

Span = Tuple[int, int]  # [start, end) in character indices

PY_MODULE   = "py:module"
PY_CLASS    = "py:class"
PY_FUNCTION = "py:function"
PY_METHOD   = "py:method"

# Markdown
MD_DOC      = "md:doc"
MD_HEADING  = "md:heading"

# Toml
TOML_DOC    = "toml:doc"
TOML_TABLE  = "toml:table"

# WIP ... YAML_KEY = "yaml:key", ...

def compute_node_id(path: PurePosixPath, kind: str, qual: Optional[str], span: Span) -> str:
    h = hashlib.sha1()
    h.update(str(path).encode("utf-8")); h.update(b"\x00")
    h.update(kind.encode("utf-8"));      h.update(b"\x00")
    h.update((qual or "").encode("utf-8")); h.update(b"\x00")
    h.update(f"{span[0]}:{span[1]}".encode("utf-8"))
    return h.hexdigest()[:16]

@dataclass
class Node:
    kind: str
    path: PurePosixPath
    span: Span
    name: Optional[str] = None
    qual: Optional[str] = None
    meta: Dict[str, object] = field(default_factory=dict)
    children: List["Node"] = field(default_factory=list)
    node_id: Optional[str] = None

@dataclass
class Document:
    path: PurePosixPath
    text: str
    roots: List[Node] = field(default_factory=list)
    nodes: List[Node] = field(default_factory=list)

def _walk_preorder(n: Node) -> Iterable[Node]:
    yield n
    for c in n.children:
        yield from _walk_preorder(c)

def assign_ids(roots: List[Node]) -> None:
    for r in roots:
        for n in _walk_preorder(r):
            n.node_id = compute_node_id(n.path, n.kind, n.qual, n.span)

def flatten(roots: List[Node]) -> List[Node]:
    out: List[Node] = []
    for r in roots:
        out.extend(list(_walk_preorder(r)))
    return out

@dataclass
class Edit:
    path: PurePosixPath
    span: Span
    replacement: str
    comment: Optional[str] = None

