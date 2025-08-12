# pytoma/engines/python_min.py
from pathlib import Path, PurePosixPath
from typing import List, Tuple, Dict, Optional

import re
import ast
import libcst as cst
from libcst import metadata

from ..ir import Document, Node, Edit, Span, PY_MODULE, PY_CLASS, PY_FUNCTION, PY_METHOD
from ..policies import Action
from ..base import register_engine
from ..render import _replacement_for_mode  # rendu Python existant
from ..collect import FuncCollector, FuncInfo, file_to_module_name
from ..ir import assign_ids, flatten

# ---------------------------
# Utils + module roots
# ---------------------------

_MODULE_ROOTS: List[Path] = []

def set_module_roots(roots: List[Path]) -> None:
    global _MODULE_ROOTS
    _MODULE_ROOTS = [r.resolve() for r in roots]

def _module_name(path: Path) -> str:
    for root in _MODULE_ROOTS:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return file_to_module_name(path, root)
    p = path.with_suffix("")
    parts = list(p.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)

def _line_starts(text: str) -> List[int]:
    starts = [0]
    acc = 0
    for line in text.splitlines(keepends=True):
        acc += len(line)
        starts.append(acc)
    return starts

def _line_span_to_char_span(ls: List[int], start_line: int, end_line: int) -> Span:
    s = ls[start_line - 1]
    t = ls[end_line]
    return (s, t)

def _visibility(name: str) -> str:
    if name.startswith("__") and name.endswith("__"):
        return "dunder"
    if name.startswith("_"):
        return "private"
    return "public"

def _decorator_to_str_py(fn_dec: cst.Decorator) -> str:
    # Rendu best-effort (sans les arguments si appel complexe)
    expr = fn_dec.decorator
    def _name(e: cst.CSTNode) -> Optional[str]:
        if isinstance(e, cst.Name):
            return e.value
        if isinstance(e, cst.Attribute):
            left = _name(e.value) or ""
            return f"{left}.{e.attr.value}".lstrip(".")
        if isinstance(e, cst.Call):
            return _name(e.func)  # @decorator(args) -> "decorator"
        return None
    return _name(expr) or "decorator"

# ---------------------------
# Classes (via ast) → (qual, span, parent, decorators)
# ---------------------------

class _ClassVisitor(ast.NodeVisitor):
    """Collecte les classes (imbriquées) avec qualnames et décorateurs."""
    def __init__(self, text: str) -> None:
        self.text = text
        self.stack: List[str] = []
        self.items: List[Dict[str, object]] = []
        self._ls = _line_starts(text)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        start_ln = min(
            [getattr(d, "lineno", node.lineno) for d in getattr(node, "decorator_list", [])] + [node.lineno]
        )
        end_ln = node.end_lineno
        end_col = node.end_col_offset
        start = self._ls[start_ln - 1]
        end = self._ls[end_ln - 1] + end_col
        qual_local = ".".join(self.stack + [node.name]) if self.stack else node.name
        parent_local = ".".join(self.stack) if self.stack else None
        # décorateurs via ast.unparse (py>=3.9)
        decos = []
        for d in getattr(node, "decorator_list", []):
            try:
                decos.append(ast.unparse(d))
            except Exception:
                decos.append("decorator")
        self.items.append({
            "name": node.name,
            "qual_local": qual_local,
            "parent_local": parent_local,
            "span": (start, end),
            "decorators": decos,
        })
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

# ---------------------------
# Collect FuncInfo via libcst
# ---------------------------

def _collect_funcs(text: str, path: Path) -> Tuple[List[str], List[FuncInfo]]:
    src = text.splitlines(keepends=True)
    mod = cst.parse_module(text)
    wrapper = metadata.MetadataWrapper(mod, unsafe_skip_copy=True)
    posmap = wrapper.resolve(metadata.PositionProvider)
    module = _module_name(path)
    collector = FuncCollector(module_name=module, source_lines=src, posmap=posmap)
    wrapper.visit(collector)
    return src, collector.funcs

def _mode_of_action(a: Action) -> str | None:
    k = a.kind
    if k == "full": return "full"
    if k == "hide": return "hide"
    if k == "sig": return "sig"
    if k == "sig+doc": return "sig+doc"
    if k == "levels":
        kk = int(a.params.get("k", 1))
        return f"body:levels={kk}"
    return None

# ---------------------------
# Engine
# ---------------------------

class PythonMinEngine:
    """
    Granularité fonction/méthode, désormais en arbre (module → classes → defs).
    """
    filetypes = {"py"}

    # NEW: optional hook
    def configure(self, roots: List[Path]) -> None:
        try:
            set_module_roots(roots)
        except Exception:
            pass

    def parse(self, path: Path, text: str) -> Document:
        posix = PurePosixPath(path.as_posix())
        src, funcs = _collect_funcs(text, path)
        ls = _line_starts(text)
        module_name = _module_name(path)

        # Racine module
        root = Node(
            kind=PY_MODULE,
            path=posix,
            span=(0, len(text)),
            name=str(posix),
            qual=f"{module_name}",
            meta={},
            children=[]
        )

        # --- Classes (avec imbrication) ---
        cls_nodes: Dict[str, Node] = {}
        try:
            t_ast = ast.parse(text)
            cv = _ClassVisitor(text)
            cv.visit(t_ast)
            for item in cv.items:
                qual_local = item["qual_local"]  # ex: "Outer.Inner"
                parent_local = item["parent_local"]
                name = item["name"]
                span = item["span"]
                decorators = item["decorators"]
                qual_full = f"{module_name}:{qual_local}"
                n = Node(
                    kind=PY_CLASS,
                    path=posix,
                    span=span,  # char-based
                    name=name,
                    qual=qual_full,
                    meta={
                        "decorators": decorators,
                        "visibility": _visibility(name),
                    },
                    children=[]
                )
                if parent_local and parent_local in cls_nodes:
                    cls_nodes[parent_local].children.append(n)
                else:
                    root.children.append(n)
                cls_nodes[qual_local] = n
        except SyntaxError:
            pass

        # --- Fonctions & méthodes (inclut defs imbriquées) ---
        fn_nodes: Dict[str, Node] = {}  # local qual ("f", "Cls.m", "f.inner") -> Node
        for fi in funcs:
            # Span (inclut décorateurs)
            s = ls[fi.deco_start_line - 1]
            e = ls[fi.end[0] - 1] + fi.end[1]
            local = fi.qualname.split(":", 1)[1]   # "func" ou "Class.meth" ou "outer.inner"
            parts = local.split(".")
            name = parts[-1]
            parent_local = ".".join(parts[:-1]) if len(parts) > 1 else None

            # Détecte décorateurs (best-effort textuel)
            decos: List[str] = []
            if fi.node.decorators:
                for d in fi.node.decorators:
                    decos.append(_decorator_to_str_py(d))

            # Type (méthode si parent est une classe)
            parent: Optional[Node] = None
            kind = PY_FUNCTION
            if parent_local and parent_local in cls_nodes:
                parent = cls_nodes[parent_local]
                kind = PY_METHOD
            elif parent_local and parent_local in fn_nodes:
                parent = fn_nodes[parent_local]  # def imbriquée
            else:
                parent = root

            n = Node(
                kind=kind,
                path=posix,
                span=(s, e),
                name=name,
                qual=fi.qualname,
                meta={
                    "decorators": decos,
                    "visibility": _visibility(name),
                    "has_doc": bool(fi.docstring),
                },
                children=[]
            )
            parent.children.append(n)
            fn_nodes[local] = n

        # IDs + liste plate
        assign_ids([root])
        flat = flatten([root])

        return Document(path=posix, text=text, roots=[root], nodes=flat)

    def supports(self, action: Action) -> bool:
        return _mode_of_action(action) in {"full", "hide", "sig", "sig+doc"} or action.kind == "levels"

    def render(self, doc: Document, decisions: List[Tuple[Node, Action]]) -> List[Edit]:
        # Rendu inchangé : on se base sur FuncInfo (plus robuste pour les remplacements)
        src, funcs = _collect_funcs(doc.text, Path(doc.path))
        ls = _line_starts(doc.text)
        by_qual: Dict[str, FuncInfo] = { fi.qualname: fi for fi in funcs }

        candidates: List[Edit] = []
        for node, action in decisions:
            mode = _mode_of_action(action)
            if not mode or mode == "full":
                continue
            if node.kind == PY_MODULE and mode == "hide":
                candidates.append(Edit(path=doc.path, span=(0, len(doc.text)), replacement=""))
                continue
            if node.kind == PY_CLASS:
                if mode == "hide":
                    candidates.append(Edit(path=doc.path, span=node.span, replacement=""))
                continue
            if node.kind in {PY_FUNCTION, PY_METHOD}:
                fi = by_qual.get(node.qual or "")
                if not fi:
                    if mode == "hide":
                        candidates.append(Edit(path=doc.path, span=node.span, replacement=""))
                    continue
                rep = _replacement_for_mode(mode, fi, src)
                if not rep:
                    continue
                a_ln, b_ln, block = rep
                span = _line_span_to_char_span(ls, a_ln, b_ln)
                candidates.append(Edit(path=doc.path, span=span, replacement=block))

        # Aucune déduplication/pruning ici : on délègue à fs.merge_edits
        return candidates

# registration
register_engine(PythonMinEngine())

