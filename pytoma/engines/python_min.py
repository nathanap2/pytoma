from pathlib import Path, PurePosixPath
from typing import List, Tuple, Dict, Optional

import re
import ast
import libcst as cst
from libcst import metadata

from ..ir import Document, Node, Edit, Span, PY_MODULE, PY_CLASS, PY_FUNCTION, PY_METHOD
from ..policies import Action
from ..base import register_engine
from ..render import _replacement_for_mode  # existing Python rendering
from ..markers import make_omission_line, DEFAULT_OPTIONS
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
    # Best-effort rendering (without arguments if it's a complex call)
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


def _format_imports_marker(
    items: List[str], *, mode: str = "list", max_items: int = 4, max_chars: int = 120
) -> str:
    items = sorted(dict.fromkeys(items))
    if mode == "count":
        return f"# [imports omitted: {len(items)}]"
    text = f"# [imports omitted: {len(items)}] " + ", ".join(items[:max_items])
    if len(items) > max_items:
        rest = len(items) - max_items
        text += f"…(+{rest})"
    if len(text) > max_chars:
        return f"# [imports omitted: {len(items)}]"
    return text


def _compute_marker_insert_pos(source: str, tree: ast.Module, ls: List[int]) -> int:
    """
    Insertion position = after docstring + __future__ imports + contiguous top-level import block,
    then skip blank lines. Never before an eventual shebang.
    """
    # 1) Respect a shebang at the top
    shebang_end = 0
    if source.startswith("#!"):
        nl = source.find("\n")
        shebang_end = len(source) if nl == -1 else nl + 1

    insert_line = 1
    body = list(tree.body)
    j = 0

    # 2) Module docstring
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        end_ln = getattr(body[0], "end_lineno", body[0].lineno)
        insert_line = end_ln + 1
        j = 1

    # 3) All consecutive `from __future__ import ...` statements
    while j < len(body):
        stmt = body[j]
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            end_ln = getattr(stmt, "end_lineno", stmt.lineno)
            insert_line = end_ln + 1
            j += 1
        else:
            break

    # 4) All consecutive top-level imports (excluding __future__) AFTER that
    k = j
    while k < len(body):
        stmt = body[k]
        if isinstance(stmt, ast.Import) or (
            isinstance(stmt, ast.ImportFrom) and stmt.module != "__future__"
        ):
            end_ln = getattr(stmt, "end_lineno", stmt.lineno)
            insert_line = end_ln + 1
            k += 1
        else:
            break

    # 5) Convert to character offset, then skip blank lines
    pos = ls[min(insert_line - 1, len(ls) - 1)]
    i = pos
    n = len(source)
    while i < n:
        line_end = source.find("\n", i)
        if line_end == -1:
            line_end = n
        if source[i:line_end].strip() == "":
            i = line_end + (1 if line_end < n else 0)
        else:
            break

    # 6) Never insert before a shebang
    return max(i, shebang_end)


def _drop_top_level_imports_with_marker(source: str, path: str) -> List[Edit]:
    tree = ast.parse(source)
    ls = _line_starts(source)  # already defined in this module

    removed_names: List[str] = []
    delete_spans: List[Tuple[int, int]] = []

    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            # Imported names
            for a in stmt.names:
                name = a.name
                removed_names.append(f"{name} as {a.asname}" if a.asname else name)
            # Remove: entire line(s) of the statement
            start = ls[stmt.lineno - 1]
            end = ls[stmt.end_lineno] if stmt.end_lineno < len(ls) else len(source)
            delete_spans.append((start, end))

        elif isinstance(stmt, ast.ImportFrom):
            # Keep __future__ imports
            if stmt.module == "__future__":
                continue
            # Module, including relative imports: "." * level + (module or "")
            mod_prefix = "." * getattr(stmt, "level", 0)
            mod = (
                (mod_prefix + (stmt.module or ""))
                if (getattr(stmt, "level", 0) or stmt.module)
                else ""
            )
            if len(stmt.names) == 1 and stmt.names[0].name == "*":
                removed_names.append(f"{mod}.*" if mod else ".*")
            else:
                for a in stmt.names:
                    item = f"{mod}.{a.name}" if mod else a.name
                    removed_names.append(f"{item} as {a.asname}" if a.asname else item)
            start = ls[stmt.lineno - 1]
            end = ls[stmt.end_lineno] if stmt.end_lineno < len(ls) else len(source)
            delete_spans.append((start, end))

    edits: List[Edit] = [
        Edit(path=path, span=span, replacement="") for span in delete_spans
    ]

    if not removed_names:
        return edits  # nothing to mark

    insert_pos = _compute_marker_insert_pos(source, tree, ls)
    marker = (
        _format_imports_marker(removed_names, mode="list", max_items=4, max_chars=120)
        + "\n"
    )
    edits.append(Edit(path=path, span=(insert_pos, insert_pos), replacement=marker))
    return edits


# ---------------------------
# Classes (via ast) → (qual, span, parent, decorators)
# ---------------------------


class _ClassVisitor(ast.NodeVisitor):
    """Collect nested classes with qualified names and decorators."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.stack: List[str] = []
        self.items: List[Dict[str, object]] = []
        self._ls = _line_starts(text)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        start_ln = min(
            [
                getattr(d, "lineno", node.lineno)
                for d in getattr(node, "decorator_list", [])
            ]
            + [node.lineno]
        )
        end_ln = node.end_lineno
        end_col = node.end_col_offset
        start = self._ls[start_ln - 1]
        end = self._ls[end_ln - 1] + end_col
        qual_local = ".".join(self.stack + [node.name]) if self.stack else node.name
        parent_local = ".".join(self.stack) if self.stack else None
        # decorators via ast.unparse (py>=3.9)
        decos = []
        for d in getattr(node, "decorator_list", []):
            try:
                decos.append(ast.unparse(d))
            except Exception:
                decos.append("decorator")
        self.items.append(
            {
                "name": node.name,
                "qual_local": qual_local,
                "parent_local": parent_local,
                "span": (start, end),
                "decorators": decos,
            }
        )
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
    if k == "full":
        return "full"
    if k == "hide":
        return "hide"
    if k == "sig":
        return "sig"
    if k == "sig+doc":
        return "sig+doc"
    if k == "levels":
        kk = int(a.params.get("k", 1))
        return f"body:levels={kk}"
    return None


# ---------------------------
# Engine
# ---------------------------


class PythonMinEngine:
    """
    Function/method granularity, structured as a tree (module → classes → defs).
    """

    filetypes = {"py"}

    # Optional hook
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

        # Module root
        root = Node(
            kind=PY_MODULE,
            path=posix,
            span=(0, len(text)),
            name=str(posix),
            qual=f"{module_name}",
            meta={},
            children=[],
        )

        # --- Classes (with nesting) ---
        cls_nodes: Dict[str, Node] = {}
        try:
            t_ast = ast.parse(text)
            cv = _ClassVisitor(text)
            cv.visit(t_ast)
            for item in cv.items:
                qual_local = item["qual_local"]  # e.g., "Outer.Inner"
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
                    children=[],
                )
                if parent_local and parent_local in cls_nodes:
                    cls_nodes[parent_local].children.append(n)
                else:
                    root.children.append(n)
                cls_nodes[qual_local] = n
        except SyntaxError:
            pass

        # --- Functions & methods (including nested defs) ---
        fn_nodes: Dict[str, Node] = {}  # local qual ("f", "Cls.m", "f.inner") -> Node
        for fi in funcs:
            # Span (includes decorators)
            s = ls[fi.deco_start_line - 1]
            e = ls[fi.end[0] - 1] + fi.end[1]
            local = fi.qualname.split(":", 1)[
                1
            ]  # "func" or "Class.meth" or "outer.inner"
            parts = local.split(".")
            name = parts[-1]
            parent_local = ".".join(parts[:-1]) if len(parts) > 1 else None

            # Detect decorators (best-effort textual)
            decos: List[str] = []
            if fi.node.decorators:
                for d in fi.node.decorators:
                    decos.append(_decorator_to_str_py(d))

            # Type (method if parent is a class)
            parent: Optional[Node] = None
            kind = PY_FUNCTION
            if parent_local and parent_local in cls_nodes:
                parent = cls_nodes[parent_local]
                kind = PY_METHOD
            elif parent_local and parent_local in fn_nodes:
                parent = fn_nodes[parent_local]  # nested def
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
                children=[],
            )
            parent.children.append(n)
            fn_nodes[local] = n

        # IDs + flat list
        assign_ids([root])
        flat = flatten([root])

        return Document(path=posix, text=text, roots=[root], nodes=flat)

    def supports(self, action: Action) -> bool:
        return (
            _mode_of_action(action) in {"full", "hide", "sig", "sig+doc"}
            or action.kind == "levels"
            or action.kind in {"file:no-imports"}
        )

    def render(self, doc: Document, decisions: List[Tuple[Node, Action]]) -> List[Edit]:
        # Rendering unchanged: based on FuncInfo (more robust for replacements)
        src, funcs = _collect_funcs(doc.text, Path(doc.path))
        ls = _line_starts(doc.text)
        by_qual: Dict[str, FuncInfo] = {fi.qualname: fi for fi in funcs}

        candidates: List[Edit] = []
        for node, action in decisions:
            # --- file-level filters ---
            if node.kind == PY_MODULE and action.kind == "file:no-imports":
                candidates.extend(
                    _drop_top_level_imports_with_marker(doc.text, doc.path)
                )
                continue

            mode = _mode_of_action(action)
            if not mode or mode == "full":
                continue
            if node.kind == PY_MODULE and mode == "hide":
                # Mark omission of the entire module
                marker = make_omission_line(
                    "py",
                    1,
                    doc.text.count("\n") + 1,
                    indent="",
                    opts=DEFAULT_OPTIONS,
                    label="module omitted",
                )
                candidates.append(
                    Edit(path=doc.path, span=(0, len(doc.text)), replacement=marker)
                )
                continue
            if node.kind == PY_CLASS:
                if mode == "hide":
                    # Compute the line range of the class
                    s, t = node.span
                    before = doc.text[:s]
                    omitted_text = doc.text[s:t]
                    start_line = before.count("\n") + 1
                    end_line = start_line + omitted_text.count("\n")
                    indent = (
                        ""  # could be refined by reusing the indent of the first line
                    )
                    marker = make_omission_line(
                        "py",
                        start_line,
                        end_line,
                        indent=indent,
                        opts=DEFAULT_OPTIONS,
                        label=f"class {node.name} omitted"
                        if node.name
                        else "class omitted",
                    )
                    candidates.append(
                        Edit(path=doc.path, span=node.span, replacement=marker)
                    )
                continue
            if node.kind in {PY_FUNCTION, PY_METHOD}:
                fi = by_qual.get(node.qual or "")
                if not fi:
                    if mode == "hide":
                        s, t = node.span
                        before = doc.text[:s]
                        omitted_text = doc.text[s:t]
                        start_line = before.count("\n") + 1
                        end_line = start_line + omitted_text.count("\n")
                        indent = ""
                        marker = make_omission_line(
                            "py",
                            start_line,
                            end_line,
                            indent=indent,
                            opts=DEFAULT_OPTIONS,
                            label="definition omitted",
                        )
                        candidates.append(
                            Edit(path=doc.path, span=node.span, replacement=marker)
                        )
                    continue
                rep = _replacement_for_mode(mode, fi, src)
                if not rep:
                    continue
                a_ln, b_ln, block = rep
                span = _line_span_to_char_span(ls, a_ln, b_ln)
                candidates.append(Edit(path=doc.path, span=span, replacement=block))

        # No deduplication/pruning here: delegate to fs.merge_edits
        return candidates


# registration
register_engine(PythonMinEngine())
