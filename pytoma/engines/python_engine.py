from pathlib import Path, PurePosixPath
from typing import List, Tuple, Dict, Optional

import libcst as cst
from libcst import metadata

from ..ir import Document, Node, Edit, PY_MODULE, PY_CLASS, PY_FUNCTION, PY_METHOD
from ..policies import Action
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
    # Fallback when 'path' is not under any configured root
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


def _line_span_to_char_span(ls: List[int], start_line: int, end_line: int) -> Tuple[int, int]:
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
    """Best-effort decorator name rendering (without arguments)."""
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


# ---------------------------
# libcst-based helpers for imports & classes
# ---------------------------


def _code_for(mod: cst.Module, node: cst.CSTNode) -> str:
    # Safe code generation for a subnode
    try:
        return mod.code_for_node(node)
    except Exception:  # pragma: no cover — best effort
        return ""


def _is_top_level_docstring(stmt: cst.BaseStatement) -> bool:
    if not isinstance(stmt, cst.SimpleStatementLine):
        return False
    if not stmt.body:
        return False
    first = stmt.body[0]
    return isinstance(first, cst.Expr) and isinstance(first.value, cst.SimpleString)


def _import_from_is_future(imp: cst.ImportFrom, mod: cst.Module) -> bool:
    # from __future__ import ... (no matter the spacing)
    if imp.module is None:
        return False
    try:
        mtxt = _code_for(mod, imp.module).strip()
    except Exception:
        return False
    return mtxt == "__future__"


def _stmt_starts_with_import(stmt: cst.BaseStatement) -> Optional[cst.CSTNode]:
    """Return the first import node if the simple statement starts with one, else None."""
    if not isinstance(stmt, cst.SimpleStatementLine) or not stmt.body:
        return None
    first = stmt.body[0]
    if isinstance(first, (cst.Import, cst.ImportFrom)):
        return first
    return None


def _compute_marker_insert_pos_cst(
    source: str,
    mod: cst.Module,
    positions: metadata.PositionProvider,  # resolved provider
    ls: List[int],
) -> int:
    """
    Insert after: shebang, optional module docstring, consecutive __future__ imports,
    then consecutive top-level imports, skip blank lines.
    """
    # 0) Shebang
    shebang_end = 0
    if source.startswith("#!"):
        nl = source.find("\n")
        shebang_end = len(source) if nl == -1 else nl + 1

    insert_line = 1
    i = 0
    body = list(mod.body)

    # 1) Module docstring
    if i < len(body) and _is_top_level_docstring(body[i]):
        rng = positions[body[i]]
        insert_line = rng.end.line + 1
        i += 1

    # 2) All consecutive `from __future__ import ...`
    while i < len(body):
        first = _stmt_starts_with_import(body[i])
        if isinstance(first, cst.ImportFrom) and _import_from_is_future(first, mod):
            rng = positions[body[i]]
            insert_line = rng.end.line + 1
            i += 1
        else:
            break

    # 3) All consecutive top-level imports (excluding __future__)
    while i < len(body):
        first = _stmt_starts_with_import(body[i])
        if isinstance(first, (cst.Import, cst.ImportFrom)) and not (
            isinstance(first, cst.ImportFrom) and _import_from_is_future(first, mod)
        ):
            rng = positions[body[i]]
            insert_line = rng.end.line + 1
            i += 1
        else:
            break

    # 4) Convert to char offset, then skip blank lines
    pos = ls[min(insert_line - 1, len(ls) - 1)]
    j = pos
    n = len(source)
    while j < n:
        line_end = source.find("\n", j)
        if line_end == -1:
            line_end = n
        if source[j:line_end].strip() == "":
            j = line_end + (1 if line_end < n else 0)
        else:
            break

    return max(j, shebang_end)


def _gather_top_level_import_edits_and_names(
    mod: cst.Module,
    positions: metadata.PositionProvider,
    ls: List[int],
) -> Tuple[List[Tuple[int, int]], List[str]]:
    """
    Return (delete_spans, removed_names) for top-level imports (excluding __future__).

    This version deletes the *entire physical line* carrying the import statement,
    including its trailing newline, so that removing imports does not leave behind
    empty lines at the top of the file.
    """
    delete_spans: List[Tuple[int, int]] = []
    removed: List[str] = []

    def add_stmt_line_span(stmt: cst.SimpleStatementLine) -> None:
        r = positions[stmt]
        # Remove the whole line (from start-of-line to start-of-next-line).
        s = ls[r.start.line - 1]  # beginning of the line
        t = ls[r.end.line]        # beginning of the next line (or end-of-text)
        delete_spans.append((s, t))

    for stmt in mod.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue

        any_import = False

        for small in stmt.body:
            if isinstance(small, cst.Import):
                # Collect removed names: "pkg", "pkg as alias", ...
                for alias in small.names:
                    name_txt = _code_for(mod, alias.name).strip()
                    if alias.asname and getattr(alias.asname, "name", None):
                        as_txt = getattr(alias.asname.name, "value", "")
                        name_txt = f"{name_txt} as {as_txt}" if as_txt else name_txt
                    removed.append(name_txt)
                any_import = True

            elif isinstance(small, cst.ImportFrom):
                # Keep __future__ imports
                if _import_from_is_future(small, mod):
                    continue

                # module prefix (including relative dots)
                dot_count = 0
                if getattr(small, "relative", None) is not None:
                    rel = small.relative
                    try:
                        dot_count = len(getattr(rel, "dots", []) or [])
                    except Exception:
                        dot_count = 0
                mod_txt = _code_for(mod, small.module) if small.module is not None else ""
                prefix = "." * dot_count
                mod_full = f"{prefix}{mod_txt}" if (prefix or mod_txt) else ""

                # Names or star import
                if isinstance(small.names, cst.ImportStar):
                    removed.append(f"{mod_full}.*" if mod_full else ".*")
                else:
                    for alias in small.names:
                        base = _code_for(mod, alias.name).strip()
                        item = f"{mod_full}.{base}" if mod_full else base
                        if alias.asname and getattr(alias.asname, "name", None):
                            as_txt = getattr(alias.asname.name, "value", "")
                            item = f"{item} as {as_txt}" if as_txt else item
                        removed.append(item)

                any_import = True

        if any_import:
            add_stmt_line_span(stmt)

    return delete_spans, removed


def _drop_top_level_imports_with_marker_cst(
    source: str,
    path: str,
    *,
    mod: cst.Module,
    positions: metadata.PositionProvider,
    ls: List[int],
) -> List[Edit]:
    delete_spans, removed_names = _gather_top_level_import_edits_and_names(mod, positions, ls)
    edits: List[Edit] = [Edit(path=path, span=span, replacement="") for span in delete_spans]
    if not removed_names:
        return edits
    insert_pos = _compute_marker_insert_pos_cst(source, mod, positions, ls)
    marker = _format_imports_marker(removed_names, mode="list", max_items=4, max_chars=120) + "\n"
    edits.append(Edit(path=path, span=(insert_pos, insert_pos), replacement=marker))
    return edits


# ---------------------------
# Classes via libcst → (qual, span, parent, decorators)
# ---------------------------


class _ClassCollectorCST(cst.CSTVisitor):
    def __init__(self, positions: metadata.PositionProvider, ls: List[int]) -> None:
        self.positions = positions
        self.ls = ls
        self.stack: List[str] = []
        self.items: List[Dict[str, object]] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        # Span should include decorators — use the earliest decorator start
        r_node = self.positions[node]
        start_line = r_node.start.line
        if node.decorators:
            deco_starts = [self.positions[d].start.line for d in node.decorators]
            start_line = min(deco_starts + [start_line])
        end_line = r_node.end.line
        end_col = r_node.end.column
        start = self.ls[start_line - 1]
        end = self.ls[end_line - 1] + end_col

        qual_local = ".".join(self.stack + [node.name.value]) if self.stack else node.name.value
        parent_local = ".".join(self.stack) if self.stack else None
        decos = [_decorator_to_str_py(d) for d in (node.decorators or [])]

        self.items.append(
            {
                "name": node.name.value,
                "qual_local": qual_local,
                "parent_local": parent_local,
                "span": (start, end),
                "decorators": decos,
            }
        )
        self.stack.append(node.name.value)

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self.stack.pop()


# ---------------------------
# Collect FuncInfo via libcst (and expose module & positions)
# ---------------------------


def _collect_funcs(text: str, path: Path) -> Tuple[List[str], List[FuncInfo], cst.Module, metadata.PositionProvider]:
    src = text.splitlines(keepends=True)
    mod = cst.parse_module(text)
    wrapper = metadata.MetadataWrapper(mod, unsafe_skip_copy=True)
    posmap = wrapper.resolve(metadata.PositionProvider)
    module = _module_name(path)
    collector = FuncCollector(module_name=module, source_lines=src, posmap=posmap)
    wrapper.visit(collector)
    return src, collector.funcs, mod, posmap


def _mode_of_action(a: Action) -> Optional[str]:
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
        src, funcs, cst_mod, posmap = _collect_funcs(text, path)
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
            cv = _ClassCollectorCST(posmap, ls)
            cst_mod.visit(cv)
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
                    name=name,  # type: ignore[arg-type]
                    qual=qual_full,
                    meta={
                        "decorators": decorators,  # type: ignore[dict-item]
                        "visibility": _visibility(name),  # type: ignore[arg-type]
                    },
                    children=[],
                )
                if parent_local and parent_local in cls_nodes:
                    cls_nodes[parent_local].children.append(n)
                else:
                    root.children.append(n)
                cls_nodes[qual_local] = n  # type: ignore[index]
        except Exception:
            # Best-effort: keep going without class structure on parse errors
            pass

        # --- Functions & methods (including nested defs) ---
        fn_nodes: Dict[str, Node] = {}  # local qual ("f", "Cls.m", "f.inner") -> Node
        for fi in funcs:
            # Span (includes decorators)
            s = ls[fi.deco_start_line - 1]
            e = ls[fi.end[0] - 1] + fi.end[1]
            local = fi.qualname.split(":", 1)[1]  # "func" or "Class.meth" or "outer.inner"
            parts = local.split(".")
            name = parts[-1]
            parent_local = ".".join(parts[:-1]) if len(parts) > 1 else None

            # Detect decorators (best-effort textual)
            decos: List[str] = []
            if fi.node.decorators:
                for d in fi.node.decorators:
                    decos.append(_decorator_to_str_py(d))

            # Type (method if parent is a class)
            parent: Optional[Node]
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

        # --- Analysis cache to avoid any re-parsing in render() ---
        analysis = {
            "src_lines": src,  # List[str]
            "line_starts": ls,  # List[int]
            "funcs_by_qual": {fi.qualname: fi for fi in funcs},
            "cst_module": cst_mod,
            "posmap": posmap,
        }
        root.meta["analysis"] = analysis

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
        # Reuse the analysis stored by parse() — no double parsing
        analysis: Dict[str, object] = {}
        for r in doc.roots:
            if r.kind == PY_MODULE and (r.meta or {}).get("analysis"):
                analysis = r.meta["analysis"]  # type: ignore[assignment]
                break

        # Defensive fallbacks for compatibility (should rarely trigger)
        if analysis:
            src: List[str] = analysis.get("src_lines") or doc.text.splitlines(keepends=True)  # type: ignore[assignment]
            ls: List[int] = analysis.get("line_starts") or _line_starts(doc.text)  # type: ignore[assignment]
            by_qual: Dict[str, FuncInfo] = analysis.get("funcs_by_qual") or {}  # type: ignore[assignment]
            cst_mod: Optional[cst.Module] = analysis.get("cst_module")  # type: ignore[assignment]
            posmap: Optional[metadata.PositionProvider] = analysis.get("posmap")  # type: ignore[assignment]
        else:
            # Old behavior (fallback): minimal re-parse
            src, funcs, cst_mod, posmap = _collect_funcs(doc.text, Path(doc.path))
            ls = _line_starts(doc.text)
            by_qual = {fi.qualname: fi for fi in funcs}

        candidates: List[Edit] = []
        for node, action in decisions:
            # --- file-level filters ---
            if node.kind == PY_MODULE and action.kind == "file:no-imports":
                if cst_mod is not None and posmap is not None:
                    candidates.extend(
                        _drop_top_level_imports_with_marker_cst(
                            doc.text,
                            doc.path,
                            mod=cst_mod,
                            positions=posmap,
                            ls=ls,
                        )
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
                candidates.append(Edit(path=doc.path, span=(0, len(doc.text)), replacement=marker))
                continue

            if node.kind == PY_CLASS:
                if mode == "hide":
                    # Compute the line range of the class
                    s, t = node.span
                    before = doc.text[:s]
                    omitted_text = doc.text[s:t]
                    start_line = before.count("\n") + 1
                    end_line = start_line + omitted_text.count("\n")
                    indent = ""  # could be refined by reusing the indent of the first line
                    marker = make_omission_line(
                        "py",
                        start_line,
                        end_line,
                        indent=indent,
                        opts=DEFAULT_OPTIONS,
                        label=f"class {node.name} omitted" if node.name else "class omitted",
                    )
                    candidates.append(Edit(path=doc.path, span=node.span, replacement=marker))
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
                        candidates.append(Edit(path=doc.path, span=node.span, replacement=marker))
                    continue

                rep = _replacement_for_mode(mode, fi, src)
                if not rep:
                    continue
                a_ln, b_ln, block = rep
                span = _line_span_to_char_span(ls, a_ln, b_ln)
                candidates.append(Edit(path=doc.path, span=span, replacement=block))

        # No deduplication/pruning here: delegate to fs.merge_edits
        return candidates



def create_engine():
    return PythonMinEngine()
