from __future__ import annotations
from pathlib import Path, PurePosixPath
from typing import List, Tuple, Dict, Optional

import libcst as cst
import re
from libcst import metadata

from ..ir import Document, Node, Edit, PY_MODULE, PY_CLASS, PY_FUNCTION, PY_METHOD
from ..policies import Action
from ..render import _replacement_for_mode  # existing Python rendering
from ..markers import make_omission_line, DEFAULT_OPTIONS
from ..collect import FuncCollector, FuncInfo, file_to_module_name
from ..ir import assign_ids, flatten
from ..utils import line_starts


_MODULE_ROOTS: List[Path] = []  # normalized and sorted, shallow → deep


def _depth(p: Path) -> int:
    """Return the path depth (number of components) after resolving symlinks."""
    return len(p.resolve().parts)


def set_module_roots(roots: List[Path]) -> None:
    """
    Register module roots in a deterministic, order-independent way:
    - resolve to absolute paths,
    - deduplicate,
    - sort by ascending depth so shallower roots come first.
    This makes module name resolution stable regardless of the order
    in which roots are provided to the CLI.
    """
    global _MODULE_ROOTS
    seen: dict[Path, Path] = {}
    for r in roots:
        rr = r.resolve()
        seen[rr] = rr  # dedupe by resolved absolute path
    _MODULE_ROOTS = sorted(seen.values(), key=_depth)


def _module_name(path: Path) -> str:
    """
    Derive a dotted Python module name for `path` by selecting the
    *shallowest* registered root that contains it. This removes any
    dependency on the order of the provided roots.

    Fallback: if no root contains `path`, derive the name from the
    filesystem path itself (old behavior).
    """
    best: Optional[Path] = None
    for root in _MODULE_ROOTS:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if best is None or _depth(root) < _depth(best):
            best = root

    if best is not None:
        return file_to_module_name(path, best)

    # Fallback when `path` is not under any configured root.
    p = path.with_suffix("")
    parts = list(p.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _line_span_to_char_span(
    ls: List[int], start_line: int, end_line: int
) -> Tuple[int, int]:
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
        t = ls[r.end.line]  # beginning of the next line (or end-of-text)
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
                mod_txt = (
                    _code_for(mod, small.module) if small.module is not None else ""
                )
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


# -------- Top-level legacy triple-quoted strings (non docstring) --------


def _gather_top_level_legacy_strings(
    mod: cst.Module,
    positions: metadata.PositionProvider,
    ls: List[int],
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Remove top-level string expressions that aren’t the module docstring.
    Return (delete_spans, n_blocks).
    """
    delete_spans: List[Tuple[int, int]] = []
    body = list(mod.body)
    doc0 = _is_top_level_docstring(body[0]) if body else False

    def _stmt_full_line_span(stmt: cst.SimpleStatementLine) -> Tuple[int, int]:
        r = positions[stmt]
        s = ls[r.start.line - 1]
        t = ls[r.end.line]
        return (s, t)

    for idx, stmt in enumerate(body):
        if not isinstance(stmt, cst.SimpleStatementLine) or not stmt.body:
            continue
        # Module docstring = the first simple string at the very beginning: keep it.
        if idx == 0 and doc0:
            continue
        expr = stmt.body[0]
        if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.SimpleString):
            delete_spans.append(_stmt_full_line_span(stmt))

    return delete_spans, len(delete_spans)


# -------- Top-level "path definitions" --------

_PATH_DEF_RX = re.compile(
    r"\b(os\.path|pathlib\.Path|__file__)\b|(^|\W)Path\s*\("  # Path(...) (often via from pathlib import Path)
)


def _scan_imports_flags(mod: cst.Module, code_for) -> dict:
    """
    Flags to make detecting ‘Path(’ safer when ‘from pathlib import Path’ is present.
    """
    has_from_pathlib_Path = False
    has_pathlib = False
    for stmt in mod.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        for small in stmt.body:
            if isinstance(small, cst.Import):
                for alias in small.names:
                    txt = code_for(alias.name).strip()
                    if txt == "pathlib":
                        has_pathlib = True
            elif isinstance(small, cst.ImportFrom):
                mtxt = code_for(small.module).strip() if small.module else ""
                if mtxt == "pathlib":
                    if isinstance(small.names, cst.ImportStar):
                        has_from_pathlib_Path = True  # prudent: * inclut Path
                    else:
                        names = [code_for(a.name).strip() for a in small.names]
                        if "Path" in names:
                            has_from_pathlib_Path = True
    return {"has_from_pathlib_Path": has_from_pathlib_Path, "has_pathlib": has_pathlib}


def _gather_path_def_edits(
    source: str,
    path: str,
    *,
    mod: cst.Module,
    positions: metadata.PositionProvider,
    ls: List[int],
) -> Tuple[List[Edit], List[str]]:
    """
    Detect top-level assignments whose RHS looks like a path definition.
    Return (edits_deletion, removed_var_names)
    """
    code_for = lambda node: _code_for(mod, node)
    flags = _scan_imports_flags(mod, code_for)

    def _stmt_line_span(stmt: cst.SimpleStatementLine) -> Tuple[int, int]:
        r = positions[stmt]
        s = ls[r.start.line - 1]
        t = ls[r.end.line]
        return (s, t)

    removed: List[str] = []
    deletions: List[Edit] = []

    def _value_matches(value: cst.CSTNode) -> bool:
        txt = code_for(value).strip()
        if not txt:
            return False
        if _PATH_DEF_RX.search(txt):
            # If we have Path(...), ensure Path really comes from pathlib (when possible).
            if re.search(r"(^|\W)Path\s*\(", txt) and not (
                flags["has_from_pathlib_Path"] or flags["has_pathlib"]
            ):
                # too aggressive if ‘Path’ hasn’t been imported.
                return False
            return True
        return False

    def _collect_names_from_assign(assign: cst.Assign) -> List[str]:
        out: List[str] = []
        for targ in getattr(assign, "targets", []) or []:
            t = getattr(targ, "target", None)
            if t is not None:
                out.append(code_for(t).strip())
        return out

    for stmt in mod.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue
        any_hit = False

        for small in stmt.body:
            if isinstance(small, cst.Assign):
                if _value_matches(small.value):
                    removed.extend(_collect_names_from_assign(small))
                    any_hit = True
            elif isinstance(small, cst.AnnAssign):
                if small.value and _value_matches(small.value):
                    name_txt = code_for(small.target).strip()
                    if name_txt:
                        removed.append(name_txt)
                    any_hit = True

        if any_hit:
            s, t = _stmt_line_span(stmt)
            deletions.append(Edit(path=path, span=(s, t), replacement=""))

    return deletions, removed


# -------- Top-level sys.path manipulations --------


def _gather_sys_path_edits(
    source: str,
    path: str,
    *,
    mod: cst.Module,
    positions: metadata.PositionProvider,
    ls: List[int],
) -> Tuple[List[Edit], List[str]]:
    """
    Remove:
      - sys.path.append/insert/extend(...)
      - Assign / AugAssign on sys.path
      - Return (edits_deletion, descriptions).
    """
    code_for = lambda node: _code_for(mod, node)

    def _stmt_line_span(stmt: cst.SimpleStatementLine) -> Tuple[int, int]:
        r = positions[stmt]
        s = ls[r.start.line - 1]
        t = ls[r.end.line]
        return (s, t)

    deletions: List[Edit] = []
    descs: List[str] = []

    def _is_sys_path_attr(node: cst.CSTNode) -> bool:
        if isinstance(node, cst.Attribute) and isinstance(node.value, cst.Attribute):
            v = node.value
            return (
                isinstance(v.value, cst.Name)
                and v.value.value == "sys"
                and isinstance(v.attr, cst.Name)
                and v.attr.value == "path"
            )
        if isinstance(node, cst.Attribute):
            # textual fallback
            return code_for(node).strip().startswith("sys.path")
        return False

    for stmt in mod.body:
        if not isinstance(stmt, cst.SimpleStatementLine):
            continue

        any_hit = False

        for small in stmt.body:
            # sys.path.append/insert/extend(...)
            if isinstance(small, cst.Expr) and isinstance(small.value, cst.Call):
                f = small.value.func
                if isinstance(f, cst.Attribute) and _is_sys_path_attr(f.value):
                    m = isinstance(f.attr, cst.Name) and f.attr.value in {
                        "append",
                        "insert",
                        "extend",
                    }
                    if m:
                        any_hit = True
                        descs.append(code_for(f).strip() + "(...)")
            # Assign sys.path = ...
            elif isinstance(small, cst.Assign):
                for targ in small.targets:
                    if _is_sys_path_attr(getattr(targ, "target", None)):
                        any_hit = True
                        descs.append("sys.path = …")
            # sys.path += [...]
            elif isinstance(small, cst.AugAssign):
                if _is_sys_path_attr(small.target):
                    any_hit = True
                    descs.append("sys.path " + code_for(small.operator).strip() + "= …")

        if any_hit:
            s, t = _stmt_line_span(stmt)
            deletions.append(Edit(path=path, span=(s, t), replacement=""))

    return deletions, descs


def _insert_markers_for_cleanup(
    source: str,
    path: str,
    *,
    mod: cst.Module,
    positions: metadata.PositionProvider,
    ls: List[int],
    imports_removed: List[str] | None = None,
    sys_path_descs: List[str] | None = None,
    path_vars_removed: List[str] | None = None,
    legacy_blocks_count: int = 0,
) -> List[Edit]:
    """
    Build markers at the same insertion point used for imports (after the shebang, docstring, futures, imports, then blank lines).
    The insertion order is deterministic.
    """
    inserts: List[Edit] = []
    insert_pos = _compute_marker_insert_pos_cst(source, mod, positions, ls)

    # 1) Imports – if provided (otherwise _drop_top_level_imports_with_marker_cst will do it).
    if imports_removed:
        txt = _format_imports_marker(imports_removed, mode="list") + "\n"
        inserts.append(Edit(path=path, span=(insert_pos, insert_pos), replacement=txt))

    def _fmt_list(tag: str, items: List[str], max_items: int = 4) -> str:
        items = sorted(dict.fromkeys(items))
        head = f"# [{tag}: {len(items)}]"
        if not items:
            return head + "\n"
        sample = ", ".join(items[:max_items])
        if len(items) > max_items:
            sample += f"…(+{len(items) - max_items})"
        return f"{head} {sample}\n"

    # 2) sys.path
    if sys_path_descs:
        txt = _fmt_list("sys.path tweaks omitted", sys_path_descs)
        inserts.append(Edit(path=path, span=(insert_pos, insert_pos), replacement=txt))

    # 3) path defs
    if path_vars_removed:
        txt = _fmt_list("path setup omitted", path_vars_removed)
        inserts.append(Edit(path=path, span=(insert_pos, insert_pos), replacement=txt))

    # 4) legacy strings
    if legacy_blocks_count:
        unit = "block" if legacy_blocks_count == 1 else "blocks"
        txt = f"# [legacy strings omitted: {legacy_blocks_count} {unit}]\n"
        inserts.append(Edit(path=path, span=(insert_pos, insert_pos), replacement=txt))

    return inserts


def _drop_top_level_imports_with_marker_cst(
    source: str,
    path: str,
    *,
    mod: cst.Module,
    positions: metadata.PositionProvider,
    ls: List[int],
) -> List[Edit]:
    delete_spans, removed_names = _gather_top_level_import_edits_and_names(
        mod, positions, ls
    )
    edits: List[Edit] = [
        Edit(path=path, span=span, replacement="") for span in delete_spans
    ]
    if not removed_names:
        return edits
    insert_pos = _compute_marker_insert_pos_cst(source, mod, positions, ls)
    marker = (
        _format_imports_marker(removed_names, mode="list", max_items=4, max_chars=120)
        + "\n"
    )
    edits.append(Edit(path=path, span=(insert_pos, insert_pos), replacement=marker))
    return edits


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

        qual_local = (
            ".".join(self.stack + [node.name.value]) if self.stack else node.name.value
        )
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


def _collect_funcs(
    text: str, path: Path
) -> Tuple[List[str], List[FuncInfo], cst.Module, metadata.PositionProvider]:
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


class PythonEngine:
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
        ls = line_starts(text)
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
            or action.kind
            in {
                "file:no-imports",
                "file:no-legacy-strings",
                "file:no-path-defs",
                "file:no-sys-path",
                "file:tidy",
            }
        )

    def render(self, doc: Document, decisions: List[Tuple[Node, Action]]) -> List[Edit]:
        """
        Interprets the decisions (node, action) into a list of Edits for a Python document.
        - Handles file-level actions:
            * file:no-imports
            * file:no-legacy-strings
            * file:no-path-defs
            * file:no-sys-path
            * file:tidy (composite of the three above + no-imports)
        - Handles node-level actions: hide / sig / sig+doc / body:levels=k
        """
        # --------- Retrieve the analysis produced by parse() (no re-parse) ---------
        analysis: Dict[str, object] = {}
        for r in doc.roots:
            if r.kind == PY_MODULE and (r.meta or {}).get("analysis"):
                analysis = r.meta["analysis"]  # type: ignore[assignment]
                break

        if analysis:
            src: List[str] = analysis.get("src_lines") or doc.text.splitlines(keepends=True)  # type: ignore[assignment]
            ls: List[int] = analysis.get("line_starts") or line_starts(doc.text)  # type: ignore[assignment]
            by_qual: Dict[str, FuncInfo] = analysis.get("funcs_by_qual") or {}  # type: ignore[assignment]
            cst_mod: Optional[cst.Module] = analysis.get("cst_module")  # type: ignore[assignment]
            posmap: Optional[metadata.PositionProvider] = analysis.get("posmap")  # type: ignore[assignment]
        else:
            # Defensive fallback: minimal re-parsing if analysis is not available.
            src, funcs, cst_mod, posmap = _collect_funcs(doc.text, Path(doc.path))
            ls = line_starts(doc.text)
            by_qual = {fi.qualname: fi for fi in funcs}

        candidates: List[Edit] = []

        # --------- Interpretation of decisions ---------
        for node, action in decisions:
            # =========================
            # 1) File-level filters
            # =========================
            if node.kind == PY_MODULE and action.kind == "file:no-imports":
                if cst_mod is not None and posmap is not None:
                    candidates.extend(
                        _drop_top_level_imports_with_marker_cst(
                            doc.text, doc.path, mod=cst_mod, positions=posmap, ls=ls
                        )
                    )
                continue

            if node.kind == PY_MODULE and action.kind in {
                "file:no-legacy-strings",
                "file:no-path-defs",
                "file:no-sys-path",
                "file:tidy",
            }:
                # If CST analysis is not available, do not attempt anything (best-effort).
                if cst_mod is None or posmap is None:
                    continue

                # 1) Imports (only for 'tidy'; otherwise leave the dedicated rule handle it)
                imports_del: List[Edit] = []
                imports_removed: List[str] = []
                if action.kind == "file:tidy":
                    spans, imports_removed = _gather_top_level_import_edits_and_names(
                        cst_mod, posmap, ls
                    )
                    imports_del = [
                        Edit(path=doc.path, span=sp, replacement="") for sp in spans
                    ]

                # 2) sys.path tweaks
                sys_del: List[Edit] = []
                sys_descs: List[str] = []
                if action.kind in {"file:no-sys-path", "file:tidy"}:
                    sys_del, sys_descs = _gather_sys_path_edits(
                        doc.text, doc.path, mod=cst_mod, positions=posmap, ls=ls
                    )

                # 3) Path defs
                path_del: List[Edit] = []
                path_vars: List[str] = []
                if action.kind in {"file:no-path-defs", "file:tidy"}:
                    path_del, path_vars = _gather_path_def_edits(
                        doc.text, doc.path, mod=cst_mod, positions=posmap, ls=ls
                    )

                # 4) Legacy triple-quoted strings (top-level, outside of module docstring)
                legacy_del_spans: List[Tuple[int, int]] = []
                legacy_count = 0
                if action.kind in {"file:no-legacy-strings", "file:tidy"}:
                    legacy_del_spans, legacy_count = _gather_top_level_legacy_strings(
                        cst_mod, posmap, ls
                    )
                legacy_del = [
                    Edit(path=doc.path, span=sp, replacement="")
                    for sp in legacy_del_spans
                ]

                # 5) Header markers (common and stable insertion point)
                markers = _insert_markers_for_cleanup(
                    doc.text,
                    doc.path,
                    mod=cst_mod,
                    positions=posmap,
                    ls=ls,
                    imports_removed=imports_removed
                    if action.kind == "file:tidy"
                    else None,
                    sys_path_descs=sys_descs if sys_del else None,
                    path_vars_removed=path_vars if path_del else None,
                    legacy_blocks_count=legacy_count if legacy_del else 0,
                )

                candidates.extend(imports_del)
                candidates.extend(sys_del)
                candidates.extend(path_del)
                candidates.extend(legacy_del)
                candidates.extend(markers)
                continue

            # =========================
            # 2) Actions on nodes (classes / functions / methods)
            # =========================
            mode = _mode_of_action(action)
            if not mode or mode == "full":
                continue

            # --- Entire module ---
            if node.kind == PY_MODULE and mode == "hide":
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

            # --- Classes ---
            if node.kind == PY_CLASS:
                if mode == "hide":
                    s, t = node.span
                    before = doc.text[:s]
                    omitted_text = doc.text[s:t]
                    start_line = before.count("\n") + 1
                    end_line = start_line + omitted_text.count("\n")
                    label = (
                        f"class {node.name} omitted" if node.name else "class omitted"
                    )
                    marker = make_omission_line(
                        "py",
                        start_line,
                        end_line,
                        indent="",
                        opts=DEFAULT_OPTIONS,
                        label=label,
                    )
                    candidates.append(
                        Edit(path=doc.path, span=node.span, replacement=marker)
                    )
                continue  # (other modes do not apply to the class block itself)

            # --- Functions / Methods ---
            if node.kind in {PY_FUNCTION, PY_METHOD}:
                fi = by_qual.get(node.qual or "")
                if not fi:
                    # Fallback if function index failed: properly hide
                    if mode == "hide":
                        s, t = node.span
                        before = doc.text[:s]
                        omitted_text = doc.text[s:t]
                        start_line = before.count("\n") + 1
                        end_line = start_line + omitted_text.count("\n")
                        marker = make_omission_line(
                            "py",
                            start_line,
                            end_line,
                            indent="",
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
                continue

            # Other Python nodes: nothing to do

        # No deduplication here: merge_edits() upstream takes care of the rest
        return candidates


def create_engine():
    return PythonEngine()
