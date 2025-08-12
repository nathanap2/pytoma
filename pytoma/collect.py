from dataclasses import dataclass
from typing import Optional, Tuple, List
import pathlib
import libcst as cst
from libcst import metadata

@dataclass
class FuncInfo:
    """
    Collected information about a single Python function or method.
    The line numbers are 1-based and inclusive where specified.
    """
    module: str
    qualname: str
    start: Tuple[int,int]
    end: Tuple[int,int]
    node: cst.FunctionDef
    docstring: Optional[str]
    doc_range: Optional[Tuple[int,int]]   # [start_line, end_line]
    body_first_line: Optional[int]
    deco_start_line: int


class FuncCollector(cst.CSTVisitor):
    def __init__(self, module_name: str, source_lines: List[str], posmap) -> None:
        self.module_name = module_name
        self.source_lines = source_lines
        self.posmap = posmap
        self.class_stack: List[str] = []
        self.func_stack: List[str] = []
        self.funcs: List[FuncInfo] = []

    # --- class stack ---
    def visit_ClassDef(self, n: cst.ClassDef) -> None:
        self.class_stack.append(n.name.value)

    def leave_ClassDef(self, n: cst.ClassDef) -> None:
        self.class_stack.pop()

    # --- function stack + collection ---
    def visit_FunctionDef(self, fn: cst.FunctionDef) -> None:
        self.func_stack.append(fn.name.value)
        p = self.posmap[fn]
        start = (p.start.line, p.start.column)
        end   = (p.end.line,   p.end.column)

        if fn.decorators:
            deco_start_line = min(self.posmap[d].start.line for d in fn.decorators)
        else:
            deco_start_line = start[0]

        # detect docstring + body first line
        docstring = None
        doc_range = None
        body_first_line = None
        body_elems = getattr(fn.body, "body", None) or []
        if body_elems:
            first_stmt = body_elems[0]
            p0 = self.posmap[first_stmt]
            body_first_line = p0.start.line

            if (
                isinstance(first_stmt, cst.SimpleStatementLine)
                and first_stmt.body
                and isinstance(first_stmt.body[0], cst.Expr)
                and isinstance(first_stmt.body[0].value, cst.SimpleString)
            ):
                ds = self.posmap[first_stmt]
                ds_start, ds_end = ds.start.line, ds.end.line
                doc_range = (ds_start, ds_end)
                # exact slice from original source (keeps quotes intact)
                docstring = "".join(self.source_lines[ds_start-1:ds_end])

        parts = self.class_stack + self.func_stack
        local = ".".join(parts) if parts else fn.name.value  # hierarchical qualname
        qual  = f"{self.module_name}:{local}"

        self.funcs.append(
            FuncInfo(
                module=self.module_name,
                qualname=qual,
                start=start,
                end=end,
                node=fn,
                docstring=docstring,
                doc_range=doc_range,
                body_first_line=body_first_line,
                deco_start_line=deco_start_line,
            )
        )

    def leave_FunctionDef(self, fn: cst.FunctionDef) -> None:
        self.func_stack.pop()

def file_to_module_name(path: pathlib.Path, root: pathlib.Path) -> str:
    """
    Convert a filesystem path to a dotted Python module name relative to 'root'.
    Examples:
      root=/repo, path=/repo/pkg/mod.py      -> "pkg.mod"
      root=/repo, path=/repo/pkg/__init__.py -> "pkg"
    If 'path' is not under 'root', derive the module from the path itself.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(p for p in parts if p)

