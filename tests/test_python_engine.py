import textwrap
from pathlib import Path

import pytest

from pytoma.engines.python_engine import PythonMinEngine
from pytoma.policies import Action
from pytoma.ir import PY_MODULE, PY_CLASS, PY_FUNCTION, PY_METHOD
from pytoma.edits import merge_edits, _apply_edits_to_text


def _apply(engine, tmp_path: Path, src: str, decisions):
    """
    Helper:
      - configure engine with tmp root,
      - parse synthetic path + src,
      - render + merge edits,
      - apply edits to src and return the new text.
    """
    engine.configure([tmp_path])
    path = tmp_path / "mod.py"
    doc = engine.parse(path, src)
    edits = engine.render(doc, decisions)
    merged = merge_edits(edits)
    # Apply only this doc's edits to the original text
    file_edits = [e for e in merged if Path(e.path).as_posix() == doc.path.as_posix()]
    return _apply_edits_to_text(src, file_edits), doc


def test_file_no_imports_removes_top_level_and_inserts_marker(tmp_path: Path):
    engine = PythonMinEngine()
    src = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        \"\"\"Module doc.\"\"\"
        from __future__ import annotations

        # comment between
        import os
        from .sub import y as z
        from pkg import *
        
        def f():
            return 42
        """
    )

    # Decision: file-level filter on the module
    # We fetch the module node after parse via helper
    # Build a temporary decisions list once we have the doc
    tmp_engine = PythonMinEngine()
    tmp_engine.configure([tmp_path])
    doc0 = tmp_engine.parse(tmp_path / "mod.py", src)
    mod = next(n for n in doc0.nodes if n.kind == PY_MODULE)
    decisions = [(mod, Action("file:no-imports"))]

    new_text, _ = _apply(engine, tmp_path, src, decisions)

    # __future__ remains, other imports are gone, and a marker is inserted
    assert "from __future__ import annotations" in new_text
    assert "import os" not in new_text
    assert "from pkg import *" not in new_text
    assert "from .sub import y as z" not in new_text
    assert "[imports omitted:" in new_text
    # Code still present
    assert "def f():" in new_text


def test_hide_class_replaces_block_with_marker(tmp_path: Path):
    engine = PythonMinEngine()
    src = textwrap.dedent(
        """\
        @decorator
        class C:
            def m(self):
                return 1

        def g():
            return 2
        """
    )
    # Parse to find the class node
    engine.configure([tmp_path])
    doc = engine.parse(tmp_path / "mod.py", src)
    cls = next(n for n in doc.nodes if n.kind == PY_CLASS and n.name == "C")
    decisions = [(cls, Action("hide"))]

    new_text, _ = _apply(engine, tmp_path, src, decisions)

    assert "class C omitted" in new_text
    assert "def m(" not in new_text
    # Function g() should remain untouched
    assert "def g():" in new_text


def test_sigdoc_function_keeps_header_and_doc_omits_body(tmp_path: Path):
    engine = PythonMinEngine()
    src = textwrap.dedent(
        """\
        def foo(a, b):
            \"\"\"This is the docstring.\"\"\"
            x = 1
            return x
        """
    )
    engine.configure([tmp_path])
    doc = engine.parse(tmp_path / "mod.py", src)
    fn = next(
        n for n in doc.nodes if n.kind in {PY_FUNCTION, PY_METHOD} and n.name == "foo"
    )
    decisions = [(fn, Action("sig+doc"))]

    new_text, _ = _apply(engine, tmp_path, src, decisions)

    assert "def foo(a, b):" in new_text
    assert "This is the docstring." in new_text
    assert "return x" not in new_text
    assert "body omitted" in new_text  # marker label


def test_levels_k1_omits_deeper_indentation(tmp_path: Path):
    engine = PythonMinEngine()
    src = textwrap.dedent(
        """\
        def h():
            a = 1
            if True:
                b = 2
                if True:
                    c = 3
            return a
        """
    )
    engine.configure([tmp_path])
    doc = engine.parse(tmp_path / "mod.py", src)
    fn = next(
        n for n in doc.nodes if n.kind in {PY_FUNCTION, PY_METHOD} and n.name == "h"
    )
    decisions = [(fn, Action("levels", {"k": 1}))]

    new_text, _ = _apply(engine, tmp_path, src, decisions)

    # Deepest line 'c = 3' should be omitted; outer lines remain
    assert "a = 1" in new_text
    assert "if True:" in new_text
    assert "b = 2" in new_text
    assert "c = 3" not in new_text
    assert "omitted" in new_text  # omission marker inserted
