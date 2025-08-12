import os
import importlib
import textwrap
import pathlib

import libcst as cst
from libcst import metadata
import pytest

import re

import textwrap

def _write(p, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(s), encoding="utf-8")



PKG = "pytoma"

collect = importlib.import_module(f"{PKG}.collect")
render = importlib.import_module(f"{PKG}.render")
core = importlib.import_module(f"{PKG}.core")


def _collect_funcs(src: str, modname: str = "m"):
    """Parse source code, collect FuncInfo objects, and return (lines, funcs)."""
    src = textwrap.dedent(src)
    module = cst.parse_module(src)
    wrapper = metadata.MetadataWrapper(module)
    posmap = wrapper.resolve(metadata.PositionProvider)  # <-- new
    lines = src.splitlines(keepends=True)
    coll = collect.FuncCollector(modname, lines, posmap)  # <-- pass posmap
    wrapper.visit(coll)
    return lines, coll.funcs



# -------------------------------------------------------------------
# Unit tests for apply_destructive (render.py)
# -------------------------------------------------------------------

def test_preserves_class_context_with_sig():
    """
    The method stays in its class context, the signature is kept on a single line,
    and a "body omitted" marker is inserted in place of the body.
    """
    src = """
    class C:
        def m(self, x):
            y = x + 1
            return y
    """
    lines, funcs = _collect_funcs(src)
    def choose_mode(q):  # contract only m
        return "sig" if q.endswith(":C.m") else "full"
    out = render.apply_destructive(lines, funcs, choose_mode)
    assert "class C:" in out
    assert "def m(self, x):" in out
    # We no longer expect "    ..."; we expect an omission comment
    assert "body omitted" in out


def test_sigdoc_inserts_placeholder_when_no_docstring():
    """
    For sig+doc, if there is no original docstring, insert a placeholder plus an omission marker.
    """
    src = """
    def f(a, b):
        return a + b
    """
    lines, funcs = _collect_funcs(src)
    def choose_mode(q): return "sig+doc"
    out = render.apply_destructive(lines, funcs, choose_mode)
    assert "def f(a, b):" in out
    # placeholder added
    assert '"""…"""' in out
    # no more "..." — we want the omission marker
    assert "body omitted" in out



def test_levels_keeps_shallow_and_marks_omissions():
    code = '''
        def g(n):
            # top-level statement kept
            if n > 0:
                total = 0
                for i in range(n):
                    total += i
                return total
            return 0
        '''
    lines, funcs = _collect_funcs(code, modname="m")
    def choose(q):  # "m:g"
        return "body:levels=0" if q.endswith(":g") else "full"

    new_code = render.apply_destructive(lines, funcs, choose)

    # Deep if/for should be omitted with a marker
    assert "# … lines " in new_code
    # Top-level statements (comment and final return) remain visible
    assert "return 0" in new_code
    assert "for i in range(n):" not in new_code


def test_async_function_with_sig():
    """
    Async functions in 'sig' also receive an omission marker.
    """
    src = """
    import asyncio

    async def fetch(x):
        await asyncio.sleep(0.01)
        return x
    """
    lines, funcs = _collect_funcs(src)
    def choose_mode(q): return "sig"
    out = render.apply_destructive(lines, funcs, choose_mode)
    assert "async def fetch(x):" in out
    assert "body omitted" in out



# -------------------------------------------------------------------
# Minimal end-to-end with build_prompt (core.py)
# -------------------------------------------------------------------

def test_build_prompt_end_to_end(tmp_path: pathlib.Path):
    """
    Small e2e test: module, class+method in 'sig', free function in 'sig+doc'.
    Verifies that omission markers replace '...'.
    """
    root = tmp_path / "root"
    pkg = root / "pkg"
    a_py = pkg / "a.py"
    _write(
        a_py,
        """
        import uuid

        class C:
            def m(self, x):
                z = x * 2
                return z

        def free_func(y):
            return y + 1
        """
    )

    core = importlib.import_module(f"{PKG}.core")
    cfg = core.Config(
        default="full",
        rules=[
            core.Rule(match=str(a_py.resolve().as_posix()), mode="full"),
            core.Rule(match="a:C.m", mode="sig"),
            core.Rule(match="a:free_func", mode="sig+doc"),
        ],
        excludes=[],
    )
    out = core.build_prompt([pkg], cfg)

    assert "### " in out and "```python" in out
    assert "class C:" in out
    assert "def m(self, x):" in out
    assert "def free_func(y):" in out

    # No more "..." body: we expect an omission comment
    assert "body omitted" in out
    # The original body must not appear
    assert "return y + 1" not in out



# -------------------------------------------------------------------
# Additional edge cases (optional)
# -------------------------------------------------------------------

def test_multiple_functions_only_one_contracted():
    """
    Only 'a' is contracted; 'b' remains in full.
    """
    src = """
    def a():
        return 1

    def b():
        return 2
    """
    lines, funcs = _collect_funcs(src)
    def choose_mode(q):  # contract only a
        return "sig" if q.endswith(":a") else "full"
    out = render.apply_destructive(lines, funcs, choose_mode)

    assert "def a():" in out
    assert "body omitted" in out  # instead of "    ..."
    assert "def b():" in out
    assert "return 2" in out
    # and the original body of a is no longer there
    assert "return 1" not in out

