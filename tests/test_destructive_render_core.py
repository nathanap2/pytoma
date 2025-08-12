import os
import importlib
import textwrap
import pathlib

import libcst as cst
from libcst import metadata
import pytest

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
    code = '''
        class _Base:
            extension = ""
            def m(self, x):
                """Doc."""
                if x:
                    return 1
                return 0
        '''
    lines, funcs = _collect_funcs(code, modname="a")
    # Apply 'sig' only to a:_Base.m
    def choose(q):  # qualname = "a:_Base.m"
        return "sig" if q.endswith(":_Base.m") else "full"

    new_code = render.apply_destructive(lines, funcs, choose)

    # Class context is preserved
    assert "class _Base" in new_code
    # Detailed body is removed
    assert "return 1" not in new_code
    # Minimal body inserted
    assert "def m(self, x):" in new_code and "..." in new_code


def test_sigdoc_inserts_placeholder_when_no_docstring():
    code = '''
        def f(a, b):
            x = a + b
            return x
        '''
    lines, funcs = _collect_funcs(code, modname="m")
    def choose(q):  # "m:f"
        return "sig+doc" if q.endswith(":f") else "full"

    new_code = render.apply_destructive(lines, funcs, choose)

    # Docstring placeholder + minimal body
    assert '"""…"""' in new_code
    assert "def f(a, b):" in new_code and "..." in new_code
    # Original implementation removed
    assert "x = a + b" not in new_code


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
    code = '''
        import asyncio

        async def fetch(x):
            await asyncio.sleep(0)
            return x
        '''
    lines, funcs = _collect_funcs(code, modname="m")
    def choose(q):  # "m:fetch"
        return "sig" if q.endswith(":fetch") else "full"

    new_code = render.apply_destructive(lines, funcs, choose)
    # Async signature preserved
    assert "async def fetch(" in new_code
    # Minimal body
    assert "..." in new_code
    assert "await asyncio.sleep" not in new_code


# -------------------------------------------------------------------
# Minimal end-to-end with build_prompt (core.py)
# -------------------------------------------------------------------

def test_build_prompt_end_to_end(tmp_path: pathlib.Path):
    # Tree: root/pkg/a.py ; passed root = pkg -> module name "a"
    root = tmp_path / "root"
    pkg = root / "pkg"
    pkg.mkdir(parents=True)
    a_py = pkg / "a.py"
    a_py.write_text(textwrap.dedent('''
        import uuid

        class _Base:
            def m(self, x):
                if x:
                    return 1
                return 0

        def free_func(y):
            return y + 1
        '''), encoding="utf-8")

    cfg = core.Config(
        default="full",
        rules=[
            core.Rule(match="a:_Base.m", mode="sig"),
            core.Rule(match="a:free_func", mode="sig+doc"),
        ],
        excludes=core.Config.load(None).excludes,
    )

    md = core.build_prompt([pkg], cfg, verbose=True)

    # Markdown must contain a block for a.py
    assert str(a_py) in md or a_py.name in md
    # Class is preserved, but the method is contracted
    assert "class _Base" in md
    assert "def m(self, x):" in md and "..." in md
    assert "return 1" not in md  # body removed
    # The free function has a docstring placeholder
    assert "def free_func(y):" in md and '"""…"""' in md


# -------------------------------------------------------------------
# Additional edge cases (optional)
# -------------------------------------------------------------------

def test_multiple_functions_only_one_contracted():
    code = '''
        def a():
            return 1

        def b():
            return 2
        '''
    lines, funcs = _collect_funcs(code, modname="m")
    def choose(q):
        return "sig" if q.endswith(":a") else "full"

    new_code = render.apply_destructive(lines, funcs, choose)
    assert "def a():" in new_code and "..." in new_code
    assert "def b():" in new_code and "return 2" in new_code

