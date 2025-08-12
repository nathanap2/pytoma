# tests/test_file_no_imports.py
import importlib
import textwrap
import pathlib
import re


def _write(p, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(s), encoding="utf-8")


def test_file_no_imports_basic(tmp_path):
    core = importlib.import_module("pytoma.core")

    pkg = tmp_path / "pkg"
    a_py = pkg / "a.py"
    _write(
        a_py,
        """
        from __future__ import annotations
        import os
        from math import sqrt
        from itertools import (
            chain,
            groupby,
        )

        def f(x):
            import sys  # inside-function: must remain
            return os.listdir(".")  # 'os' will be visible in the prompt even if the import is removed
        """,
    )

    abs_root = pkg.resolve().as_posix()
    cfg = core.Config(
        default="full",
        rules=[
            core.Rule(
                match=f"{abs_root}/*.py", mode="file:no-imports"
            ),  # files directly under pkg
            core.Rule(
                match=f"{abs_root}/**/*.py", mode="file:no-imports"
            ),  # and subfolders
        ],
        excludes=[],
    )

    out = core.build_prompt([pkg], cfg)

    # The pack contains a section '### <path>' followed by a ```python fence
    assert "### " in out and "```python" in out

    # Top-level imports removed
    assert "import os\n" not in out
    assert "from math import sqrt" not in out
    assert "from itertools import" not in out
    assert "chain" not in out and "groupby" not in out  # multi-line removal works

    # __future__ kept
    assert "from __future__ import annotations" in out

    # Inside-function import kept
    assert "import sys  # inside-function" in out

    # Function body present
    assert "def f(x):" in out
    assert "return os.listdir" in out


def test_file_no_imports_composes_with_sig_rule(tmp_path):
    core = importlib.import_module("pytoma.core")

    pkg = tmp_path / "pkg"
    b_py = pkg / "b.py"
    _write(
        b_py,
        """
        import os
        from math import sqrt

        def f(x):
            return os.getenv("HOME")

        def g(y):
            return sqrt(y) + 1
        """,
    )

    # 1) absolute rule on the file -> no-imports
    abs_b = b_py.resolve().as_posix()
    # 2) rule by qualname -> signature for g
    #    The module will be 'b' if we pass 'pkg' as root to build_prompt.
    cfg = core.Config(
        default="full",
        rules=[
            core.Rule(match=abs_b, mode="file:no-imports"),
            core.Rule(match="b:g", mode="sig"),
        ],
        excludes=[],
    )

    out = core.build_prompt([pkg], cfg)

    # Top-level imports removed
    assert "import os\n" not in out
    assert "from math import sqrt" not in out

    # f stays in "full"
    assert "def f(x):" in out
    assert 'return os.getenv("HOME")' in out

    # g is contracted to signature + omission marker for the body
    assert "def g(y):" in out

    # Right after the signature, we expect a commented line containing "body omitted"
    m = re.search(r"def\s+g\(y\):\s*\n([ \t]*# .+)", out)
    assert m, "Expected an omission marker comment after 'def g(y):'"
    assert "body omitted" in m.group(1)

    # The original body must no longer appear
    assert "return sqrt(y) + 1" not in out
