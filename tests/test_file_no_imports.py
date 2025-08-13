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
            core.Rule(match=f"{abs_root}/*.py", mode="file:no-imports"),
            core.Rule(match=f"{abs_root}/**/*.py", mode="file:no-imports"),
        ],
        excludes=[],
    )

    out = core.build_prompt([pkg], cfg)

    # The pack contains a header and a Python fence
    assert "### " in out and "```python" in out

    # ---- Top-level import statements have disappeared (regex anchored) ----
    assert not re.search(r"(?m)^\s*import\s+os\b", out)
    assert not re.search(r"(?m)^\s*from\s+math\s+import\s+sqrt\b", out)
    assert not re.search(r"(?m)^\s*from\s+itertools\s+import\b", out)

    # __future__ is preserved
    assert "from __future__ import annotations" in out

    # ---- Omission marker is present and informative ----
    assert "# [imports omitted:" in out
    # we do not assume ordering, only that key elements are present
    assert "math.sqrt" in out
    assert "itertools.chain" in out
    assert "itertools.groupby" in out
    # (Optional) check exactly 4 omitted items in this case
    assert "[imports omitted: 4]" in out or "# [imports omitted: 4]" in out

    # ---- Marker placement: after __future__, before the first def ----
    pos_future = out.find("from __future__ import annotations")
    pos_marker = out.find("# [imports omitted:")
    pos_def = out.find("def f(x):")
    assert pos_future != -1 and pos_marker != -1 and pos_def != -1
    assert pos_future < pos_marker < pos_def

    # Intra-function import is preserved
    assert "import sys  # inside-function" in out

    # Function body is present (the 'os' reference remains in the text)
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

    abs_b = b_py.resolve().as_posix()
    cfg = core.Config(
        default="full",
        rules=[
            core.Rule(match=abs_b, mode="file:no-imports"),
            core.Rule(match="b:g", mode="sig"),
        ],
        excludes=[],
    )

    out = core.build_prompt([pkg], cfg)

    # 1) Top-level imports have disappeared (regex anchored at line start)
    assert not re.search(r"(?m)^\s*import\s+os\b", out)
    assert not re.search(r"(?m)^\s*from\s+math\s+import\s+sqrt\b", out)

    # 2) Import omission marker: appears only once and is informative
    assert out.count("# [imports omitted:") == 1
    assert "os" in out and "math.sqrt" in out

    # 3) Placement: marker appears before the first definitions
    pos_marker = out.find("# [imports omitted:")
    pos_def_f = out.find("def f(x):")
    pos_def_g = out.find("def g(y):")
    assert pos_marker != -1 and pos_def_f != -1 and pos_def_g != -1
    assert pos_marker < min(pos_def_f, pos_def_g)

    # 4) f remains in "full"
    assert "def f(x):" in out
    assert 'return os.getenv("HOME")' in out

    # 5) g is contracted to signature + "body omitted" marker, exactly once
    assert len(re.findall(r"(?m)^def\s+g\(y\):", out)) == 1
    m = re.search(r"(?s)^def\s+g\(y\):\s*\n([ \t]*# .+)$", out, flags=re.M)
    assert m, "Expected an omission marker comment right after 'def g(y):'"
    assert "body omitted" in m.group(1)

    # 6) g's original body is no longer present
    assert "return sqrt(y) + 1" not in out
