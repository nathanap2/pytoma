import importlib
import pathlib
import textwrap
import re
import pytest

def _write(p: pathlib.Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(s).lstrip("\n"), encoding="utf-8")


def test_pre_resolve_reorders_literal_hide_before_glob_file():
    """
    Problematic case: a 'hide' rule targeting a specific file
    must come before a glob 'file:no-imports' that covers that file.
    """
    core = importlib.import_module("pytoma.core")

    # Intentionally "wrong" order config
    cfg = core.Config(
        default="full",
        rules=[
            core.Rule(match="pytoma/**/*.py", mode="file:no-imports"),
            core.Rule(match="pytoma/engines/toml_min.py", mode="hide"),
        ],
        excludes=[],
    )

    # Must exist on the core side (per the proposed patch)
    assert hasattr(core, "pre_resolve_path_rules"), "core.pre_resolve_path_rules is missing"
    cfg2, warns = core.pre_resolve_path_rules(cfg)

    # In cfg2, 'hide' (literal) must precede 'file:no-imports' (glob)
    rules_str = [(r.match, r.mode) for r in (cfg2.rules or [])]
    pos_hide = rules_str.index(("pytoma/engines/toml_min.py", "hide"))
    pos_glob = rules_str.index(("pytoma/**/*.py", "file:no-imports"))
    assert pos_hide < pos_glob, f"Expected literal 'hide' before glob 'file:no-imports'; got {rules_str}"

    # Optional: a useful warning may be emitted
    # (don't make the test brittle on this)
    assert isinstance(warns, list)


def test_end_to_end_no_overlap_after_preresolve(tmp_path: pathlib.Path):
    """
    Reduced e2e repro: mini-repo with:
      - pytoma/engines/toml_min.py (must be 'hide')
      - pytoma/**.py (glob 'file:no-imports')
      - tests/t.py (no-imports + sig+doc)
      - pyproject.toml with [project.urls] to hide
    We check: no exception and correct markers in the pack.
    """
    core = importlib.import_module("pytoma.core")

    root = tmp_path / "proj"
    # --- sources ---
    toml_min = root / "pytoma" / "engines" / "toml_min.py"
    _write(toml_min, """
        import os
        from math import sqrt

        def impl(x):
            # tiny body
            return sqrt(x) + (os.name == "posix")
    """)

    tests_py = root / "tests" / "t.py"
    _write(tests_py, """
        import os
        from math import sqrt

        def g(y):
            return sqrt(y) + 1
    """)

    pyproject = root / "pyproject.toml"
    _write(pyproject, """
        [project]
        name = "demo"
        version = "0.1.0"

        [project.urls]
        homepage = "https://example.com"
        source = "https://example.com/src"
    """)

    # --- config (intentionally problematic order) ---
    cfg = core.Config(
        default="full",
        rules=[
            core.Rule(match="**/tests/**.py", mode="file:no-imports"),
            core.Rule(match="**/tests/**.py", mode="sig+doc"),
            core.Rule(match="pytoma/**/*.py", mode="file:no-imports"),
            core.Rule(match="pyproject.toml", mode="full"),
            core.Rule(match="pyproject.toml:project.urls", mode="hide"),
            core.Rule(match="pytoma/engines/toml_min.py", mode="hide"),
        ],
        excludes=[],
    )

    # Pre-resolution: produces a re-ordered cfg
    assert hasattr(core, "pre_resolve_path_rules"), "core.pre_resolve_path_rules is missing"
    cfg2, _ = core.pre_resolve_path_rules(cfg)

    # Build: MUST NOT raise (overlaps resolved)
    pack = core.build_prompt([root], cfg2)

    # --- Assertions on the rendered pack ---

    # 1) The literal-targeted file is hidden by a single module marker
    assert "pytoma/engines/toml_min.py" in pack
    assert "module omitted" in pack
    # And top-level imports should no longer appear
    assert "from math import sqrt" not in pack
    assert re.search(r"\bdef\s+impl\(", pack) is None  # entire module hidden

    # 2) Test file: imports removed + function rendered as sig+doc (placeholder doc + body omitted)
    # Section present
    assert "tests/t.py" in pack
    # Top-level imports removed
    assert "import os\n" not in pack
    assert "from math import sqrt" not in pack
    # Signature + placeholder + body omitted
    assert re.search(r"def\s+g\(\s*y\s*\):", pack)
    assert '"""…"""' in pack or '"""...' in pack  # depending on ascii_only
    assert "body omitted" in pack
    # Original body not present
    assert "return sqrt(y) + 1" not in pack

    # 3) TOML: [project.urls] hidden, [project] visible
    assert "[project]" in pack
    assert "table [project.urls] omitted" in pack

