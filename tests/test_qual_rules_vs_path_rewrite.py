import importlib
import textwrap


def test_qual_rule_on_python_module_not_rewritten_as_path(tmp_path):
    """
    Regression guard:

    We build a tiny package 'pkg' with 'mod.py' defining f().
    We configure a QUALNAME rule: match="pkg.mod:*", mode="sig+doc".
    build_prompt() should render the function's signature + docstring and omit the body.

    This test ensures the left-hand side of a Python qualname ("pkg.mod") is NOT
    rewritten as if it were a filesystem path. If such rewriting happened, the
    rule "pkg.mod:*" would stop matching and the body would remain visible.
    """
    # --- Arrange: minimal repo tree ---
    repo = tmp_path
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "mod.py").write_text(
        textwrap.dedent(
            '''
            def f(x: int) -> int:
                """Tiny doc."""
                return x * 2
            '''
        ).lstrip("\n"),
        encoding="utf-8",
    )

    # --- Import project components ---
    core = importlib.import_module("pytoma.core")
    config_mod = importlib.import_module("pytoma.config")

    # QUALNAME rule (left side is a Python *module name*, not a filesystem path)
    cfg = config_mod.Config(
        default="full",
        rules=[config_mod.Rule(match="pkg.mod:*", mode="sig+doc")],
        excludes=[],
    )

    # --- Act: build the pack ---
    out = core.build_prompt([repo], cfg)

    # --- Assert: signature + docstring, body omitted ---
    assert "def f(x: int) -> int:" in out        # signature present
    assert '"""Tiny doc."""' in out              # docstring present
    assert "body omitted" in out                 # omission marker present
    assert "return x * 2" not in out             # body is not present


def test_qual_rule_should_be_order_invariant_even_with_deeper_root_first(tmp_path):
    """
    Order-dependent bug reproducer (expected to FAIL before the fix, pass after):

    If we pass a *deeper* root (repo/pkg) BEFORE the repository root (repo),
    module resolution may yield 'mod' instead of 'pkg.mod'. In that case, the
    rule "pkg.mod:*" no longer matches and the body is kept.

    Desired behavior: the order of roots must NOT affect qualname matching.
    """
    repo = tmp_path
    (repo / "pkg").mkdir()
    (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg" / "mod.py").write_text(
        textwrap.dedent(
            """
            def f(x: int) -> int:
                \"\"\"Tiny doc.\"\"\"
                return x * 2
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )

    core = importlib.import_module("pytoma.core")
    config_mod = importlib.import_module("pytoma.config")

    cfg = config_mod.Config(
        default="full",
        rules=[config_mod.Rule(match="pkg.mod:*", mode="sig+doc")],
        excludes=[],
    )

    # Intentionally pass the deeper root *first*, then the repo root.
    out = core.build_prompt([repo / "pkg", repo], cfg)

    # Expected: still apply 'sig+doc' (order-invariant)
    assert "def f(x: int) -> int:" in out
    assert '"""Tiny doc."""' in out
    assert "body omitted" in out
    assert "return x * 2" not in out

