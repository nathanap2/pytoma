import sys
import builtins
import pytest
from pathlib import Path


def test_missing_md_dependency_raises(tmp_path, monkeypatch):
    """
    Simulates the absence of 'markdown_it' by intercepting its import,
    while forcing a re-import of the Markdown engine.
    """
    from pytoma import core
    from pytoma.config import Config

    # 1) Create a .md file so the discovery process finds the extension
    (tmp_path / "doc.md").write_text("# T\n", encoding="utf-8")

    # 2) Purge the engine module from the cache to force a re-import
    monkeypatch.delitem(sys.modules, "pytoma.engines.markdown_engine", raising=False)

    # 3) Reset the global state of the engine loader
    core._ENGINES.clear()
    core._LOADED_EXTS.clear()
    core._ENGINE_LOAD_ERRORS.clear()

    # 4) Intercept the import of markdown_it BEFORE build_prompt
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "markdown_it" or name.startswith("markdown_it."):
            raise ImportError("markdown-it-py not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # 5) Execution: we expect a strict failure
    cfg = Config.load(None, "full")
    with pytest.raises(RuntimeError) as ei:
        core.build_prompt([tmp_path], cfg)

    msg = str(ei.value)
    assert "No engine available for discovered extension" in msg
    assert "md" in msg  # the list of missing extensions must include md
