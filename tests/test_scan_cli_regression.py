# tests/test_cli_scan_regression.py
import sys
import subprocess
from pathlib import Path

import pytest
import os
import sys

@pytest.fixture()
def tmp_git_repo(tmp_path: Path) -> Path:
    """
    Mini git repo with .py/.md/.toml, initial commit.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pkg").mkdir()
    (repo / "pkg" / "a.py").write_text("def f(x):\n    return x * 2\n", encoding="utf-8")
    (repo / "README.md").write_text("# Demo\n\nSome text.\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[tool.demo]\nname='demo'\n", encoding="utf-8")

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)
    return repo


def test_iter_files_accepts_path_and_str(tmp_git_repo: Path):
    """
    iter_files must accept Path and str and return the same selection.
    (Before fix: str failed/returned 0)
    """
    import pytoma.scan as s

    paths_from_path = list(s.iter_files([tmp_git_repo]))
    assert len(paths_from_path) > 0, "Sanity check: the repo should contain files"

    # same call passing a str
    paths_from_str = list(s.iter_files([str(tmp_git_repo)]))
    assert len(paths_from_str) == len(paths_from_path)


def _dump_debug(stderr: str):
    # handy to read in CI logs
    lines = [ln for ln in stderr.splitlines() if "[pytoma:debug]" in ln]
    print("\n".join(lines))  # pytest -s will display them

def test_cli_main_on_dot_should_emit_content(tmp_git_repo: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_git_repo)
    os.environ["PYTOMA_DEBUG"] = "1"

    import pytoma.cli as cli
    rc = cli.main(["."])
    captured = capsys.readouterr()
    out, err = captured.out, captured.err

    # if it fails, dump debug to understand why
    if "# (no files found)" in out.lower():
        _dump_debug(err)

    assert rc == 0
    assert out.strip(), "The CLI must emit some content"
    assert "(no files found)" not in out.lower()

def test_cli_main_with_explicit_git_ls_files(tmp_git_repo: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_git_repo)
    os.environ["PYTOMA_DEBUG"] = "1"

    rels = (
        subprocess.run(
            ["git", "ls-files", "*.py", "*.md", "*.toml"],
            cwd=tmp_git_repo, text=True, capture_output=True, check=True
        )
        .stdout.strip().splitlines()
    )
    assert rels, "git ls-files must return at least one file"

    import pytoma.cli as cli
    rc = cli.main(rels)
    captured = capsys.readouterr()
    out, err = captured.out, captured.err

    if "# (no files found)" in out.lower():
        _dump_debug(err)

    assert rc == 0
    assert out.strip()
    assert "(no files found)" not in out.lower()


def test_cli_main_no_files_prints_sentinel(tmp_path: Path, monkeypatch, capsys):
    """
    Edge case: empty directory -> must print the standard sentinel.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)

    import pytoma.cli as cli
    rc = cli.main(["."])
    out = capsys.readouterr().out

    assert rc == 0
    assert out.strip() == "# (no files found)"


def test_iter_files_with_explicit_includes(tmp_git_repo: Path):
    """
    iter_files with includes = ('**/*.py','**/*.md','**/*.toml') must see >= 3 files
    in the fake repo.
    """
    from pytoma.scan import iter_files
    files = list(iter_files([tmp_git_repo], includes=("**/*.py","**/*.md","**/*.toml")))
    assert len(files) >= 3, f"iter_files(includes=...) returned nothing? files={files[:3]}"




def test_build_prompt_on_minirepo(tmp_git_repo: Path):
    # Call build_prompt directly, with an explicit Config (default="full")
    from pytoma.core import Config, build_prompt
    from pytoma.scan import iter_files

    files = list(iter_files([tmp_git_repo], includes=("**/*.py","**/*.md","**/*.toml")))
    assert len(files) >= 3

    cfg = Config(default="full", rules=None, excludes=None)  # force "full"
    text = build_prompt(files, cfg)
    assert text.strip() and "(no files found)" not in text.lower()

def test_build_prompt_accepts_files_directly(tmp_git_repo: Path):
    from pytoma.core import Config, build_prompt
    cfg = Config(default="full", rules=None, excludes=None)
    files = [tmp_git_repo/"README.md", tmp_git_repo/"pyproject.toml", tmp_git_repo/"pkg/a.py"]
    text = build_prompt(files, cfg)
    assert text.strip() and "(no files found)" not in text.lower()

def test_build_prompt_expands_directories_only(tmp_git_repo: Path):
    from pytoma.core import Config, build_prompt
    cfg = Config(default="full", rules=None, excludes=None)
    text = build_prompt([tmp_git_repo], cfg)  # directory; should expand
    assert text.strip() and "(no files found)" not in text.lower()
    
# tests/test_build_prompt_regression.py

import pathlib
from pytoma.core import Config, build_prompt

def test_build_prompt_with_explicit_file(tmp_path: pathlib.Path):
    """
    Regression: build_prompt([]) on explicit files must not 
    return "# (no files found)" if at least one eligible file is provided.
    """

    # Arrange: create a minimal .py
    f = tmp_path / "hello.py"
    f.write_text("def f(x):\n    return x + 1\n")

    cfg = Config(default="full", rules=None, excludes=None)

    # Act
    text = build_prompt([f], cfg)

    # Assert
    assert text.strip(), "The prompt must not be empty"
    assert "(no files found)" not in text.lower(), "Must never print (no files found) when a direct file was provided"
    assert "def f" in text, "The file's content must be present in the prompt"

