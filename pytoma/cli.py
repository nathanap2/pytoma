from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List


from .log import debug
from .config import Config
from .core import build_prompt, _display_path, get_engine_for, _ensure_engine_loaded_for
from .scan import iter_files


def _get_version_string() -> str:
    """
    Resolve the installed package version (from importlib.metadata).
    Falls back to a sentinel if unavailable.
    """
    try:
        from importlib.metadata import version as _v  # Python >= 3.8
    except Exception:  # pragma: no cover
        return "pytoma 0+unknown"
    try:
        return f"pytoma {_v('pytoma')}"
    except Exception:
        return "pytoma 0+unknown"


def _as_paths(paths: Iterable[Path | str]) -> List[Path]:
    """Normalize a sequence of path-like values to Path objects."""
    out: List[Path] = []
    for p in paths:
        out.append(p if isinstance(p, Path) else Path(p))
    return out


def _load_engines_for(exts: Iterable[str]) -> None:
    """
    Ensure engines are loaded for the given set of file extensions.
    Useful when --scan requests to display which engine would be used.
    """
    seen: set[str] = set()
    for e in exts:
        if not e:
            continue
        ext = e.lstrip(".").lower()
        if ext and ext not in seen:
            seen.add(ext)
            _ensure_engine_loaded_for(ext)


def _run_scan(
    roots: List[Path],
    *,
    cfg: Config,
    show_abs: bool,
    show_engine: bool,
) -> int:
    """
    Discovery-only, but mirrors core.build_prompt:
      - if an input is a directory: iter_files(dir, includes=py/md/toml, excludes=cfg.excludes)
      - if an input is a file: include it directly and use its parent as a root
      - deduplicate + deterministic sort
      - same display path logic relative to the deepest matching root
    """
    # 1) Normalise inputs
    inputs = [p.resolve() for p in roots]

    # 2) Discover files exactly like core
    display_roots: List[Path] = []
    discovered: List[Path] = []
    for p in inputs:
        if p.is_dir():
            display_roots.append(p)
            for f in iter_files(
                [p],
                includes=("**/*.py", "**/*.md", "**/*.toml"),
                excludes=(cfg.excludes or []),
            ):
                discovered.append(f.resolve())
        elif p.is_file():
            display_roots.append(p.parent.resolve())
            discovered.append(p.resolve())
        else:
            # nonexistent path: silently ignore (same policy as core)
            continue

    # 3) Dedup + deterministic sort
    dedup = {q.as_posix(): q for q in discovered}
    files = sorted(dedup.values(), key=lambda q: q.as_posix())

    # 4) Optionally preload engines to show engine names
    if show_engine:
        _load_engines_for(p.suffix for p in files)

    # 5) Output
    if not files:
        print("(no files found)")
        return 0

    for p in files:
        disp = p.as_posix() if show_abs else _display_path(p, display_roots)
        if show_engine:
            eng = get_engine_for(p)
            tag = getattr(eng, "__class__", type(eng)).__name__ if eng else "-"
            print(f"{disp}\t[{tag}]")
        else:
            print(disp)

    print(f"# total: {len(files)}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pytoma — render a repo into one LLM-ready text file"
    )
    ap.add_argument("-V", "--version", action="version", version=_get_version_string())

    # Verbosity / debugging (enables internal debug logs)
    ap.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (sets PYTOMA_DEBUG). Repeat for more logs.",
    )

    # Discovery-only mode (no rendering)
    ap.add_argument(
        "--scan", action="store_true", help="List discovered files (no rendering)."
    )
    ap.add_argument(
        "--abs",
        dest="abs_paths",
        action="store_true",
        help="With --scan, show absolute paths instead of stripped display paths.",
    )
    ap.add_argument(
        "--engines",
        action="store_true",
        help="With --scan, also show which engine would handle each file.",
    )

    # Rendering (default behavior)
    ap.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories (e.g. '.' or 'src/')",
    )
    ap.add_argument(
        "--config", type=Path, default=None, help="YAML with default/rules/excludes"
    )
    ap.add_argument(
        "--default", type=str, default="full", help="Default mode if no rule matches"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write output to file instead of stdout",
    )

    args = ap.parse_args(argv)

    # Verbose → turn on internal debugs across modules
    if args.verbose and not os.environ.get("PYTOMA_DEBUG"):
        os.environ["PYTOMA_DEBUG"] = "1"

    debug("python=", sys.executable, "cli_file=", __file__, tag="cli")
    try:
        import pytoma as _pkg  # resolves installed package location

        debug("pkg_file=", getattr(_pkg, "__file__", None), tag="cli")
    except Exception as e:
        debug("pkg_import_error:", repr(e), tag="cli")

    debug("cwd=", str(Path.cwd()), tag="cli")
    debug("argv=", list(sys.argv), tag="cli")
    debug("args.paths=", [str(p) for p in args.paths], tag="cli")

    # Load configuration (for excludes + default)
    cfg = Config.load(args.config, args.default)

    # -------- SCAN MODE --------
    if args.scan:
        roots = _as_paths(args.paths)
        return _run_scan(
            roots,
            cfg=cfg,
            show_abs=args.abs_paths,
            show_engine=args.engines,
        )

    # -------- RENDER MODE (default) --------
    try:
        text = build_prompt(args.paths, cfg)
    except RuntimeError as e:
        sys.stderr.write(f"[pytoma:error] {e}\n")
        return 2
    debug(
        "post-build_prompt: out_len=",
        len(text) if isinstance(text, str) else "<?>",
        tag="cli",
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = text if text.strip() else "# (no files found)\n"
        args.out.write_text(payload, encoding="utf-8")
        return 0
    else:
        if text.strip():
            sys.stdout.write(text)
        else:
            debug("empty-output: printing sentinel to stdout", tag="cli")
            sys.stdout.write("# (no files found)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
