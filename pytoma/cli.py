# pytoma/cli.py
import argparse
import os
import pathlib
import sys
from pathlib import Path
from typing import Iterable, List

from .core import Config, build_prompt
from .scan import iter_files

DEFAULT_INCLUDES: tuple[str, ...] = ("**/*.py", "**/*.md", "**/*.toml")
DEFAULT_EXCLUDES: tuple[str, ...] = ()


def _debug(*parts):
    if os.environ.get("PYTOMA_DEBUG"):
        sys.stderr.write("[pytoma:debug] " + " ".join(map(str, parts)) + "\n")


def _discover_files(paths: Iterable[Path]) -> List[Path]:
    """
    Walk directories via iter_files, keep explicit files,
    de-duplicate and sort (deterministic order). No fallback.
    """
    files: list[Path] = []
    seen: set[str] = set()

    norm_paths = [p if isinstance(p, Path) else Path(p) for p in paths]
    _debug("discover:start paths=", [str(p) for p in norm_paths])

    for p in norm_paths:
        _debug("path:", str(p),
               "cwd=", str(Path.cwd()),
               "exists=", p.exists(),
               "is_dir=", p.is_dir(),
               "is_file=", p.is_file())

        if p.is_dir():
            base = p.resolve()
            _debug("iter_files: base=", str(base),
                   "includes=", list(DEFAULT_INCLUDES),
                   "excludes=", list(DEFAULT_EXCLUDES))
            cnt = 0
            for f in iter_files([base], includes=DEFAULT_INCLUDES, excludes=DEFAULT_EXCLUDES):
                rp = f.resolve()
                key = rp.as_posix()
                if key not in seen:
                    seen.add(key)
                    files.append(rp)
                    cnt += 1
            _debug("iter_files: matched=", cnt, "for base=", str(base))
        elif p.is_file():
            rp = p.resolve()
            key = rp.as_posix()
            if key not in seen:
                seen.add(key)
                files.append(rp)
        else:
            _debug("ignore-nonexistent:", str(p))

    files.sort(key=lambda q: q.resolve().as_posix())
    _debug("discover:done total_files=", len(files))
    if files[:5]:
        _debug("discover:sample", [str(x) for x in files[:5]])
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-function code slicer for LLM prompts")
    ap.add_argument(
        "paths",
        nargs="+",
        type=pathlib.Path,
        help="Python files or directories (e.g. '.' or 'src/')",
    )
    ap.add_argument(
        "--config", type=pathlib.Path, default=None, help="YAML file with default/rules"
    )
    ap.add_argument(
        "--default", type=str, default="full", help="Default mode if no rule matches"
    )
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="Write output to file instead of stdout",
    )
    args = ap.parse_args(argv)

    _debug("python=", sys.executable, "cli_file=", __file__)
    try:
        import pytoma as _pkg
        _debug("pkg_file=", _pkg.__file__)
    except Exception as e:
        _debug("pkg_import_error:", repr(e))
    _debug("cwd=", str(Path.cwd()))
    _debug("argv=", list(sys.argv))
    _debug("args.paths=", [str(p) for p in args.paths])

    cfg = Config.load(args.config, args.default)

    file_list = _discover_files(args.paths)

    _debug("post-discover: n_files=", len(file_list))


    text = build_prompt(file_list, cfg)
    _debug("post-build_prompt: out_len=", len(text) if isinstance(text, str) else "<?>")
 

    if args.out:
        # If build_prompt returned an empty string (edge case), stay consistent:
        if not text.strip():
            _debug("empty-output: writing sentinel to file")
            args.out.write_text("# (no files found)\n", encoding="utf-8")
        return 0
    else:
        if text.strip():
            sys.stdout.write(text)
        else:
            _debug("empty-output: printing sentinel to stdout")
            sys.stdout.write("# (no files found)")
        return 0




if __name__ == "__main__":
    raise SystemExit(main())

