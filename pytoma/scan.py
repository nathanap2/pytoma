import fnmatch
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Union
import os
import sys


def _dbg(*a):
    if os.environ.get("PYTOMA_DEBUG"):
        sys.stderr.write("[pytoma:debug:scan] " + " ".join(map(str, a)) + "\n")

RootT = Union[str, os.PathLike, Path]

def iter_files(
    roots: Iterable[RootT],
    includes: Iterable[str] = ("**/*",),
    excludes: Iterable[str] = (),
) -> Iterator[Path]:
    """
    File discovery with glob/fnmatch patterns (language-agnostic).

    - `roots`: search roots. Accepts str / PathLike / Path.
    - `includes`: glob patterns *relative to each root* (e.g., "**/*.py").
    - `excludes`: fnmatch patterns on the *relative POSIX path* (e.g., "tests/**").

    Behavior:
      1) Normalize roots to `Path`.
      2) For each root, apply every `includes` pattern.
      3) Ignore directories, keep only files.
      4) Apply `excludes` on the POSIX-relative path.
      5) De-duplicate by (root, relative path) and **sort** results.
      6) `yield` paths after sorting (deterministic order).
    """
    # -- 1) Normalize roots to Path
    if isinstance(roots, (str, Path, os.PathLike)):
        roots = [roots]  # type: ignore[assignment]

    norm_roots: list[Path] = []
    for r in roots:  # type: ignore[assignment]
        norm_roots.append(r if isinstance(r, Path) else Path(r))

    _dbg("iter.start", "roots=", [str(r) for r in norm_roots],
         "includes=", list(includes), "excludes=", list(excludes))

    seen: set[tuple[str, str]] = set()    # (base_resolved, rel_posix)
    out: list[Path] = []

    # -- 2) Walk roots and patterns
    for root in norm_roots:
        base = root.resolve()
        for inc in includes:
            matched_this_pattern = 0
            # -- 3) Traverse files matching the `inc` pattern
            for p in base.glob(inc):
                if not p.is_file():
                    continue

                # POSIX-relative path to the root
                rel_posix = PurePosixPath(p.relative_to(base).as_posix())

                # -- 4) Exclusions
                if any(fnmatch.fnmatch(str(rel_posix), pat) for pat in excludes):
                    continue

                # -- 5) Deduplication
                key = (str(base), str(rel_posix))
                if key in seen:
                    continue
                seen.add(key)

                out.append(p)
                matched_this_pattern += 1

            _dbg("iter.glob", "base=", str(base), "inc=", inc, "matched=", matched_this_pattern)

    # -- 6) Deterministic sort then yield
    out.sort(key=lambda q: (str(q.parent.resolve()), q.name))

    _dbg("iter.done", "total=", len(out))
    for p in out:
        yield p

