import fnmatch
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator


def iter_files(
    roots: Iterable[Path],
    includes: Iterable[str] = ("**/*",),
    excludes: Iterable[str] = (),
) -> Iterator[Path]:
    """
    File discovery with glob/fnmatch patterns (language-agnostic).
    - 'includes' apply relative to each root.
    - 'excludes' are fnmatch-style patterns on the POSIX *relative* path.
    """
    roots = list(roots)
    for root in roots:
        base = root.resolve()
        for inc in includes:
            for p in base.glob(inc):
                if not p.is_file():
                    continue
                rel = PurePosixPath(p.relative_to(base).as_posix())
                if any(fnmatch.fnmatch(rel, pat) for pat in excludes):
                    continue
                yield p
