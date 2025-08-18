import pathlib, re
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import importlib

import yaml

from pathlib import Path

from .scan import iter_files
from .edits import apply_edits_preview, merge_edits
from .policies import Action, to_action, validate_mode
from .ir import Edit, Document, Node, PY_MODULE, MD_DOC


from .pre_resolution import pre_resolve_path_rules

from .config import Config, Rule

import os, sys

from .log import debug


# --- Display options ---
# "absolute": shows the absolute path
# "strip"   : shows the path relative to the provided root (the deepest one if multiple)
DISPLAY_PATH_MODE = "strip"  # or "absolute"

_ENGINES = {}


# --- Lazy engine loader (no side effects) ------------------------------------
# Mapping extension -> "module:factory"; the factory must return an Engine instance.
_ENGINE_FACTORY_BY_EXT: Dict[str, str] = {
    "py": "pytoma.engines.python_engine:create_engine",  # choose ONE implementation
    "md": "pytoma.engines.markdown_engine:create_engine",
    "toml": "pytoma.engines.toml_engine:create_engine",
}
_LOADED_EXTS: set[str] = set()  # extensions we attempted to load


def _register_engine_instance(engine) -> None:
    # Registers the instance for all of its declared extensions
    for ext in getattr(engine, "filetypes", []):
        key = ext.lower().lstrip(".")
        _ENGINES[key] = engine


def _ensure_engine_loaded_for(ext: str) -> None:
    """
    Explicitly load the factory defined for 'ext', instantiate the engine,
    and register it — without relying on import-time side effects.
    """
    if ext in _ENGINES or ext in _LOADED_EXTS:
        return
    spec = _ENGINE_FACTORY_BY_EXT.get(ext)
    _LOADED_EXTS.add(ext)
    if not spec:
        return  # no factory defined for this extension
    mod_name, _, factory_name = spec.partition(":")
    if not factory_name:
        factory_name = "create_engine"
    try:
        mod = importlib.import_module(mod_name)
        factory = getattr(mod, factory_name)
        engine = factory()
        _register_engine_instance(engine)
    except Exception:
        return


def get_engine_for(path: Path):
    return _ENGINES.get(path.suffix.lower().lstrip("."))


def _display_path(path: pathlib.Path, roots: list[pathlib.Path]) -> str:
    """
    Return a path for display.
    - In "strip" mode: path relative to the most specific root (longest prefix match).
      Falls back to absolute if no root contains 'path'.
    - In "absolute" mode: POSIX absolute path.
    """

    p = path.resolve()
    if DISPLAY_PATH_MODE != "strip":
        return p.as_posix()

    best_rel = None
    best_len = -1
    for r in roots:
        try:
            rel = p.relative_to(r)
        except ValueError:
            continue
        # Prefer the deepest root (longest path)
        l = len(r.as_posix())
        if l > best_len:
            best_len = l
            best_rel = rel.as_posix()
    return best_rel if best_rel is not None else p.as_posix()


def fnmatchcase(s: str, pat: str) -> bool:
    import fnmatch

    return fnmatch.fnmatchcase(s, pat)


def _path_candidates(path: pathlib.Path, roots: list[pathlib.Path]) -> list[str]:
    abs_posix = path.as_posix()
    cands = [abs_posix]
    for r in roots:
        try:
            rel = path.relative_to(r).as_posix()
        except ValueError:
            continue
        cands.append(rel)
        if r.name and rel:
            cands.append(f"{r.name}/{rel}")  # <--- NEW
    return list(dict.fromkeys(cands))


def _qual_candidates(node: Node, roots: list[pathlib.Path]) -> list[str]:
    """
    Return variants of node.qual where the 'path' part (before ':')
    is rewritten as absolute, relative to each root, and 'basename(root)/rel'.
    """
    if not node.qual or ":" not in node.qual:
        return [node.qual] if node.qual else []
    path_str, rest = node.qual.split(":", 1)
    p = pathlib.Path(path_str)
    cands = []
    for c in _path_candidates(p, roots):  # reuse your existing helper
        cands.append(f"{c}:{rest}")
    # deduplicate while preserving order
    return list(dict.fromkeys(cands))


def _decide_for_node(
    node: Node, cfg: Config, path_candidates: list[str], roots: list[pathlib.Path]
) -> Action:
    # 1) Qualname rules (now tested against relative/absolute variants)
    if node.qual:
        qvars = _qual_candidates(node, roots)
        for r in cfg.rules or []:
            if ":" in r.match and any(fnmatchcase(q, r.match) for q in qvars):
                return to_action(r.mode)

    # 2) Path rules (same as yours, on ABS + REL + basename/REL)
    for r in cfg.rules or []:
        if ":" not in r.match and any(fnmatchcase(c, r.match) for c in path_candidates):
            a = to_action(r.mode)
            if a.kind.startswith("file:"):
                if node.kind in {PY_MODULE, MD_DOC}:  # add TOML_DOC if needed
                    return a
                continue
            return a

    # 3) Default
    return to_action(cfg.default)


def _fence_lang_for(path: pathlib.Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return {
        "py": "python",
        "md": "markdown",
        "yaml": "yaml",
        "yml": "yaml",
        "toml": "toml",
    }.get(ext, "")


def build_prompt(paths: List[pathlib.Path], cfg: Config) -> str:
    debug(
        ("start", "n_files=", len(paths), "default=", getattr(cfg, "default", None)),
        tag="core",
    )

    out: List[str] = []

    # 1) Resolve potential conflicts in the config
    cfg, warns = pre_resolve_path_rules(cfg)

    # 2) Normalize inputs + expand directories only
    norm_inputs: List[pathlib.Path] = []
    for p in paths:
        pp = p if isinstance(p, pathlib.Path) else pathlib.Path(p)
        norm_inputs.append(pp.resolve())

    roots: List[pathlib.Path] = []
    discovered: List[pathlib.Path] = []

    for p in norm_inputs:
        if p.is_dir():
            roots.append(p)
            # expand directories only: same extensions as the CLI
            for f in iter_files(
                [p],
                includes=("**/*.py", "**/*.md", "**/*.toml"),
                excludes=(cfg.excludes or []),
            ):
                discovered.append(f.resolve())
        elif p.is_file():
            roots.append(p.parent)
            discovered.append(p)
        else:
            # nonexistent path: silently ignore (same policy as the CLI)
            continue

    # dedup + deterministic sort
    dedup: Dict[str, pathlib.Path] = {q.as_posix(): q for q in discovered}
    discovered = sorted(dedup.values(), key=lambda q: q.as_posix())

    # 3) Lazy loading of required engines (if your infra requires it)
    needed_exts = {p.suffix.lower().lstrip(".") for p in discovered}
    for ext in sorted(needed_exts):
        _ensure_engine_loaded_for(ext)

    all_edits: List[Edit] = []
    eligible: List[pathlib.Path] = []
    docs_text: Dict[pathlib.Path, str] = {}

    # 4) parse -> decide -> render
    for path in discovered:
        engine = get_engine_for(path)
        debug(
            ("route", str(path), "engine=", getattr(engine, "__name__", repr(engine))),
            tag="core",
        )
        if not engine:
            debug(("skip-noengine", str(path)), tag="core")
            continue

        # Optional configuration hook for the engine (e.g., project roots)
        configure = getattr(engine, "configure", None)
        if callable(configure):
            try:
                # pass resolved (and deduplicated) roots to the engine
                cfg_roots = sorted(
                    {r.resolve().as_posix(): r.resolve() for r in roots}.values(),
                    key=lambda r: r.as_posix(),
                )
                configure(cfg_roots)
            except Exception:
                pass

        # Read the text once for files supported by an engine
        text = path.read_text(encoding="utf-8")
        docs_text[path] = text
        eligible.append(path)

        doc: Document = engine.parse(path, text)
        path_posix = path.as_posix()
        decisions: List[Tuple[Node, Action]] = []

        cands = _path_candidates(path, roots)

        for node in doc.nodes:
            a = _decide_for_node(node, cfg, cands, roots)
            if engine.supports(a):
                decisions.append((node, a))

        all_edits.extend(engine.render(doc, decisions))

    # 5) Global resolution of overlaps before preview
    all_edits = merge_edits(all_edits)

    # 6) Preview (apply edits)
    previews: Dict[pathlib.Path, str] = apply_edits_preview(all_edits)

    # 7) Packing: only "eligible" files (handled by an engine),
    # reusing docs_text to avoid a second disk read
    for path in eligible:
        shown = previews.get(path, docs_text[path])
        lang = _fence_lang_for(path)
        fence = f"```{lang}" if lang else "```"
        display = _display_path(path, roots)
        out.append(f"\n### {display}\n\n{fence}\n{shown}\n```\n")

    debug(("done",), tag="core")

    return "# (no files found)\n" if not out else "".join(out)
