from __future__ import annotations

import importlib
import os
import sys
import pathlib
from pathlib import Path
from typing import List, Dict, Tuple

from .scan import iter_files
from .edits import apply_edits_preview, merge_edits
from .policies import Action, to_action
from .ir import Edit, Document, Node, PY_MODULE, MD_DOC, TOML_DOC
from .pre_resolution import pre_resolve_path_rules
from .config import Config, Rule
from .log import debug


# --- Display options ---
# "absolute": shows the absolute path
# "strip"   : shows the path relative to the provided root (the deepest one if multiple)
DISPLAY_PATH_MODE = "strip"  # or "absolute"

# Registered engine instances by extension (e.g. "py" -> PythonMinEngine instance)
_ENGINES: Dict[str, object] = {}

# --- Lazy engine loader (no side effects) ------------------------------------
# Mapping extension -> "module:factory"; the factory must return an Engine instance.
_ENGINE_FACTORY_BY_EXT: Dict[str, str] = {
    "py": "pytoma.engines.python_engine:create_engine",
    "md": "pytoma.engines.markdown_engine:create_engine",
    "toml": "pytoma.engines.toml_engine:create_engine",
    "xml": "pytoma.engines.xml_engine:create_engine",
}
_LOADED_EXTS: set[str] = set()  # extensions we attempted to load
_ENGINE_LOAD_ERRORS: Dict[str, str] = {}  # ext -> concise failure reason


def _register_engine_instance(engine) -> None:
    """Register a single engine instance for all of its declared filetypes."""
    for ext in getattr(engine, "filetypes", []):
        key = ext.lower().lstrip(".")
        _ENGINES[key] = engine


def _ensure_engine_loaded_for(ext: str) -> None:
    """
    Try to load and register the engine for 'ext'. Record the reason on failure.
    Never raises here; enforcement happens after discovery in build_prompt().
    """
    key = (ext or "").lstrip(".").lower()
    if not key or key in _ENGINES or key in _LOADED_EXTS:
        return

    _LOADED_EXTS.add(key)
    spec = _ENGINE_FACTORY_BY_EXT.get(key)
    if not spec:
        _ENGINE_LOAD_ERRORS[key] = "no engine factory registered for this extension"
        return

    mod_name, _, factory_name = spec.partition(":")
    factory_name = factory_name or "create_engine"
    try:
        mod = importlib.import_module(mod_name)
        factory = getattr(mod, factory_name)
        engine = factory()
        _register_engine_instance(engine)
    except Exception as e:
        _ENGINE_LOAD_ERRORS[key] = f"{type(e).__name__}: {e!s}"


def get_engine_for(path: Path):
    """Return the engine instance registered for the file's extension, if any."""
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
    """
    Produce candidate strings for matching a path rule:
    - absolute POSIX path
    - for each root that contains the path: relative path and "basename(root)/relative"
    """
    abs_posix = path.as_posix()
    cands = [abs_posix]
    for r in roots:
        try:
            rel = path.relative_to(r).as_posix()
        except ValueError:
            continue
        cands.append(rel)
        if r.name and rel:
            cands.append(f"{r.name}/{rel}")
    return list(dict.fromkeys(cands))


def _qual_candidates(node: Node, roots: list[pathlib.Path]) -> list[str]:
    """
    Construit des variantes de node.qual :
    - la qualname originale (module ou chemin selon l'engine),
    - puis des variantes où la partie "avant ':'" est réécrite en chemins
      (absolu, relatif à chaque root, basename(root)/rel), basées sur node.path.
    """
    if not node.qual or ":" not in node.qual:
        return [node.qual] if node.qual else []

    # Séparer mais ne pas convertir ce qui est avant ":" en chemin
    _before, rest = node.qual.split(":", 1)

    cands = [node.qual]

    # Utiliser le vrai chemin fichier du nœud pour générer les variantes chemin
    p = pathlib.Path(node.path)
    for c in _path_candidates(p, roots):
        cands.append(f"{c}:{rest}")

    # dédupe en conservant l'ordre
    return list(dict.fromkeys(cands))


def _decide_for_node(
    node: Node, cfg: Config, path_candidates: list[str], roots: list[pathlib.Path]
) -> Action:
    # 1) Qualname rules (tested against relative/absolute variants)
    if node.qual:
        qvars = _qual_candidates(node, roots)

        # --- helper: glob-lite (only * and ? are wildcards; [] are literals) ---
        def _glob_lite_match(pat: str, s: str) -> bool:
            # Fast-path: exact match when no wildcards
            if "*" not in pat and "?" not in pat:
                return s == pat
            import re as _re

            rx = _re.escape(pat).replace(r"\*", ".*").replace(r"\?", ".")
            return _re.fullmatch(rx, s) is not None

        for r in cfg.rules or []:
            if ":" in r.match:
                if any(_glob_lite_match(r.match, q) for q in qvars):
                    return to_action(r.mode)

    # 2) Path rules (on ABS + REL + basename/REL)
    for r in cfg.rules or []:
        if ":" not in r.match and any(fnmatchcase(c, r.match) for c in path_candidates):
            a = to_action(r.mode)
            if a.kind.startswith("file:"):
                if node.kind in {PY_MODULE, MD_DOC, TOML_DOC}:
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
        "xml": "xml",
    }.get(ext, "")


def build_prompt(paths: List[pathlib.Path], cfg: Config) -> str:
    debug(
        ("start", "n_files=", len(paths), "default=", getattr(cfg, "default", None)),
        tag="core",
    )

    out: List[str] = []

    # 1) Resolve potential conflicts in the config
    cfg, warns = pre_resolve_path_rules(cfg)
    if warns:
        for w in warns:
            debug(("pre_resolve", w), tag="core")

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
            for f in iter_files(
                [p],
                includes=("**/*.py", "**/*.md", "**/*.toml", "**/*.xml"),
                excludes=(cfg.excludes or []),
            ):
                discovered.append(f.resolve())
        elif p.is_file():
            roots.append(p.parent)
            discovered.append(p.resolve())
        else:
            # nonexistent path: silently ignore (same policy as the CLI)
            continue

    # dedup + deterministic sort
    dedup: Dict[str, pathlib.Path] = {q.as_posix(): q for q in discovered}
    discovered = sorted(dedup.values(), key=lambda q: q.as_posix())

    # 3) Lazy load required engines
    needed_exts = {p.suffix.lower().lstrip(".") for p in discovered if p.suffix}
    for ext in sorted(needed_exts):
        _ensure_engine_loaded_for(ext)

    missing = sorted(e for e in needed_exts if e and e not in _ENGINES)
    if missing:
        details = ", ".join(
            f".{e} → {_ENGINE_LOAD_ERRORS.get(e, 'unknown cause')}" for e in missing
        )
        raise RuntimeError(
            "No engine available for discovered extension(s): "
            f"{details}\n\nHints:\n"
            " • Install the corresponding engine or its dependencies "
            "(e.g. `pip install markdown-it-py` for Markdown).\n"
            " • Or exclude those files in config.yml → excludes (e.g. '**/*.md')."
        )

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
            # Should not happen in strict mode (we already enforced above),
            # but keep a defensive skip in case of race conditions.
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
