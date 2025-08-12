import pathlib, re
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import yaml

from .scan import iter_files
from .edits import apply_edits_preview, merge_edits
from .base import get_engine_for
from .policies import Action, to_action, validate_mode
from .ir import Edit, Document, Node, PY_MODULE, MD_DOC

# load engines (side-effect: register_engine)
from .engines import python_min  # noqa: F401
from .engines import markdown_min  # noqa: F401 
from .engines import toml_min     # noqa: F401


# --- Display options ---
# "absolute": shows the absolute path
# "strip"   : shows the path relative to the provided root (the deepest one if multiple)

DISPLAY_PATH_MODE = "strip"   # or "absolute"


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


@dataclass
class Rule:
    # match = either a qualname containing ":" (e.g. "pkg.mod:Class.func*"),
    # or a POSIX path glob "pkg/**/file.py"
    match: str
    mode: str

@dataclass
class Config:
    default: str = "full"
    rules: List[Rule] = None  # type: ignore
    excludes: List[str] = None

    @staticmethod
    def _coerce_rules(obj: object) -> List[Rule]:
        out: List[Rule] = []
        if not obj:
            return out
        if not isinstance(obj, list):
            raise TypeError("rules must be a list of {match, mode} objects")
        for i, r in enumerate(obj):
            if not isinstance(r, dict):
                raise TypeError(f"rules[{i}] must be a dict")
            match = r.get("match")
            mode = r.get("mode")
            if not isinstance(match, str) or not isinstance(mode, str):
                raise TypeError(f"rules[{i}] must contain 'match' (str) and 'mode' (str)")
            out.append(Rule(match=match, mode=validate_mode(str(mode))))
        return out

    @staticmethod
    def load(path: Optional[pathlib.Path], fallback_default: str = "full") -> "Config":
        default = validate_mode(str(fallback_default))
        rules: List[Rule] = []
        excludes = [
          ".venv/**","venv/**","**/__pycache__/**","dist/**","build/**","site-packages/**","**/*.pyi",
          "*.egg-info/**","**/*.egg-info/**","*.dist-info/**","**/*.dist-info/**", "*.pytest_cache/**", "**/*.pytest_cache/**"
        ]
        if path is None:
            return Config(default=default, rules=rules, excludes=excludes)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data is None:
            return Config(default=default, rules=rules, excludes=excludes)
        if not isinstance(data, dict):
            raise TypeError("YAML config must be a mapping")
        if "default" in data:
            default = validate_mode(str(data["default"]))
        rules = Config._coerce_rules(data.get("rules"))
        ex = data.get("excludes")
        if ex:
            if not isinstance(ex, list) or not all(isinstance(x, str) for x in ex):
                raise TypeError("excludes must be a list of strings")
            excludes = ex
        return Config(default=default, rules=rules, excludes=excludes)

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

    
def _decide_for_node(node: Node, cfg: Config, path_candidates: list[str], roots: list[pathlib.Path]) -> Action:
    # 1) qualname rules (now tested against relative/absolute variants)
    if node.qual:
        qvars = _qual_candidates(node, roots)
        for r in (cfg.rules or []):
            if ":" in r.match and any(fnmatchcase(q, r.match) for q in qvars):
                return to_action(r.mode)

    # 2) path rules (same as yours, on ABS + REL + basename/REL)
    for r in (cfg.rules or []):
        if ":" not in r.match and any(fnmatchcase(c, r.match) for c in path_candidates):
            a = to_action(r.mode)
            if a.kind.startswith("file:"):
                if node.kind in {PY_MODULE, MD_DOC}:  # add TOML_DOC if needed
                    return a
                continue
            return a

    # 3) default
    return to_action(cfg.default)


def _fence_lang_for(path: pathlib.Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return {"py": "python", "md": "markdown", "yaml": "yaml", "yml": "yaml", "toml": "toml"}.get(ext, "")

def build_prompt(paths: List[pathlib.Path], cfg: Config) -> str:
    out: List[str] = []

    # Roots: dirs that contain the provided paths
    roots: List[pathlib.Path] = []
    for p in paths:
        p = p.resolve()
        roots.append(p if p.is_dir() else p.parent)

    # Generic discovery
    discovered = list(iter_files(paths, includes=("**/*",), excludes=(cfg.excludes or [])))
    discovered.sort(key=lambda p: p.as_posix())

    all_edits: List[Edit] = []
    eligible: List[pathlib.Path] = []
    docs_text: Dict[pathlib.Path, str] = {}

    # parse -> decide -> edits
    for path in discovered:
        engine = get_engine_for(path)
        if not engine:
            continue

        # Optional hook
        configure = getattr(engine, "configure", None)
        if callable(configure):
            try:
                configure([r.resolve() for r in roots])
            except Exception:
                pass

        # Read the text only once (files handled by an engine)
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

    # resolve overlaps globally before preview
    all_edits = merge_edits(all_edits)

    # preview
    previews: Dict[pathlib.Path, str] = apply_edits_preview(all_edits)


    # pack — only eligible files (handled by an engine),
    # and without re-reading from disk (reuse docs_text)
    for path in eligible:
        shown = previews.get(path, docs_text[path])
        lang = _fence_lang_for(path)
        fence = f"```{lang}" if lang else "```"
        display = _display_path(path, roots)
        out.append(f"\n### {display}\n\n{fence}\n{shown}\n```\n")

    return "# (no files found)\n" if not out else "".join(out)

