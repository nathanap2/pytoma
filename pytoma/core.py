import pathlib, re
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import yaml

from .scan import iter_files
from .edits import apply_edits_preview, merge_edits
from .base import get_engine_for
from .policies import Action, to_action, validate_mode
from .ir import Edit, Document, Node

# load engines (side-effect: register_engine)
from .engines import python_min  # noqa: F401
from .engines import markdown_min  # noqa: F401 

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
        excludes = [".venv/**","**/__pycache__/**","dist/**","build/**","site-packages/**","**/*.pyi"]
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

def _decide_for_node(node: Node, cfg: Config, path_posix: str) -> Action:
    # 1) rules on qualname (heuristic: presence of ":")
    if node.qual:
        for r in (cfg.rules or []):
            if ":" in r.match and fnmatchcase(node.qual, r.match):
                return to_action(r.mode)
    # 2) rules on path (glob)
    for r in (cfg.rules or []):
        if ":" not in r.match and fnmatchcase(path_posix, r.match):
            return to_action(r.mode)
    # 3) default
    return to_action(cfg.default)

def _fence_lang_for(path: pathlib.Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return {"py": "python", "md": "markdown", "yaml": "yaml", "yml": "yaml", "toml": "toml"}.get(ext, "")

def build_prompt(paths: List[pathlib.Path], cfg: Config, verbose: bool = False) -> str:
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

        for node in doc.nodes:
            a = _decide_for_node(node, cfg, path_posix)
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
        out.append(f"\n### {path}\n\n{fence}\n{shown}\n```\n")

    return "# (no files found)\n" if not out else "".join(out)
