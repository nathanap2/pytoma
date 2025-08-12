from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import pathlib
from .policies import validate_mode
import yaml


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
