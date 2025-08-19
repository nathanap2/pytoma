from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

from .policies import validate_mode
from .policies import to_action  # re-export used elsewhere


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

    # ------------------------------
    # Parsing helpers (unchanged)
    # ------------------------------
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
                raise TypeError(
                    f"rules[{i}] must contain 'match' (str) and 'mode' (str)"
                )
            out.append(Rule(match=match, mode=validate_mode(str(mode))))
        return out

    @staticmethod
    def _default_excludes() -> List[str]:
        # Keep this in sync with README and scan.py docs.
        return [
            ".venv/**",
            "venv/**",
            "**/__pycache__/**",
            "dist/**",
            "build/**",
            "site-packages/**",
            "**/*.pyi",
            "*.egg-info/**",
            "**/*.egg-info/**",
            "*.dist-info/**",
            "**/*.dist-info/**",
            "*.pytest_cache/**",
            "**/*.pytest_cache/**",
        ]

    # ------------------------------
    # Loaders (path / text / builtin)
    # ------------------------------
    @staticmethod
    def _from_mapping(data: dict, fallback_default: str) -> "Config":
        """Common builder from a raw YAML mapping."""
        default = validate_mode(str(fallback_default))
        rules: List[Rule] = []
        excludes = Config._default_excludes()

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

    @staticmethod
    def load(path: Optional[pathlib.Path], fallback_default: str = "full") -> "Config":
        """Original path-based loader (kept for backward compatibility)."""
        default = validate_mode(str(fallback_default))
        if path is None:
            return Config(
                default=default, rules=[], excludes=Config._default_excludes()
            )
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data is None:
            return Config(
                default=default, rules=[], excludes=Config._default_excludes()
            )
        return Config._from_mapping(data, default)

    @staticmethod
    def load_text(yaml_text: str, fallback_default: str = "full") -> "Config":
        """Parse from a YAML string (used for built-in resources)."""
        default = validate_mode(str(fallback_default))
        data = yaml.safe_load(yaml_text)
        if data is None:
            return Config(
                default=default, rules=[], excludes=Config._default_excludes()
            )
        return Config._from_mapping(data, default)

    @staticmethod
    def list_builtins() -> List[str]:
        """
        Return the list of built-in config names (without extension).
        """
        try:
            from importlib import resources as _res

            pkg = "pytoma.configs"
            base = _res.files(pkg)
            names: List[str] = []
            for p in base.iterdir():
                if not p.is_file():
                    continue
                if p.name.endswith((".yaml", ".yml")):
                    stem = p.name.rsplit(".", 1)[0]
                    names.append(stem)
            return sorted(dict.fromkeys(names))
        except Exception:
            return []

    @staticmethod
    def load_builtin(name: str, fallback_default: str = "full") -> "Config":
        """
        Load a built-in YAML config by name (e.g., 'skeleton').
        Looks for pytoma/configs/<name>.yaml or .yml.
        """
        from importlib import resources as _res

        pkg = "pytoma.configs"

        # try .yaml then .yml
        for ext in (".yaml", ".yml"):
            fname = f"{name}{ext}"
            try:
                yaml_text = _res.files(pkg).joinpath(fname).read_text(encoding="utf-8")
                return Config.load_text(yaml_text, fallback_default)
            except FileNotFoundError:
                continue
            except Exception as e:
                raise RuntimeError(
                    f"failed to load built-in config '{name}': {e}"
                ) from e
        raise ValueError(
            f"unknown built-in config '{name}'. "
            f"Available: {', '.join(Config.list_builtins()) or '(none found)'}"
        )

    @staticmethod
    def load_any(arg: Optional[str], fallback_default: str = "full") -> "Config":
        """
        Resolve --config argument that may be:
          - a filesystem path (if exists), or
          - a built-in name (no path exists), or
          - explicitly marked: 'builtin:<name>', 'preset:<name>', '@<name>', ':<name>'.
        """
        default = validate_mode(str(fallback_default))
        if not arg:
            return Config(
                default=default, rules=[], excludes=Config._default_excludes()
            )

        # explicit builtin markers
        markers = ("builtin:", "preset:", "@", ":")
        for m in markers:
            if arg.startswith(m):
                name = arg[len(m) :]
                if not name:
                    raise ValueError("empty built-in config name")
                return Config.load_builtin(name, default)

        # path if it exists, else treat as builtin name
        p = pathlib.Path(arg)
        if p.exists():
            return Config.load(p, default)
        return Config.load_builtin(arg, default)
