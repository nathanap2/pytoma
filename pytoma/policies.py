from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable
import re


@dataclass(frozen=True)
class Action:
    """Declarative action (to be interpreted by the relevant engine)."""

    kind: str  # "hide" | "full" | "sig+doc" | "sig" | "levels" | ...
    params: Dict[str, Any] = field(default_factory=dict)


# Concise helpers (quality of life)
def hide() -> Action:
    return Action("hide")


def full() -> Action:
    return Action("full")


def sig_doc() -> Action:
    return Action("sig+doc")


def sig() -> Action:
    return Action("sig")


def levels(k: int) -> Action:
    return Action("levels", {"k": int(k)})


# (Optional) action registry if you want centralized validation
_VALIDATORS: Dict[str, Callable[[Action], None]] = {}


def register_action(
    kind: str, validator: Optional[Callable[[Action], None]] = None
) -> None:
    _VALIDATORS[kind] = validator or (lambda a: None)


def validate_action(a: Action) -> None:
    if a.kind in _VALIDATORS:
        _VALIDATORS[a.kind](a)


# default registrations
register_action("hide")
register_action("full")
register_action("sig+doc")
register_action("sig")
register_action(
    "levels",
    validator=lambda a: (
        isinstance(a.params.get("k"), int)
        or (_ for _ in ()).throw(ValueError("levels: k must be int"))
    ),
)
register_action("file:no-imports")

# -----------------------------
# Modes (string) → Action
# -----------------------------

MODE_RE = re.compile(
    r"^(hide|sig|sig\+doc|full|body:levels=(\d+)"
    r"|file:no-imports|file:no-legacy-strings|file:no-path-defs|file:no-sys-path|file:tidy)$"
)


def validate_mode(mode: str) -> str:
    if not MODE_RE.match(mode):
        raise ValueError(f"invalid mode: {mode}")
    return mode


def to_action(mode: str) -> Action:
    validate_mode(mode)
    if mode == "hide":
        return hide()
    if mode == "full":
        return full()
    if mode == "sig":
        return sig()
    if mode == "sig+doc":
        return sig_doc()
    if mode == "file:no-imports":
        return Action("file:no-imports")
    if mode == "file:no-legacy-strings":
        return Action("file:no-legacy-strings")
    if mode == "file:no-path-defs":
        return Action("file:no-path-defs")
    if mode == "file:no-sys-path":
        return Action("file:no-sys-path")
    if mode == "file:tidy":
        return Action("file:tidy")
    m = re.match(r"body:levels=(\d+)", mode)
    if m:
        return levels(int(m.group(1)))
    return full()
