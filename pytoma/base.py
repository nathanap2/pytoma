from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Protocol, Dict, Tuple

from .ir import Document, Node, Edit
from .policies import Action, validate_action


class Engine(Protocol):
    """One engine per file type (detected via extension)."""

    filetypes: set[str]  # e.g., {"py"} or {"md"} or {"yaml","yml","toml"}

    def parse(self, path: Path, text: str) -> Document:
        """Build the Document + Nodes (a lightweight tree)."""
        ...

    def supports(self, action: Action) -> bool:
        """Advertise which policies/actions this engine understands."""
        ...

    def render(self, doc: Document, decisions: List[Tuple[Node, Action]]) -> List[Edit]:
        """
        Translate (Node, Action) pairs into concrete Edits (spans + replacements).
        Contract: all Edits must target doc.path.
        """
        ...

    # Optional: engines may need repo roots (e.g., to compute module names)
    def configure(self, roots: List[Path]) -> None:  # type: ignore[empty-body]
        ...


_ENGINES: Dict[str, Engine] = {}


def register_engine(engine: Engine) -> None:
    for ext in engine.filetypes:
        _ENGINES[ext.lower().lstrip(".")] = engine


def get_engine_for(path: Path) -> Engine | None:
    return _ENGINES.get(path.suffix.lower().lstrip("."))
