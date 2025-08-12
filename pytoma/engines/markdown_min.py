from pathlib import Path, PurePosixPath
from typing import List, Tuple, Dict, Optional
import re
import unicodedata

from ..ir import Document, Node, Edit, MD_DOC, MD_HEADING, assign_ids, flatten
from ..policies import Action
from ..base import register_engine

_HEADING_RE = re.compile(r'^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$', re.MULTILINE)

def _line_starts(text: str) -> List[int]:
    starts = [0]
    acc = 0
    for line in text.splitlines(keepends=True):
        acc += len(line)
        starts.append(acc)
    return starts

def _slugify(title: str) -> str:
    s = unicodedata.normalize("NFKD", title)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "section"

class MarkdownMinEngine:
    filetypes = {"md"}

    def configure(self, roots: List[Path]) -> None:  # no-op (signature pour homogénéité)
        return

    def parse(self, path: Path, text: str) -> Document:
        posix = PurePosixPath(path.as_posix())
        ls = _line_starts(text)

        # Racine document
        root = Node(
            kind=MD_DOC,
            path=posix,
            span=(0, len(text)),
            name=str(posix),
            qual=str(posix),   # on peut cibler le doc entier via un match sur le chemin
            meta={},
            children=[]
        )

        # Collecte des titres
        headings: List[Tuple[int, int, int, str]] = []  # (level, line_idx, char_start, title)
        for m in _HEADING_RE.finditer(text):
            hashes = m.group(1)
            title = m.group(2).strip()
            level = len(hashes)
            char_start = m.start()
            # convertir char_start en numéro de ligne (binaire sur ls)
            # astuce: trouve la plus grande i tel que ls[i] <= char_start
            # (recherche linéaire suffit pour un md; garder simple)
            line_idx = 0
            for i in range(len(ls)-1):
                if ls[i] <= char_start < ls[i+1]:
                    line_idx = i + 1  # 1-based
                    break
            headings.append((level, line_idx, char_start, title))

        # Déterminer les spans de section : jusqu’au prochain titre de niveau <= courant
        nodes: List[Node] = []
        n = len(headings)
        for i, (lvl, line_idx, start_char, title) in enumerate(headings):
            end_char = len(text)
            for j in range(i+1, n):
                nxt_lvl, _, nxt_start_char, _ = headings[j]
                if nxt_lvl <= lvl:
                    end_char = nxt_start_char
                    break
            slug = _slugify(title)
            qual = f"{posix}:{slug}"
            node = Node(
                kind=MD_HEADING,
                path=posix,
                span=(start_char, end_char),
                name=title,
                qual=qual,
                meta={"level": lvl, "slug": slug},
                children=[]
            )
            nodes.append(node)

        # Arbre plat (sections comme enfants de la racine)
        root.children.extend(nodes)
        assign_ids([root])
        flat = flatten([root])

        return Document(path=posix, text=text, roots=[root], nodes=flat)

    def supports(self, action: Action) -> bool:
        return action.kind in {"hide", "full"}

    def render(self, doc: Document, decisions: List[Tuple[Node, Action]]) -> List[Edit]:
        candidates: List[Edit] = []
        for node, action in decisions:
            if action.kind == "full":
                continue
            if action.kind == "hide":
                # hide sur la racine -> supprimer tout le doc
                if node.kind == MD_DOC:
                    candidates.append(Edit(path=doc.path, span=(0, len(doc.text)), replacement=""))
                elif node.kind == MD_HEADING:
                    candidates.append(Edit(path=doc.path, span=node.span, replacement=""))
        return candidates

register_engine(MarkdownMinEngine())

