# tests/test_core_qual_candidates.py
from pathlib import Path, PurePosixPath

import pytest

from pytoma.ir import Node, PY_FUNCTION
from pytoma.core import _qual_candidates, _path_candidates, _decide_for_node
from pytoma.config import Config, Rule
from pytoma.policies import to_action


def _make_node(abs_file: Path, qual: str) -> Node:
    # On construit un Node minimal : seuls path/qual/kind/span sont utilisés ici.
    return Node(
        kind=PY_FUNCTION,
        path=PurePosixPath(abs_file.as_posix()),
        span=(0, 1),
        name="foo",
        qual=qual,
        meta={},
        children=[],
    )


def test_qual_candidates_includes_original_module_qualname(tmp_path: Path):
    """
    Attendu : la qualname d'origine (module:local) doit apparaître
    dans les variantes retournées par _qual_candidates.

    État actuel (avant patch) : échoue, car _qual_candidates n'inclut pas
    la qualname originale "pkg.mod:foo".
    """
    root = tmp_path / "repo"
    p = root / "pkg" / "mod.py"
    node = _make_node(p, qual="pkg.mod:foo")

    cands = _qual_candidates(node, roots=[root])

    assert "pkg.mod:foo" in cands, (
        "La qualname d'origine 'pkg.mod:foo' devrait être présente dans les "
        "variantes de matching (échoue avant patch)."
    )


def test_qual_candidates_generates_variants_from_node_path(tmp_path: Path):
    """
    Attendu : les variantes basées sur le CHEMIN réel du nœud doivent apparaître :
      - absolu:    /.../repo/pkg/mod.py:foo
      - relatif:   pkg/mod.py:foo
      - base/rel : repo/pkg/mod.py:foo

    État actuel (avant patch) : échoue, car la partie avant ':' est traitée
    comme un chemin littéral "pkg.mod" (cwd/pkg.mod) au lieu d'utiliser node.path.
    """
    root = tmp_path / "repo"
    p = root / "pkg" / "mod.py"
    node = _make_node(p, qual="pkg.mod:foo")

    cands = _qual_candidates(node, roots=[root])

    abs_var = f"{p.as_posix()}:foo"
    rel = p.relative_to(root).as_posix()
    rel_var = f"{rel}:foo"
    base_rel_var = f"{root.name}/{rel}:foo"

    # Les trois variantes issues de _path_candidates(node.path, roots)
    assert (
        abs_var in cands
    ), "La variante chemin absolu doit être présente (échoue avant patch)."
    assert (
        rel_var in cands
    ), "La variante chemin relatif doit être présente (échoue avant patch)."
    assert (
        base_rel_var in cands
    ), "La variante basename(root)/rel doit être présente (échoue avant patch)."


def test_decide_for_node_matches_module_qual_rule(tmp_path: Path):
    """
    Test d'intégration minimal : une règle qualname 'pkg.mod:*' doit matcher
    un nœud dont qual = 'pkg.mod:foo' (et retourner l'action 'sig').

    État actuel (avant patch) : _decide_for_node ne matche pas la règle qualname,
    et tombe sur le 'default' (full).
    """
    root = tmp_path / "repo"
    p = root / "pkg" / "mod.py"
    node = _make_node(p, qual="pkg.mod:foo")

    cfg = Config(
        default="full",
        rules=[Rule(match="pkg.mod:*", mode="sig")],
        excludes=[],
    )

    path_candidates = _path_candidates(p, [root])
    action = _decide_for_node(node, cfg, path_candidates, [root])

    assert action.kind == "sig", (
        "La règle de qualname 'pkg.mod:*' devrait matcher et produire 'sig' "
        "(échoue avant patch car la variante 'pkg.mod:foo' n'est pas générée)."
    )
