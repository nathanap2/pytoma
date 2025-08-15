import pytest
from pathlib import Path, PurePosixPath
from pytoma.ir import Edit
from pytoma.edits import merge_edits, apply_edits_preview


def _E(path, span, repl):
    return Edit(path=PurePosixPath(path), span=span, replacement=repl)


def test_nested_edits_outer_wins_merge():
    edits = [
        _E("a.py", (0, 100), "OUT"),
        _E("a.py", (10, 20), "INNER"),
        _E("a.py", (30, 40), "INNER2"),
    ]
    merged = merge_edits(edits)
    # only the outer edit remains
    assert len(merged) == 1
    assert merged[0].span == (0, 100)
    assert merged[0].replacement == "OUT"


def test_partial_overlap_raises():
    edits = [
        _E("a.py", (0, 50), "A"),
        _E("a.py", (40, 80), "B"),
    ]
    with pytest.raises(ValueError):
        merge_edits(edits)


def test_disjoint_edits_apply_preview():
    text = "0123456789abcdefghij"
    edits = [
        _E("a.txt", (2, 4), "XX"),  # replaces "23" -> "0 1 XX 4 ..."
        _E("a.txt", (10, 12), "YY"),  # replaces "ab" -> "... 9 YY c ..."
    ]
    # apply_edits_preview must call merge_edits internally
    # write a virtual file? Here we simulate: we will monkeypatch the reading.
    # To keep it simple and without I/O, we test merge + _apply_edits_to_text indirectly
    # by creating a small local helper (or alternatively create an integration test with tmp_path).


def test_apply_preview_with_tmp_path(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0123456789abcdefghij", encoding="utf-8")
    edits = [
        _E(p.as_posix(), (2, 4), "XX"),
        _E(p.as_posix(), (10, 12), "YY"),
    ]
    out = apply_edits_preview(edits)
    assert p in out
    assert out[p] == "01XX456789YYcdefghij"


def test_merge_is_per_file():
    edits = [
        _E("a.py", (0, 10), "A"),
        _E("b.py", (0, 10), "B"),
        _E("a.py", (2, 5), "INNER"),  # nested in a.py -> removed by merge
    ]
    merged = merge_edits(edits)
    # a.py outer only + b.py unique
    assert {(e.path.as_posix(), e.span) for e in merged} == {
        ("a.py", (0, 10)),
        ("b.py", (0, 10)),
    }


def test_apply_preview_rejects_partial_overlap(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("abcdefghij", encoding="utf-8")
    edits = [
        _E(p.as_posix(), (0, 6), "X"),
        _E(p.as_posix(), (4, 9), "Y"),  # partial overlap
    ]
    with pytest.raises(ValueError):
        apply_edits_preview(edits)
        



def test_merge_edits_outermost_and_insertions():
    p = PurePosixPath("x.py")
    # One outer deletion [10,20), one fully-contained deletion [12,15), and two insertions
    outer = Edit(path=p, span=(10, 20), replacement="")
    inner = Edit(path=p, span=(12, 15), replacement="")
    ins_inside = Edit(path=p, span=(14, 14), replacement="# INS\n")
    ins_before = Edit(path=p, span=(0, 0), replacement="# BOF\n")

    merged = merge_edits([inner, outer, ins_inside, ins_before])

    # After merge:
    # - only 'outer' survives among deletions
    # - insertion inside is shifted to 20
    # - insertion at 0 stays at 0
    spans = [(e.span[0], e.span[1]) for e in merged if Path(e.path).name == "x.py"]
    assert (10, 20) in spans
    assert (0, 0) in spans
    assert (20, 20) in spans
    assert (12, 15) not in spans  # inner deletion dropped


def test_merge_edits_partial_overlap_raises():
    p = PurePosixPath("x.py")
    a = Edit(path=p, span=(10, 20), replacement="")
    b = Edit(path=p, span=(15, 25), replacement="")
    with pytest.raises(ValueError):
        merge_edits([a, b])

