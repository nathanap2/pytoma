# tests/test_xml_engine.py
from __future__ import annotations
from textwrap import dedent
from pathlib import Path

import pytest

from pytoma.policies import hide
from pytoma.edits import apply_edits_preview
from pytoma.ir import XML_DOC, XML_ELEMENT
from pytoma.engines.xml_engine import create_engine

import re


def _count_open_a(s: str) -> int:
    s_nc = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    return len(re.findall(r"<\s*a(?=[\s/>])", s_nc))


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_render_hide_element_replaces_only_target(tmp_path: Path):
    xml = dedent(
        """\
        <root>
          <a/>
          <a><b/><b/></a>
          <a><b>t</b></a>
        </root>
    """
    )
    p = _write(tmp_path, "hide_elem.xml", xml)
    eng = create_engine()
    doc = eng.parse(p, xml)

    # Target: /root[1]/a[2]
    target = next(n for n in doc.nodes if n.kind == XML_ELEMENT and n.meta.get("xpath") == "/root[1]/a[2]")  # type: ignore[union-attr]
    edits = eng.render(doc, [(target, hide())])

    # apply_edits_preview reads the physical file: it must exist
    preview = apply_edits_preview(edits)[p]
    # An XML comment has been inserted, with the engine’s label
    assert "<!--" in preview and "-->" in preview
    assert "element <a> omitted" in preview
    # The number of <a ...> tags decreases by 1 (excluding comments)
    assert _count_open_a(preview) == _count_open_a(xml) - 1


def test_parse_builds_xpath_and_indices(tmp_path: Path):
    xml = dedent(
        """\
        <root>
          <a/>
          <a><b/><b/></a>
          <a><b>t</b></a>
        </root>
    """
    )
    p = _write(tmp_path, "tree.xml", xml)
    eng = create_engine()
    doc = eng.parse(p, xml)

    # Document root present
    assert any(n.kind == XML_DOC for n in doc.roots)
    # All elements in prefix traversal
    els = [n for n in doc.nodes if n.kind == XML_ELEMENT]
    xpaths = [n.meta["xpath"] for n in els]  # type: ignore[index]
    assert xpaths == [
        "/root[1]",
        "/root[1]/a[1]",
        "/root[1]/a[2]",
        "/root[1]/a[2]/b[1]",
        "/root[1]/a[2]/b[2]",
        "/root[1]/a[3]",
        "/root[1]/a[3]/b[1]",
    ]
    # Fully addressable qualnames: "<abs_path>:/root[1]/a[2]"
    for n in els:
        assert str(p.as_posix()) in (n.qual or "")


def test_render_hide_document(tmp_path: Path):
    xml = "<root><x/><y><z/></y></root>\n"
    p = _write(tmp_path, "hide_doc.xml", xml)
    eng = create_engine()
    doc = eng.parse(p, xml)
    doc_node = next(n for n in doc.nodes if n.kind == XML_DOC)
    edits = eng.render(doc, [(doc_node, hide())])
    preview = apply_edits_preview(edits)[p]
    # Document replaced by a single comment
    assert preview.strip().startswith("<!--")
    assert preview.strip().endswith("-->")
    assert "<root" not in preview


def test_parser_skips_comments_cdata_pi_and_handles_self_closing(tmp_path: Path):
    xml = dedent(
        """\
        <?xml version="1.0"?>
        <!DOCTYPE root>
        <root>
          <!-- comment -->
          <![CDATA[ <fake><a/>inside</fake> ]]>
          <?pi target?>
          <x/>
        </root>
    """
    )
    p = _write(tmp_path, "skips.xml", xml)
    eng = create_engine()
    doc = eng.parse(p, xml)

    els = [n for n in doc.nodes if n.kind == XML_ELEMENT]
    xpaths = [n.meta["xpath"] for n in els]  # type: ignore[index]
    # Only root and x exist as elements of the textual DOM
    assert xpaths == ["/root[1]", "/root[1]/x[1]"]


def test_core_build_prompt_with_xml_rule_by_qualname(tmp_path: Path, monkeypatch):
    """
    Minimalistic integration with core.build_prompt:
    - we manually register the engine (no need for the .xml mapping to be present),
    - we pass a direct file (no directory exploration),
    - qualified rule to hide /root[1]/a[2].
    """
    xml = dedent(
        """\
        <root>
          <a/>
          <a><b/><b/></a>
          <a><b>t</b></a>
        </root>
    """
    )
    p = _write(tmp_path, "pack.xml", xml)

    # Config with a rule on absolute qualname
    from pytoma.config import Config, Rule

    cfg = Config(
        default="full",
        rules=[Rule(match=f"{p.as_posix()}:/root[1]/a[2]", mode="hide")],
        excludes=[],
    )

    # Inject the engine into core (we avoid the lazy/factory loader)
    import pytoma.core as core

    eng = create_engine()
    # register the engine for its extension
    core._register_engine_instance(eng)  # type: ignore[attr-defined]

    out = core.build_prompt([p], cfg)
    # The wrapping must contain a block for the file and the omission comment
    assert "###" in out and "```" in out
    assert "element <a> omitted" in out
    # The last <a><b>t</b></a> remains visible
    assert "<b>t</b>" in out
