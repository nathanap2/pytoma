from __future__ import annotations
import types
from pathlib import Path
import importlib
import sys
import re

import pytest
from pytoma.markers import make_omission_line, DEFAULT_OPTIONS
from pytoma.engines.markdown_naive import MarkdownFallbackEngine

_engines_mod = importlib.import_module("pytoma.engines.markdown_engine")


MarkdownEngine = getattr(_engines_mod, "MarkdownEngine")
HAS_MDIT = getattr(_engines_mod, "MarkdownIt", None) is not None

# Small "duck-typed" Action: only the .kind property is read by the engine.
class _Action:
    def __init__(self, kind: str):
        self.kind = kind


def _apply_edits(text: str, edits):
    """Apply Edits (non-overlapping) to the source text."""
    if not edits:
        return text
    edits = sorted(edits, key=lambda e: (e.span[0], e.span[1]))
    out = []
    pos = 0
    for e in edits:
        a, b = e.span
        out.append(text[pos:a])
        out.append(e.replacement)
        pos = b
    out.append(text[pos:])
    return "".join(out)


@pytest.mark.skipif(
    not HAS_MDIT,
    reason="This test requires markdown-it-py to ignore # in fences.",
)
def test_headings_ignore_hash_in_code_fence_and_section_span_is_correct():
    md = """Intro

## Une section

~~~python
# Ceci est un commentaire Python
print("hello")
~~~

## Autre section
du texte
"""
    eng = MarkdownEngine()
    doc = eng.parse(Path("dummy.md"), md)

    # Retrieve sections (direct children of the root)
    sections = [n for n in doc.roots[0].children]
    titles = [s.name for s in sections]
    assert titles == ["Une section", "Autre section"]

    # The first section must cover the entire code block in its span
    s0 = sections[0]
    chunk0 = doc.text[s0.span[0] : s0.span[1]]
    assert "# Ceci est un commentaire Python" in chunk0
    assert 'print("hello")' in chunk0
    # And must not include the next heading
    assert "## Autre section" not in chunk0


@pytest.mark.skipif(
    not HAS_MDIT,
    reason="This test requires markdown-it-py for correct parsing of fences.",
)
def test_render_hide_removes_only_target_section():
    md = """Intro

## Une section

~~~python
# Ceci est un commentaire Python
print("hello")
~~~

## Autre section
du texte
"""
    eng = MarkdownEngine()
    doc = eng.parse(Path("dummy.md"), md)

    target = [n for n in doc.roots[0].children if n.name == "Une section"][0]
    edits = eng.render(doc, decisions=[(target, _Action("hide"))])
    out = _apply_edits(doc.text, edits)

    # The hidden section (heading + content) has disappeared
    assert "## Une section" not in out
    assert "# Ceci est un commentaire Python" not in out
    assert 'print("hello")' not in out
    # The other section is preserved
    assert "## Autre section" in out
    assert "du texte" in out


@pytest.mark.skipif(
    not HAS_MDIT, reason="Setext headings correctly recognized via markdown-it-py."
)
def test_setext_headings_parsed_with_levels():
    md = """Titre de niveau 1
======================

Intro

Sous-titre
----------

Corps
"""
    eng = MarkdownEngine()
    doc = eng.parse(Path("x.md"), md)

    secs = doc.roots[0].children
    assert [s.name for s in secs] == ["Titre de niveau 1", "Sous-titre"]
    assert secs[0].meta.get("level") == 1
    assert secs[1].meta.get("level") == 2


def test_atx_closed_heading_text_is_clean():
    md = """###   Title with hashes   ###   
Body
"""
    eng = MarkdownEngine()
    doc = eng.parse(Path("x.md"), md)
    secs = doc.roots[0].children
    assert len(secs) == 1
    assert secs[0].name == "Title with hashes"
    # The span includes the body (no following heading)
    chunk = doc.text[secs[0].span[0] : secs[0].span[1]]
    assert "Body" in chunk


def test_slugify_removes_accents_and_punctuation():
    md = "## Élévation & coût\nTexte"
    eng = MarkdownEngine()
    doc = eng.parse(Path("x.md"), md)
    slug = doc.roots[0].children[0].meta.get("slug")
    assert slug == "elevation-cout"


def test_hide_on_document_inserts_marker():
    md = """# Titre
Texte 1

## S1
a

## S2
b
"""
    eng = MarkdownEngine()
    doc = eng.parse(Path("x.md"), md)
    root = doc.roots[0]
    edits = eng.render(doc, decisions=[(root, _Action("hide"))])
    out = _apply_edits(doc.text, edits)

    n_lines = md.count("\n") + 1
    expected = make_omission_line(
        lang="md",
        a=1,
        b=n_lines,
        indent="",
        opts=DEFAULT_OPTIONS,
        label="document omitted",
    )
    assert out == expected


def test_markdown_fallback_parses_headings_basic():
    eng = MarkdownFallbackEngine()
    md = "# H1\nx\n## H2\ny\n"
    doc = eng.parse(Path("x.md"), md)
    secs = doc.roots[0].children
    assert [s.name for s in secs] == ["H1", "H2"]


def test_markdown_fallback_ignores_fenced_code_headings():
    eng = MarkdownFallbackEngine()
    md = """# H1

```

# not a heading

```

## H2
"""
    doc = eng.parse(Path("x.md"), md)
    secs = doc.roots[0].children
    assert [s.name for s in secs] == ["H1", "H2"]


def test_markdown_fallback_ignores_blockquote_headings():
    eng = MarkdownFallbackEngine()
    md = """# Top

> # Quoted heading
>> ## Nested quoted

## Bottom
"""
    doc = eng.parse(Path("x.md"), md)
    secs = doc.roots[0].children
    assert [s.name for s in secs] == ["Top", "Bottom"]


@pytest.mark.skipif(
    not HAS_MDIT,
    reason="The bug appears only with markdown-it-py (headings in blockquotes).",
)
def test_heading_inside_blockquote_does_not_terminate_section():
    md = """## Installation

~~~bash
# For now, in a freshly cloned version of pytoma : 
pip install -e .
# There will be a more official release soon
~~~

> # This looks like a heading inside a quote
> Requires Python ≥ 3.9 (uses `ast.unparse`) and deps: `libcst`, `pyyaml`.

## Usage
(du texte)
"""
    eng = MarkdownEngine()
    doc = eng.parse(Path("README.md"), md)

    # Target the "Installation" section
    sec = [n for n in doc.roots[0].children if n.name == "Installation"][0]
    chunk = doc.text[sec.span[0] : sec.span[1]]

    # Expected behavior (currently incorrect):
    # - the blockquote with "# ..." must NOT close the section
    # - the section must extend until "## Usage" (excluded)
    assert "> # This looks like a heading inside a quote" in chunk
    assert "## Usage" not in chunk  # next boundary

    # If we hide the section, none of this content should "leak" after rendering.
    edits = eng.render(doc, decisions=[(sec, _Action("hide"))])
    out = _apply_edits(doc.text, edits)
    assert "For now, in a freshly cloned version" not in out
    assert "There will be a more official release soon" not in out
    assert "This looks like a heading inside a quote" not in out
    assert "Requires Python ≥ 3.9" not in out
