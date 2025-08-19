from __future__ import annotations
from pathlib import Path, PurePosixPath
from typing import List
import re, unicodedata, codecs


def line_starts(text: str) -> List[int]:
    """Return character offsets of the start of each line, with a final sentinel."""
    starts = [0]
    acc = 0
    for line in text.splitlines(keepends=True):
        acc += len(line)
        starts.append(acc)
    return starts


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    """Deterministic, accent-insensitive slug."""
    s = unicodedata.normalize("NFKD", title)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = _SLUG_RE.sub("-", s).strip("-")
    return s or "section"


def posix(p: Path | PurePosixPath) -> str:
    return PurePosixPath(str(p)).as_posix()


# Regex to detect encoding in an XML declaration like <?xml ... encoding="ISO-8859-1"?>
_XML_DECL_RX = re.compile(
    rb'<\?xml[^>]*encoding=["\'](?P<enc>[A-Za-z0-9._-]+)["\']', re.I
)


def _normalize_encoding_name(name: str) -> Optional[str]:
    """Return a normalized encoding name if Python recognizes it, else None."""
    try:
        codecs.lookup(name)
        return name
    except Exception:
        alias = name.lower().replace("_", "-")
        try:
            codecs.lookup(alias)
            return alias
        except Exception:
            return None


def _sniff_xml_decl_encoding(data: bytes) -> Optional[str]:
    """Look for an encoding declaration in the first 2KB of an XML file."""
    head = data[:2048]
    m = _XML_DECL_RX.search(head)
    if not m:
        return None
    return _normalize_encoding_name(m.group("enc").decode("ascii", "replace"))


def _sniff_bom_encoding(data: bytes) -> Optional[str]:
    """Detect BOM (Byte Order Mark) for UTF-8/16/32 and return the right codec name."""
    if data.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if data.startswith(codecs.BOM_UTF16_LE):
        return "utf-16-le"
    if data.startswith(codecs.BOM_UTF16_BE):
        return "utf-16-be"
    if data.startswith(codecs.BOM_UTF32_LE):
        return "utf-32-le"
    if data.startswith(codecs.BOM_UTF32_BE):
        return "utf-32-be"
    return None


def decode_bytes_best_effort(data: bytes, *, file_suffix: str = "") -> str:
    """
    Decode bytes into text robustly:
      1. Check for BOM (highest priority).
      2. If XML: honor <?xml ... encoding="..."> declaration.
      3. Try UTF-8 strictly.
      4. Try Latin-1 strictly.
      5. Fallback: UTF-8 with replacement characters (never crash).
    """
    # 1) BOM
    enc = _sniff_bom_encoding(data)
    if enc:
        try:
            return data.decode(enc)
        except Exception:
            pass

    # 2) XML declaration
    looks_xml = file_suffix.lower() == ".xml" or data.lstrip().startswith(b"<")
    if looks_xml:
        enc2 = _sniff_xml_decl_encoding(data)
        if enc2:
            try:
                return data.decode(enc2, errors="strict")
            except Exception:
                pass  # fallback if the declaration is wrong

    # 3) Standard attempts
    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc, errors="strict")
        except Exception:
            continue

    # 4) Last resort
    return data.decode("utf-8", errors="replace")


def read_text_any(path: Path) -> str:
    """Read a file into text with tolerant encoding detection (BOM, XML decl, fallbacks)."""
    data = path.read_bytes()
    return decode_bytes_best_effort(data, file_suffix=path.suffix)
