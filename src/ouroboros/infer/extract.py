"""Turn an uploaded document into the plain text the detector should score.

Typesetting is not authorship. A PDF hard-wraps every line at the page width
and hyphenates across lines; a DOCX stores paragraphs as XML runs. Both are
reduced to the text the author wrote: one line per paragraph, paragraphs
separated by a blank line.
"""

from __future__ import annotations

import re
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree

SUPPORTED = (".pdf", ".docx", ".txt", ".md")

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def reflow(text: str) -> str:
    """Join the lines of each paragraph; keep paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\f", "\n\n")
    text = re.sub(r"\n\s*\d+\s*/\s*\d+\s*\n", "\n\n", text)  # page numbers "1 / 3"
    text = re.sub(r"(\w)-\n\s*(\w)", r"\1-\2", text)  # line break after a hyphen
    paragraphs = re.split(r"\n\s*\n", text)
    out = []
    for p in paragraphs:
        line = " ".join(part.strip() for part in p.split("\n") if part.strip())
        line = re.sub(r"[ \t]{2,}", " ", line)
        if line:
            out.append(line)
    return "\n\n".join(out)


def pdf_text(path: str | Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, check=True
    )
    return reflow(result.stdout)


def docx_text(path: str | Path) -> str:
    """Paragraph text of a .docx, read straight from word/document.xml."""
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs = []
    for p in root.iter(f"{_W}p"):
        parts = []
        for node in p.iter():
            if node.tag == f"{_W}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{_W}tab":
                parts.append(" ")
            elif node.tag in (f"{_W}br", f"{_W}cr"):
                parts.append(" ")
        line = re.sub(r"\s{2,}", " ", "".join(parts)).strip()
        if line:
            paragraphs.append(line)
    return "\n\n".join(paragraphs)


def extract_text(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return pdf_text(path)
    if suffix == ".docx":
        return docx_text(path)
    if suffix in (".txt", ".md"):
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    raise ValueError(f"unsupported file type {suffix!r}; expected one of {', '.join(SUPPORTED)}")


# --------------------------------------------------------------------------
# layout normalization for pasted text
# --------------------------------------------------------------------------
_LIST_ITEM = re.compile(r"^\s*(?:[-*•·▪◦–]|\d{1,3}[.)]|[a-zA-Z][.)])\s+")
_SENT_END = re.compile(r"[.!?…:;»\"”)]\s*$")
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀ-ÖØ-Þ«\"“(])")


def _unwrap_block(lines: list[str], width: int) -> list[str]:
    """Join hard-wrapped lines back into paragraphs.

    A line is a paragraph end when it closes a sentence and stops well short
    of the wrap width, or when the next line is a list item. Headings (short,
    unpunctuated, followed by a capital) stay on their own.
    """
    out: list[str] = []
    current = ""
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if _LIST_ITEM.match(line) and current:
            out.append(current)
            current = ""
        if current and current.endswith("-") and line[:1].islower():
            current = current + line  # "celui-" + "là"
        else:
            current = f"{current} {line}".strip()
        short = len(line) < 0.8 * width
        ends = bool(_SENT_END.search(line))
        heading = short and not ends and len(line) < 0.6 * width and nxt[:1].isupper() and len(line.split()) <= 12
        item_end = bool(_LIST_ITEM.match(current)) and short
        if not nxt or (short and ends) or heading or item_end or _LIST_ITEM.match(nxt):
            out.append(current)
            current = ""
    if current:
        out.append(current)
    return out


def _split_flat(paragraph: str, sentences_per_paragraph: int = 4) -> list[str]:
    """Cut one very long unbroken paragraph at sentence boundaries."""
    sentences = _SENT_SPLIT.split(paragraph)
    if len(sentences) <= sentences_per_paragraph * 2:
        return [paragraph]
    return [
        " ".join(sentences[i : i + sentences_per_paragraph])
        for i in range(0, len(sentences), sentences_per_paragraph)
    ]


def normalize_layout(text: str, split_flat: bool = True) -> str:
    """Canonical layout: one line per paragraph, blank line between paragraphs.

    The detector learned layout cues that are not authorship: hard-wrapped or
    flattened text reads as human, airy paragraphs as AI. Every input is brought
    to the same canonical shape so only the wording is left to judge. Text that
    is already in that shape comes back unchanged.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return text.strip()
    lengths = sorted(len(l.strip()) for l in lines)
    # wrap width: robust high percentile, or the longest line when there are few
    width = lengths[-1] if len(lengths) < 10 else lengths[int(0.9 * (len(lengths) - 1))]
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block_lines = [l for l in block.split("\n") if l.strip()]
        if not block_lines:
            continue
        if len(block_lines) == 1:
            paragraphs.append(re.sub(r"[ \t]{2,}", " ", block_lines[0].strip()))
        else:
            paragraphs.extend(_unwrap_block(block_lines, width))
    if split_flat:
        paragraphs = [p for para in paragraphs for p in (_split_flat(para) if len(para.split()) > 180 else [para])]
    paragraphs = [re.sub(r"[ \t]{2,}", " ", p).strip() for p in paragraphs if p.strip()]
    out = ""
    for i, para in enumerate(paragraphs):
        if i:
            # consecutive list items stay a list (single newline), as markdown writes them
            both_items = _LIST_ITEM.match(para) and _LIST_ITEM.match(paragraphs[i - 1])
            out += "\n" if both_items else "\n\n"
        out += para
    return out


def looks_hard_wrapped(text: str, min_lines: int = 4) -> bool:
    """True when most lines stop at a common width without ending a sentence:
    the signature of a PDF / e-mail hard wrap, not of an author's line breaks."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < min_lines:
        return False
    lengths = sorted(len(l) for l in lines)
    width = lengths[int(0.9 * (len(lengths) - 1))]
    if width < 40:
        return False
    wrapped = sum(1 for l in lines[:-1] if len(l) >= 0.75 * width and not _SENT_END.search(l))
    return wrapped / max(1, len(lines) - 1) >= 0.4


def unwrap(text: str, paragraph_sep: str = "\n\n") -> str:
    """Minimal fix: re-join hard-wrapped lines, change nothing else.

    Text that is not hard-wrapped is returned unchanged -- rewriting clean text
    (e.g. turning single newlines into paragraphs) moves human text towards the
    layout the model associates with AI, which costs false positives.
    """
    if not looks_hard_wrapped(text):
        return text
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [l for l in text.split("\n") if l.strip()]
    lengths = sorted(len(l.strip()) for l in lines)
    width = lengths[-1] if len(lengths) < 10 else lengths[int(0.9 * (len(lengths) - 1))]
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block_lines = [l for l in block.split("\n") if l.strip()]
        if block_lines:
            paragraphs.extend(_unwrap_block(block_lines, width))
    return paragraph_sep.join(re.sub(r"[ \t]{2,}", " ", q).strip() for q in paragraphs)
