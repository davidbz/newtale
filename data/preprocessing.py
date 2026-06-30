from __future__ import annotations

import re
import unicodedata

import xxhash

_HTML_TAG_RE = re.compile(r"<[^>]{0,200}>")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_RE.sub("", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text


def length_filter(text: str) -> bool:
    n = len(text)
    return 200 <= n <= 100_000


def strip_html(text: str) -> str | None:
    html_chars = sum(len(m.group()) for m in _HTML_TAG_RE.finditer(text))
    if len(text) > 0 and html_chars / len(text) > 0.20:
        return None
    return _HTML_TAG_RE.sub("", text)


def repetition_filter(text: str) -> bool:
    if len(text) < 5:
        return True
    grams: dict[str, int] = {}
    total = 0
    for i in range(len(text) - 4):
        gram = text[i : i + 5]
        grams[gram] = grams.get(gram, 0) + 1
        total += 1
    if total == 0:
        return True
    top_count = max(grams.values())
    return (top_count / total) <= 0.3


class ExactDedup:
    def __init__(self, max_entries: int = 500_000) -> None:
        self._seen: set[str] = set()
        self._max_entries = max_entries

    def is_duplicate(self, text: str) -> bool:
        h = xxhash.xxh64(text.encode()).hexdigest()
        if h in self._seen:
            return True
        if len(self._seen) < self._max_entries:
            self._seen.add(h)
        return False


def preprocess(text: str, dedup: ExactDedup | None = None) -> str | None:
    text = normalize_text(text)
    if not length_filter(text):
        return None
    stripped = strip_html(text)
    if stripped is None:
        return None
    if not repetition_filter(stripped):
        return None
    if dedup is not None and dedup.is_duplicate(stripped):
        return None
    return stripped
