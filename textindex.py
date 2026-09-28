"""The offset contract.

Every character span this project reports or replaces refers to **the decoded
logical string at a given JSON Pointer** -- never to a byte offset in the
serialized file.

Why this needs stating: ``tool.result`` is a JSON string, so a literal ``\\n``
inside it is two bytes in the file and one character in the value. If the
detector reported file offsets and the sanitizer consumed value offsets, every
downstream span would drift and the replay fidelity tests would fail silently.
Artifacts are therefore always rewritten wholesale through canonical JSON
serialization; nothing is ever byte-patched in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

Json = Any


# --- JSON Pointer (RFC 6901) -----------------------------------------------


def pointer_escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def pointer_unescape(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def join_pointer(base: str, token: str | int) -> str:
    return f"{base}/{pointer_escape(str(token))}"


def resolve_pointer(doc: Json, pointer: str) -> Json:
    """Resolve an RFC 6901 pointer. Empty string denotes the whole document."""
    if pointer in ("", "/"):
        return doc
    if not pointer.startswith("/"):
        raise ValueError(f"pointer must be empty or start with '/': {pointer!r}")
    node = doc
    for raw_token in pointer.split("/")[1:]:
        token = pointer_unescape(raw_token)
        if isinstance(node, list):
            node = node[int(token)]
        elif isinstance(node, dict):
            node = node[token]
        else:
            raise KeyError(f"cannot descend into {type(node).__name__} at {token!r}")
    return node


# --- JSONPath-flavoured expression (RFC 9535 style, display only) ----------


def join_path(base: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{base}[{key}]"
    if key.isidentifier():
        return f"{base}.{key}"
    return f"{base}[{key!r}]"


_LEAF_TYPES = (str, int, float, bool, type(None))


def iter_leaves(
    doc: Json, *, base_pointer: str = "", base_path: str = "$"
) -> Iterator[tuple[str, str, Json]]:
    """Yield ``(pointer, path_expr, value)`` for every scalar leaf in ``doc``."""
    if isinstance(doc, dict):
        for key, value in doc.items():
            yield from iter_leaves(
                value,
                base_pointer=join_pointer(base_pointer, key),
                base_path=join_path(base_path, str(key)),
            )
    elif isinstance(doc, list):
        for index, value in enumerate(doc):
            yield from iter_leaves(
                value,
                base_pointer=join_pointer(base_pointer, index),
                base_path=join_path(base_path, index),
            )
    elif isinstance(doc, _LEAF_TYPES):
        yield base_pointer, base_path, doc
    # Non-JSON values are ignored rather than guessed at.


# --- Decoded-string indexing ----------------------------------------------


@dataclass(frozen=True)
class Location:
    """A span within a decoded string, plus its human-readable position."""

    char_start: int
    char_end: int
    line: int
    col: int
    end_line: int
    end_col: int

    def as_dict(self) -> dict[str, int]:
        return {
            "char_start": self.char_start,
            "char_end": self.char_end,
            "line": self.line,
            "col": self.col,
            "end_line": self.end_line,
            "end_col": self.end_col,
        }


class TextIndex:
    """Offset bookkeeping for one decoded logical string.

    Line/column numbers are 1-based and computed on the decoded value, which is
    what a human reading the rendered artifact would see. ``raw_hint`` may carry
    a serialized-form offset for debugging, but it is never load-bearing.
    """

    __slots__ = ("text", "_line_starts")

    def __init__(self, text: str) -> None:
        self.text = text
        starts = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                starts.append(i + 1)
        self._line_starts: Sequence[int] = starts

    def __len__(self) -> int:
        return len(self.text)

    def _line_col(self, offset: int) -> tuple[int, int]:
        # Binary search for the last line start <= offset.
        lo, hi = 0, len(self._line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1, offset - self._line_starts[lo] + 1

    def locate(self, start: int, end: int) -> Location:
        if not (0 <= start <= end <= len(self.text)):
            raise ValueError(f"span ({start}, {end}) out of range for text of length {len(self.text)}")
        line, col = self._line_col(start)
        end_line, end_col = self._line_col(end)
        return Location(start, end, line, col, end_line, end_col)

    def slice(self, start: int, end: int) -> str:
        return self.text[start:end]

    def find_all(self, needle: str, *, limit: int = 1000) -> list[tuple[int, int]]:
        """All non-overlapping occurrences of ``needle`` as ``(start, end)``."""
        if not needle:
            return []
        spans: list[tuple[int, int]] = []
        cursor = 0
        while len(spans) < limit:
            found = self.text.find(needle, cursor)
            if found < 0:
                break
            spans.append((found, found + len(needle)))
            cursor = found + len(needle)
        return spans


def index_document(doc: Json) -> dict[str, TextIndex]:
    """Build a ``{pointer: TextIndex}`` map for every string leaf in ``doc``."""
    out: dict[str, TextIndex] = {}
    for pointer, _, value in iter_leaves(doc):
        if isinstance(value, str):
            out[pointer] = TextIndex(value)
    return out
