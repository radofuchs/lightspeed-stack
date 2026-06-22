"""Utilities for repairing truncated markdown content.

Used when a streaming response is interrupted mid-content to close
any open markdown constructs (code fences, HTML block tags) that
would otherwise break rendering.
"""

import re
from typing import Final

BLOCK_HTML_TAGS: Final[frozenset[str]] = frozenset(
    {
        "div",
        "table",
        "tr",
        "td",
        "th",
        "thead",
        "tbody",
        "details",
        "summary",
        "pre",
    }
)

_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"^(\s{0,3})((`{3,})|(~{3,}))")
_TAG_RE: Final[re.Pattern[str]] = re.compile(r"<(/?)(\w+)([^>]*?)(/?)>")


def _process_html_tags(line: str, html_stack: list[str]) -> None:
    """Update *html_stack* with block-level HTML open/close tags found in *line*.

    Parameters:
        line: A single line of text to scan for HTML tags.
        html_stack: Mutable stack tracking open block-level tags.
    """
    for tag_match in _TAG_RE.finditer(line):
        is_closing = tag_match.group(1) == "/"
        tag_name = tag_match.group(2).lower()
        is_self_closing = tag_match.group(4) == "/"

        if tag_name not in BLOCK_HTML_TAGS or is_self_closing:
            continue

        if is_closing:
            if html_stack and html_stack[-1] == tag_name:
                html_stack.pop()
        else:
            html_stack.append(tag_name)


def close_open_markdown(text: str) -> str:
    """Return a suffix that closes any open markdown constructs in *text*.

    Scans for unclosed fenced code blocks and unclosed HTML block-level
    tags.  Returns only the closing characters (callers append the result).
    Returns an empty string when nothing needs closing.

    Parameters:
        text: Partial markdown content that may contain open constructs.

    Returns:
        A suffix string to append that closes open constructs.
    """
    if not text or not text.strip():
        return ""

    lines = text.split("\n")
    in_code_fence = False
    fence_char = ""
    fence_len = 0
    html_stack: list[str] = []

    for line in lines:
        fence_match = _FENCE_RE.match(line)
        if not fence_match:
            if not in_code_fence:
                _process_html_tags(line, html_stack)
            continue

        group_3 = fence_match.group(3)
        group_4 = fence_match.group(4)
        matched_group = group_3 or group_4
        char = "`" if group_3 else "~"
        if not in_code_fence:
            in_code_fence = True
            fence_char = char
            fence_len = len(matched_group)
        elif (
            char == fence_char
            and len(matched_group) >= fence_len
            and line[fence_match.end() :].strip(" \t") == ""
        ):
            in_code_fence = False
            fence_char = ""
            fence_len = 0

    suffix_parts: list[str] = []
    if in_code_fence:
        suffix_parts.append(f"\n{fence_char * fence_len}")

    for tag in reversed(html_stack):
        suffix_parts.append(f"\n</{tag}>")

    return "".join(suffix_parts)
