"""Unit tests for markdown repair utilities."""

from utils.markdown_repair import close_open_markdown


class TestCloseOpenMarkdownCodeFences:
    """Tests for closing unclosed code fences."""

    def test_unclosed_backtick_fence(self) -> None:
        """Unclosed triple-backtick fence gets closed."""
        text = "Some text\n```\ncode here"
        result = close_open_markdown(text)
        assert result == "\n```"

    def test_unclosed_tilde_fence(self) -> None:
        """Unclosed tilde fence gets closed with tildes."""
        text = "Some text\n~~~\ncode here"
        result = close_open_markdown(text)
        assert result == "\n~~~"

    def test_unclosed_fence_with_language(self) -> None:
        """Unclosed fence with language specifier gets closed."""
        text = "Some text\n```python\ndef foo():\n    pass"
        result = close_open_markdown(text)
        assert result == "\n```"

    def test_closed_fence_no_repair(self) -> None:
        """Properly closed fence needs no repair."""
        text = "Some text\n```\ncode\n```\nmore text"
        result = close_open_markdown(text)
        assert result == ""

    def test_multiple_fences_last_unclosed(self) -> None:
        """Multiple fences where only the last is unclosed."""
        text = "```\nfirst\n```\n\n```\nsecond block"
        result = close_open_markdown(text)
        assert result == "\n```"

    def test_backticks_mid_line_not_fence(self) -> None:
        """Triple backticks not at line start are not fences."""
        text = "Use ```code``` for inline code"
        result = close_open_markdown(text)
        assert result == ""

    def test_fence_no_trailing_newline(self) -> None:
        """Unclosed fence at end of text without trailing newline."""
        text = "```\ncode"
        result = close_open_markdown(text)
        assert result == "\n```"

    def test_fence_with_text_after_backticks(self) -> None:
        """Backticks at line start followed by text is a fence with info string."""
        text = "```this is my sentence"
        result = close_open_markdown(text)
        assert result == "\n```"

    def test_shorter_fence_inside_longer_fence_not_closer(self) -> None:
        """A 3-backtick line inside a 4-backtick fence is content, not a closer."""
        text = "````python\ndef foo():\n```\n    pass"
        result = close_open_markdown(text)
        assert result == "\n````"


class TestCloseOpenMarkdownHtmlTags:
    """Tests for closing unclosed HTML block tags."""

    def test_unclosed_div(self) -> None:
        """Single unclosed div tag gets closed."""
        text = "<div>\nsome content"
        result = close_open_markdown(text)
        assert result == "\n</div>"

    def test_unclosed_table(self) -> None:
        """Single unclosed table tag gets closed."""
        text = "<table>\n<tr><td>cell</td></tr>"
        result = close_open_markdown(text)
        assert result == "\n</table>"

    def test_unclosed_pre(self) -> None:
        """Single unclosed pre tag gets closed."""
        text = "<pre>\nformatted text"
        result = close_open_markdown(text)
        assert result == "\n</pre>"

    def test_multiple_unclosed_tags_reversed(self) -> None:
        """Multiple unclosed tags are closed in reverse order."""
        text = "<div>\n<table>\ncontent"
        result = close_open_markdown(text)
        assert result == "\n</table>\n</div>"

    def test_nested_inner_closed(self) -> None:
        """Nested tags where inner is closed but outer is not."""
        text = "<div>\n<table>\ncontent\n</table>"
        result = close_open_markdown(text)
        assert result == "\n</div>"

    def test_properly_closed_tags_no_repair(self) -> None:
        """Properly closed tags need no repair."""
        text = "<div>\ncontent\n</div>"
        result = close_open_markdown(text)
        assert result == ""

    def test_self_closing_tag_no_repair(self) -> None:
        """Self-closing tags do not generate closing tags."""
        text = "text with <br/> and <img src='x' />"
        result = close_open_markdown(text)
        assert result == ""

    def test_partial_html_tag_ignored(self) -> None:
        """Incomplete HTML tag (no closing >) is ignored."""
        text = "text with <div"
        result = close_open_markdown(text)
        assert result == ""

    def test_inline_tags_ignored(self) -> None:
        """Inline tags like span, em, strong are not repaired."""
        text = "<span>some <em>text"
        result = close_open_markdown(text)
        assert result == ""


class TestCloseOpenMarkdownCombinations:
    """Tests for combinations of constructs."""

    def test_html_inside_unclosed_fence_ignored(self) -> None:
        """HTML tags inside an unclosed code fence are literal text."""
        text = "```\n<div>\n<table>"
        result = close_open_markdown(text)
        assert result == "\n```"

    def test_html_before_unclosed_fence(self) -> None:
        """HTML opened before a code fence is still tracked."""
        text = "<div>\nsome text\n```\ncode"
        result = close_open_markdown(text)
        assert result == "\n```\n</div>"

    def test_closed_fence_then_unclosed_html(self) -> None:
        """Closed code fence followed by unclosed HTML."""
        text = "```\ncode\n```\n<div>\ncontent"
        result = close_open_markdown(text)
        assert result == "\n</div>"


class TestCloseOpenMarkdownEdgeCases:
    """Tests for edge cases."""

    def test_empty_string(self) -> None:
        """Empty input returns empty suffix."""
        assert close_open_markdown("") == ""

    def test_plain_text(self) -> None:
        """Plain text with no markdown constructs."""
        assert close_open_markdown("Hello world\nThis is plain text.") == ""

    def test_whitespace_only(self) -> None:
        """Whitespace-only input returns empty suffix."""
        assert close_open_markdown("   \n  \n") == ""

    def test_inline_formatting_only(self) -> None:
        """Bold, italic, links do not trigger repair."""
        text = "Some **bold** and *italic* and [link](url)"
        assert close_open_markdown(text) == ""
