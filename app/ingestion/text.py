import re
from html.parser import HTMLParser

MAX_TEXT_CHARS = 20_000  # per post: bounds what one feed entry can store

# Tags that end a line, so "<p>One</p><p>Two</p>" becomes two lines instead of "OneTwo".
_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "tr", "table", "blockquote", "pre",
    "h1", "h2", "h3", "h4", "h5", "h6", "hr", "section", "article", "header", "footer",
}  # fmt: skip
_SKIPPED_TAGS = {"script", "style", "noscript", "template", "head", "iframe", "object", "svg"}
_SPACES = re.compile(r"[^\S\n]+")  # runs of whitespace other than newlines
_BLANK_LINES = re.compile(r"\n{3,}")


class _TextExtractor(HTMLParser):
    """Collects the visible text of an HTML fragment and throws the markup away.

    This only *reads* HTML to get text out; the result is stored as plain text and is never
    rendered as HTML, so nothing here needs to be "sanitized HTML".
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)  # &amp; &#39; etc. become real characters
        self._parts: list[str] = []
        self._skip_depth = 0  # >0 while inside <script>, <style>, ...

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def html_to_text(html: str) -> str:
    """Plain text from an HTML fragment (or from text that is not HTML at all)."""
    extractor = _TextExtractor()
    extractor.feed(html[: MAX_TEXT_CHARS * 5])  # markup is bulky: look at a bounded amount of it
    extractor.close()
    text = _SPACES.sub(" ", extractor.text())
    lines = (line.strip() for line in text.split("\n"))
    text = _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()
    return text[:MAX_TEXT_CHARS]
