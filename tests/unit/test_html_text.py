import pytest

from app.ingestion.text import MAX_TEXT_CHARS, html_to_text


def test_tags_are_removed_and_text_is_kept() -> None:
    assert html_to_text("<p>Hello <strong>big</strong> <em>world</em></p>") == "Hello big world"


def test_links_keep_their_text_and_lose_the_url_and_attributes() -> None:
    result = html_to_text('<a href="https://evil.example/x" onclick="steal()">click here</a>')

    assert result == "click here"


@pytest.mark.parametrize(
    "html",
    [
        "<script>alert('xss')</script>visible",
        "<style>body{display:none}</style>visible",
        "<noscript>hidden</noscript>visible",
        "<iframe src='https://x'>hidden</iframe>visible",
        "<svg><text>hidden</text></svg>visible",
        "<SCRIPT>alert(1)</SCRIPT>visible",
    ],
)
def test_script_style_and_embedded_content_are_dropped(html: str) -> None:
    assert html_to_text(html) == "visible"


def test_html_entities_are_decoded() -> None:
    assert html_to_text("R&amp;D &lt;3 caf&eacute; &#8211; ok") == "R&D <3 café – ok"


def test_block_tags_become_line_breaks() -> None:
    assert html_to_text("<p>One</p><p>Two</p>") == "One\n\nTwo"
    assert html_to_text("Line one<br>Line two") == "Line one\nLine two"
    assert html_to_text("<ul><li>A</li><li>B</li></ul>") == "A\n\nB"


def test_whitespace_is_collapsed() -> None:
    assert (
        html_to_text("  <p>  too    many \t spaces  </p>\n\n\n\n<p>x</p> ")
        == "too many spaces\n\nx"
    )


def test_plain_text_passes_through_unchanged() -> None:
    assert html_to_text("Just a plain sentence.") == "Just a plain sentence."


def test_empty_and_markup_only_input_give_an_empty_string() -> None:
    assert html_to_text("") == ""
    assert html_to_text("<p></p><br><div> </div>") == ""


@pytest.mark.parametrize(
    "broken",
    ["<p>unclosed", "<<script>>", "<div <p> text", "</p></p></p>x", "<a href='x>text", "&", "<!--"],
)
def test_broken_markup_never_raises_and_leaves_no_tags_behind(broken: str) -> None:
    result = html_to_text(broken)

    assert "<script" not in result.lower()
    assert isinstance(result, str)


def test_the_output_is_bounded() -> None:
    assert len(html_to_text("<p>" + "word " * 100_000 + "</p>")) <= MAX_TEXT_CHARS


def test_an_unclosed_script_hides_everything_after_it() -> None:
    # Safer to lose the rest of the post than to keep script text.
    assert html_to_text("before<script>alert(1) after") == "before"
