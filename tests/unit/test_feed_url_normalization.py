import pytest

from app.services.source_service import normalize_feed_url

BASE = "https://example.com/feed.xml"


@pytest.mark.parametrize(
    "variant",
    [
        "https://example.com/feed.xml",
        "https://EXAMPLE.com/feed.xml",
        "HTTPS://Example.Com/feed.xml",
        "https://example.com/feed.xml/",
        "https://example.com/feed.xml//",
        "https://example.com/feed.xml#latest",
        "https://example.com/feed.xml/#latest",
        "https://example.com:443/feed.xml",
        "https://example.com./feed.xml",
        "  https://example.com/feed.xml  ",
    ],
)
def test_variants_of_the_same_feed_share_a_key(variant: str) -> None:
    assert normalize_feed_url(variant) == normalize_feed_url(BASE)


@pytest.mark.parametrize(
    "different",
    [
        "http://example.com/feed.xml",  # http and https are different URLs
        "https://example.com/Feed.xml",  # paths are case-sensitive
        "https://example.com/other.xml",
        "https://example.com/feed.xml?tag=python",  # a query often selects a different feed
        "https://example.com:8443/feed.xml",
        "https://feeds.example.com/feed.xml",
        "https://example.org/feed.xml",
    ],
)
def test_genuinely_different_feeds_get_different_keys(different: str) -> None:
    assert normalize_feed_url(different) != normalize_feed_url(BASE)


def test_a_bare_domain_and_its_root_path_are_the_same_feed() -> None:
    assert normalize_feed_url("https://example.com") == normalize_feed_url("https://example.com/")


def test_the_key_keeps_the_query_string() -> None:
    assert normalize_feed_url("https://example.com/feed/?tag=python#x") == (
        "https://example.com/feed?tag=python"
    )


def test_a_malformed_url_does_not_raise() -> None:
    assert normalize_feed_url("http://example.com:abc/feed") == "http://example.com:abc/feed"
