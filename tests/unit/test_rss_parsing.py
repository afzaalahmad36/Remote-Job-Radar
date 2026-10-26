import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.ingestion.base import FetchError
from app.ingestion.rss import MAX_ENTRIES, MAX_EXTERNAL_ID_LENGTH, _posted_at, parse_feed
from app.ingestion.text import MAX_TEXT_CHARS
from tests.fakes import feed_bytes


def rss(*items: str) -> bytes:
    body = "".join(f"<item>{item}</item>" for item in items)
    return (
        '<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>'
        f"<link>https://example.com/</link><description>d</description>{body}</channel></rss>"
    ).encode()


# --- identity: GUID, then link, then a hash ------------------------------------------------------


def test_the_guid_is_the_external_id() -> None:
    _, jobs = parse_feed(feed_bytes("rss2_basic.xml"))

    assert [job.external_id for job in jobs] == ["job-101", "job-102", "job-103"]


def test_atom_entry_ids_are_used() -> None:
    _, jobs = parse_feed(feed_bytes("atom_basic.xml"))

    assert [job.external_id for job in jobs] == [
        "tag:example.org,2026:job-a1",
        "tag:example.org,2026:job-a2",
    ]


def test_without_a_guid_the_link_is_used_then_a_hash() -> None:
    _, jobs = parse_feed(feed_bytes("no_guid.xml"))

    assert jobs[0].external_id == "https://example.com/jobs/linked-1"
    assert jobs[1].external_id.startswith("sha256:")
    assert jobs[2].external_id.startswith("sha256:")
    assert jobs[1].external_id != jobs[2].external_id


def test_the_fallback_hash_is_stable_between_runs() -> None:
    first = parse_feed(feed_bytes("no_guid.xml"))[1]
    second = parse_feed(feed_bytes("no_guid.xml"))[1]

    assert [j.external_id for j in first] == [j.external_id for j in second]


def test_the_fallback_hash_depends_on_title_and_date() -> None:
    def entry(title: str, date: str) -> bytes:
        return rss(f"<title>{title}</title><pubDate>{date}</pubDate><description>x</description>")

    monday, tuesday = "Mon, 26 Oct 2026 10:00:00 GMT", "Tue, 27 Oct 2026 10:00:00 GMT"
    feeds = (entry("Role", monday), entry("Other", monday), entry("Role", tuesday))

    ids = {parse_feed(feed)[1][0].external_id for feed in feeds}

    assert len(ids) == 3


def test_an_empty_guid_falls_back_to_the_link() -> None:
    feed = rss("<title>T</title><guid>   </guid><link>https://example.com/a</link>")

    assert parse_feed(feed)[1][0].external_id == "https://example.com/a"


def test_surrounding_whitespace_in_a_guid_is_ignored() -> None:
    assert parse_feed(rss("<title>T</title><guid>  abc  </guid>"))[1][0].external_id == "abc"


def test_an_id_too_long_for_the_column_is_replaced_by_its_hash() -> None:
    long_guid = "https://example.com/" + "x" * 400

    job = parse_feed(rss(f"<title>T</title><guid>{long_guid}</guid>"))[1][0]

    assert job.external_id.startswith("sha256:")
    assert len(job.external_id) <= MAX_EXTERNAL_ID_LENGTH


# --- duplicates ----------------------------------------------------------------------------------


def test_repeated_entries_inside_one_feed_are_stored_once_and_the_first_wins() -> None:
    fetched, jobs = parse_feed(feed_bytes("duplicates.xml"))

    assert fetched == 5  # the feed returned five entries...
    assert [job.external_id for job in jobs] == [  # ...but only three distinct posts
        "dup-1",
        "https://example.com/jobs/same-link",
        "unique-1",
    ]
    assert jobs[0].raw_text.startswith("Original post\n\nFirst time")


# --- text ----------------------------------------------------------------------------------------


def test_html_in_descriptions_becomes_plain_text() -> None:
    _, jobs = parse_feed(feed_bytes("html_descriptions.xml"))

    first = jobs[0].raw_text
    assert first == (
        "R&D Engineer (Remote)\n\nAbout the role\n\n"
        "Work on exciting things & more.\n\nLine one\n\nLine two"
    )
    for fragment in ("<", ">", "alert", "display:none", "onclick", "onerror", "steal"):
        assert fragment not in first


def test_full_content_is_preferred_over_the_teaser() -> None:
    _, jobs = parse_feed(feed_bytes("html_descriptions.xml"))

    assert "The full job description" in jobs[1].raw_text
    assert "teaser" not in jobs[1].raw_text


def test_atom_content_is_preferred_over_the_summary() -> None:
    _, jobs = parse_feed(feed_bytes("atom_basic.xml"))

    assert "build distributed systems in Go" in jobs[0].raw_text
    assert "Short summary" not in jobs[0].raw_text


def test_raw_text_is_the_title_then_the_body() -> None:
    _, jobs = parse_feed(feed_bytes("rss2_basic.xml"))

    assert (
        jobs[2].raw_text == "DevOps Engineer at Initech\n\nKubernetes, Terraform and a great team."
    )


def test_an_entry_with_only_a_title_keeps_just_the_title() -> None:
    assert (
        parse_feed(rss("<title>Only a title</title><guid>1</guid>"))[1][0].raw_text
        == "Only a title"
    )


def test_an_entry_with_no_text_at_all_is_skipped() -> None:
    fetched, jobs = parse_feed(rss("<guid>empty</guid>", "<title>Real</title><guid>real</guid>"))

    assert (fetched, [j.external_id for j in jobs]) == (2, ["real"])


def test_the_cap_is_20_000_characters_per_entry() -> None:
    assert MAX_TEXT_CHARS == 20_000


def test_very_long_text_is_cut_to_the_cap() -> None:
    feed = rss(f"<title>T</title><guid>big</guid><description>{'word ' * 50_000}</description>")

    assert len(parse_feed(feed)[1][0].raw_text) == MAX_TEXT_CHARS


def test_the_cap_covers_title_and_body_together() -> None:
    feed = rss(
        f"<title>{'T' * 100_000}</title><guid>big</guid>"
        f"<description>{'word ' * 50_000}</description>"
    )

    assert len(parse_feed(feed)[1][0].raw_text) == MAX_TEXT_CHARS


def test_text_under_the_cap_is_kept_whole() -> None:
    body = "word " * 1000  # 5,000 characters

    job = parse_feed(rss(f"<title>T</title><guid>g</guid><description>{body}</description>"))[1][0]

    assert job.raw_text == f"T\n\n{body.strip()}"


# --- dates ---------------------------------------------------------------------------------------


def test_publication_dates_are_converted_to_utc() -> None:
    _, jobs = parse_feed(feed_bytes("rss2_basic.xml"))

    assert [job.posted_at for job in jobs] == [
        datetime(2026, 10, 26, 12, 30, tzinfo=UTC),  # 14:30 +0200
        datetime(2026, 10, 27, 9, 0, tzinfo=UTC),  # GMT
        datetime(2026, 10, 28, 13, 15, tzinfo=UTC),  # 08:15 -0500
    ]
    assert all(job.posted_at is not None and job.posted_at.utcoffset() is not None for job in jobs)


def test_atom_dates_are_converted_and_published_beats_updated() -> None:
    _, jobs = parse_feed(feed_bytes("atom_basic.xml"))

    assert jobs[0].posted_at == datetime(2026, 10, 26, 9, 0, tzinfo=UTC)  # updated, 10:00 +01:00
    assert jobs[1].posted_at == datetime(2026, 10, 25, 22, 0, tzinfo=UTC)  # published 18:00 -04:00


def test_an_impossible_date_that_slips_past_feedparser_gives_none() -> None:
    year_100000 = (100_000, 1, 1, 0, 0, 0, 0, 1, 0)  # a struct_time shape datetime cannot hold

    assert _posted_at({"published_parsed": year_100000}) is None


def test_a_missing_date_gives_none() -> None:
    _, jobs = parse_feed(feed_bytes("no_guid.xml"))

    assert jobs[2].posted_at is None


@pytest.mark.parametrize("date", ["not a date", "Mon, 99 Foo 2026 99:99:99 GMT", "", "0000-00-00"])
def test_an_unparseable_date_gives_none_and_does_not_drop_the_post(date: str) -> None:
    feed = rss(
        f"<title>T</title><guid>g</guid><pubDate>{date}</pubDate><description>x</description>"
    )

    job = parse_feed(feed)[1][0]

    assert job.posted_at is None


# --- not a feed, hostile feeds, limits --------------------------------------------------------


@pytest.mark.parametrize("name", ["malformed.xml", "not_a_feed.html"])
def test_input_that_is_not_a_feed_is_an_error(name: str) -> None:
    with pytest.raises(FetchError, match="Not a valid RSS or Atom feed"):
        parse_feed(feed_bytes(name))


@pytest.mark.parametrize("body", [b"", b"   ", b"\x00\x01\x02", b'{"json": true}'])
def test_empty_and_binary_input_is_an_error(body: bytes) -> None:
    with pytest.raises(FetchError):
        parse_feed(body)


def test_a_valid_feed_with_no_entries_is_fine() -> None:
    assert parse_feed(feed_bytes("empty.xml")) == (0, [])


def test_a_truncated_feed_keeps_the_entries_that_arrived_intact() -> None:
    full = feed_bytes("rss2_basic.xml")
    cut = full[: full.index(b"Frontend Developer") + 30]

    _, jobs = parse_feed(cut)

    assert [job.external_id for job in jobs] == ["job-101"]


def test_at_most_max_entries_are_read() -> None:
    items = [
        f"<title>Job {i}</title><guid>g{i}</guid><description>x</description>"
        for i in range(MAX_ENTRIES + 50)
    ]

    fetched, jobs = parse_feed(rss(*items))

    assert (fetched, len(jobs)) == (MAX_ENTRIES, MAX_ENTRIES)
    assert jobs[-1].external_id == f"g{MAX_ENTRIES - 1}"


def test_an_entity_expansion_bomb_is_not_expanded() -> None:
    started = time.perf_counter()

    _, jobs = parse_feed(feed_bytes("billion_laughs.xml"))

    assert time.perf_counter() - started < 2  # the bomb would take minutes and gigabytes
    assert "lol" * 3 not in jobs[0].raw_text
    assert len(jobs[0].raw_text) < 200


def test_external_entities_are_never_resolved() -> None:
    _, jobs = parse_feed(feed_bytes("xxe.xml"))

    text = jobs[0].raw_text
    assert "[fonts]" not in text.lower()  # contents of C:/Windows/win.ini
    assert "ami-id" not in text  # what the cloud metadata endpoint would answer
    assert "169.254" not in text


def test_a_body_that_is_a_local_file_path_is_not_opened(tmp_path: Path) -> None:
    # feedparser.parse() opens a str/bytes argument as a file if it can. We pass a file-like
    # object instead, so a hostile server cannot make us read local files by sending a path.
    local_feed = tmp_path / "local.xml"
    local_feed.write_bytes(rss("<title>From disk</title><guid>local</guid>"))

    with pytest.raises(FetchError):
        parse_feed(str(local_feed).encode())


def test_a_body_that_is_a_url_is_not_fetched(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("feedparser tried to fetch a URL")

    monkeypatch.setattr("feedparser.http.get", forbidden)

    with pytest.raises(FetchError):
        parse_feed(b"http://169.254.169.254/latest/meta-data/")
