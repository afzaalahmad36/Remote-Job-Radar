import re

import pytest

from app.extraction.text_hash import normalize_post_text, raw_text_hash


def test_whitespace_is_trimmed_and_collapsed() -> None:
    assert (
        normalize_post_text("  Python\t dev \n\n at  Acme\u00a0Corp  ") == "Python dev at Acme Corp"
    )


def test_case_is_kept() -> None:
    assert normalize_post_text("Go Developer") != normalize_post_text("GO DEVELOPER")
    assert raw_text_hash("Go Developer") != raw_text_hash("go developer")


def test_posts_that_differ_only_in_whitespace_share_a_hash() -> None:
    assert raw_text_hash("Hiring:\nPython dev") == raw_text_hash("  Hiring: Python   dev\n")


def test_other_changes_give_another_hash() -> None:
    assert raw_text_hash("Python dev") != raw_text_hash("Python devs")


def test_the_hash_is_64_hex_characters_and_stable() -> None:
    value = raw_text_hash("hello world")

    assert re.fullmatch(r"[0-9a-f]{64}", value)
    assert value == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


@pytest.mark.parametrize("empty", ["", "   ", "\n\t"])
def test_empty_text_hashes_the_empty_string(empty: str) -> None:
    assert raw_text_hash(empty) == raw_text_hash("")
