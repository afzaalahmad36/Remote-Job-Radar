"""The English check, on invented posts only (tests/language_posts.py).

The threshold was fixed from these texts, never from stored jobs; this test pins the margin so a
change to the word list or the threshold has to be a conscious one.
"""

import re

import pytest

from app.matching.scoring import (
    ENGLISH_MIN_RATIO,
    ENGLISH_MIN_WORDS,
    ENGLISH_WORDS,
    looks_english,
)
from tests.language_posts import ENGLISH_POSTS, ENGLISH_STUB, NON_ENGLISH_POSTS


def ratio(text: str) -> float:
    words = re.findall(r"[^\W\d_]+", text[:1800].lower())
    return sum(w in ENGLISH_WORDS for w in words) / len(words)


@pytest.mark.parametrize("name", sorted(ENGLISH_POSTS))
def test_english_posts_look_english(name: str) -> None:
    assert looks_english(ENGLISH_POSTS[name])


@pytest.mark.parametrize("name", sorted(NON_ENGLISH_POSTS))
def test_german_french_spanish_and_portuguese_posts_do_not(name: str) -> None:
    assert not looks_english(NON_ENGLISH_POSTS[name])


def test_a_short_text_gives_no_verdict_and_counts_as_english() -> None:
    assert looks_english("Ingeniero backend remoto")
    assert looks_english("")
    assert len(ENGLISH_STUB.split()) >= ENGLISH_MIN_WORDS  # the stub is judged on its words
    assert looks_english(ENGLISH_STUB)


def test_the_threshold_sits_in_the_gap_between_the_two_groups() -> None:
    lowest_english = min(ratio(t) for t in [*ENGLISH_POSTS.values(), ENGLISH_STUB])
    highest_other = max(ratio(t) for t in NON_ENGLISH_POSTS.values())

    assert highest_other < ENGLISH_MIN_RATIO < lowest_english
    # Documented in docs/MATCHING.md: the worst English post is 0.267, the best other one 0.013.
    assert lowest_english > 0.25
    assert highest_other < 0.02


def test_only_the_start_of_the_post_is_judged() -> None:
    english_start = ENGLISH_POSTS["prose"]
    long_german_tail = " ".join([NON_ENGLISH_POSTS["german prose"]] * 40)

    assert looks_english(english_start + "\n" + long_german_tail)


def test_the_word_list_has_no_word_that_other_target_languages_use_everywhere() -> None:
    for common in ("a", "in", "an", "on", "as", "de", "la", "le", "der", "die", "und"):
        assert common not in ENGLISH_WORDS
