import pytest

from app.extraction.plan import ExtractionError, failed_plan, plan_rules_phase
from app.extraction.text_hash import raw_text_hash
from app.models import ExtractionStatus, LlmEnrichment

MIN_CHARS = 500

RICH = (
    "Senior Python Developer at Acme\n\nWe are hiring a Senior Python Developer for our remote "
    "team.\nStack: Python, Django, PostgreSQL, Docker.\nApply: https://acme.com/careers/senior"
)
# A clear title but no technology the dictionary knows.
TITLE_ONLY_SHORT = "Marketing Manager at Acme\n\nWe are hiring a Marketing Manager."
TITLE_ONLY_LONG = TITLE_ONLY_SHORT + "\n\n" + "We build great products for happy customers. " * 20
# Nothing the rules recognise as a title or a technology.
NOTHING_SHORT = "lorem ipsum dolor sit amet " * 4
NOTHING_LONG = "lorem ipsum dolor sit amet " * 40

EXTRACTED, PENDING, FAILED = (
    ExtractionStatus.EXTRACTED,
    ExtractionStatus.PENDING,
    ExtractionStatus.EXTRACTION_FAILED,
)
NOT_NEEDED, WAITING, SKIPPED = (
    LlmEnrichment.NOT_NEEDED,
    LlmEnrichment.PENDING,
    LlmEnrichment.SKIPPED,
)


def test_the_test_posts_have_the_lengths_the_cases_rely_on() -> None:
    assert len(TITLE_ONLY_SHORT) < MIN_CHARS <= len(TITLE_ONLY_LONG)
    assert len(NOTHING_SHORT) < MIN_CHARS <= len(NOTHING_LONG)


@pytest.mark.parametrize(
    ("text", "can_enrich", "status", "enrichment", "error"),
    [
        # Title and stack found: nothing for a model to do, whether or not one exists.
        (RICH, False, EXTRACTED, NOT_NEEDED, None),
        (RICH, True, EXTRACTED, NOT_NEEDED, None),
        # A title but no stack: usable now; waits for a model (also when none is switched on yet).
        (TITLE_ONLY_LONG, True, EXTRACTED, WAITING, None),
        (TITLE_ONLY_LONG, False, EXTRACTED, WAITING, None),
        # ...unless the post is too short to be worth a request.
        (TITLE_ONLY_SHORT, True, EXTRACTED, SKIPPED, ExtractionError.TOO_SHORT),
        # No title: the model may still find one...
        (NOTHING_LONG, True, PENDING, WAITING, None),
        # ...but if it cannot be asked, fail at once with a reason.
        (NOTHING_LONG, False, FAILED, SKIPPED, ExtractionError.MODEL_NOT_ENABLED),
        (NOTHING_SHORT, True, FAILED, SKIPPED, ExtractionError.TOO_SHORT),
        (NOTHING_SHORT, False, FAILED, SKIPPED, ExtractionError.TOO_SHORT),
        ("", True, FAILED, SKIPPED, ExtractionError.TOO_SHORT),
    ],
)
def test_the_state_table(
    text: str,
    can_enrich: bool,
    status: ExtractionStatus,
    enrichment: LlmEnrichment,
    error: ExtractionError | None,
) -> None:
    plan = plan_rules_phase(text, can_enrich=can_enrich, min_chars=MIN_CHARS)

    assert (plan.status, plan.enrichment, plan.error) == (status, enrichment, error)


def test_an_extracted_plan_always_has_a_title_because_the_database_requires_it() -> None:
    for text in (RICH, TITLE_ONLY_SHORT, TITLE_ONLY_LONG, NOTHING_SHORT, NOTHING_LONG, ""):
        for can_enrich in (True, False):
            plan = plan_rules_phase(text, can_enrich=can_enrich, min_chars=MIN_CHARS)

            assert plan.status is not EXTRACTED or plan.values.title is not None


def test_the_plan_carries_the_rules_values_and_the_text_hash() -> None:
    plan = plan_rules_phase(RICH, can_enrich=False, min_chars=MIN_CHARS)

    assert plan.values.title == "Senior Python Developer"
    assert plan.values.stack == ("Python", "Django", "PostgreSQL", "Docker")
    assert plan.text_hash == raw_text_hash(RICH)


def test_a_gate_of_zero_lets_a_short_post_wait_for_the_model() -> None:
    plan = plan_rules_phase(NOTHING_SHORT, can_enrich=True, min_chars=0)

    assert (plan.status, plan.enrichment) == (PENDING, WAITING)


def test_the_failed_plan_is_a_failure_that_a_retry_can_revive() -> None:
    plan = failed_plan(RICH)

    assert (plan.status, plan.enrichment, plan.error) == (
        FAILED,
        SKIPPED,
        ExtractionError.UNEXPECTED,
    )
    assert plan.values.title is None and plan.values.stack == ()
    assert plan.text_hash == raw_text_hash(RICH)


def test_stored_error_texts_are_a_small_fixed_set_that_fits_the_column() -> None:
    assert {error.value for error in ExtractionError} == {
        "post too short for the model",
        "no title found; the model is not enabled",
        "unexpected error",
    }
    assert all(len(error.value) <= 500 for error in ExtractionError)
