"""`scripts/eval_fix_labels.py`: the deterministic label fixer.

Every example is invented. The database read is replaced, so no database, network or provider is
used, and the fixer never touches a stored or recomputed extraction.
"""

import ast
import json
import uuid
from collections.abc import Mapping
from pathlib import Path

import pytest

from app.extraction.stack import spellings
from app.extraction.text_hash import raw_text_hash
from app.models import RemotePolicy, SalaryPeriod, Seniority
from scripts import eval_fix_labels as fix
from scripts.db_guard import ScriptError
from scripts.eval_fix_labels import (
    Change,
    Override,
    PostFix,
    Review,
    Session,
    company_spellings,
    entry_names,
    fix_post,
    format_changes,
    format_counts,
    format_lists,
    format_reviews,
    load_overrides,
    overrides_kept,
    periods_near,
    plan_fixes,
    run_fix,
    seniority_evidence,
    seniority_from_title,
    split_headline,
    verify_only_allowed_changes,
)
from scripts.eval_labelfile import LabelEntry, LabelFile, Labels, load_label_file, save_label_file
from tests.settings_helpers import settings_with

ROOT = Path(__file__).resolve().parent.parent.parent
TEXT = (
    "Acme Corp: Senior Backend Engineer\n\n"
    "We use Python and k8s. Pay: $90,000 - $120,000 per year. Fully remote."
)
SETTINGS = settings_with(database_url="postgresql+asyncpg://u:secret@db:5432/jobradar")
NO_ALIASES: Mapping[str, tuple[str, ...]] = {}


def messy(**changes: object) -> Labels:
    """Labels with the mistakes the fixer corrects, for TEXT."""
    values: dict[str, object] = {
        "title": "Acme Corp: Senior Backend Engineer",
        "company": "Acme",
        "stack": ["Python", "Kubernetes", "Rust"],
        "remote_policy": RemotePolicy.REMOTE,
        "seniority": Seniority.LEAD,
        "salary_min": 90000,
        "salary_max": 130000,
        "salary_currency": "USD",
        "salary_period": SalaryPeriod.MONTH,
    }
    return Labels(**{**values, **changes})  # type: ignore[arg-type]


def fixed(
    labels: Labels, text: str = TEXT, aliases: Mapping[str, tuple[str, ...]] | None = None
) -> PostFix:
    return fix_post(1, labels, text, NO_ALIASES if aliases is None else aliases)


# --- rule 1: title and company -------------------------------------------------------------------


def test_the_headline_is_the_first_non_empty_line() -> None:
    assert split_headline("\n  \n  Acme: Backend Engineer  \nbody: more") == (
        "Acme",
        "Backend Engineer",
    )


@pytest.mark.parametrize(
    "company", ["A·Team", "Abnormal AI", "ZoomInfo", "Acme Corp", "O'Neil & Co"]
)
def test_the_companys_own_spelling_is_kept_character_for_character(company: str) -> None:
    assert split_headline(f"{company}: Staff Engineer") == (company, "Staff Engineer")


@pytest.mark.parametrize(
    ("headline", "problem"),
    [
        ("Backend Engineer at Acme", "exactly one"),
        ("Acme:Engineer", "exactly one"),
        ("Acme: Staff Engineer: Platform", "exactly one"),
        ("One Two Three Four Five: Engineer", "1 to 4 words"),
        ("Acme 2: Engineer", "digit"),
        ("Acme: Engineer ($90k)", "'$' or a '('"),
        ("Acme: Engineer (Remote)", "'$' or a '('"),
        ("Acme: Engineer FULL TIME", "full time"),
        ("Acme: Engineer, part-time", "full time"),
        ("Acme: Hiring Engineer", "hiring"),
        ("Acme: " + "x" * 70, "70 characters"),
        (": Engineer", "1 to 4 words"),
    ],
)
def test_a_headline_that_fails_a_check_is_not_split(headline: str, problem: str) -> None:
    assert split_headline(headline) is None
    assert problem in (fix.headline_problem(headline) or "")


def test_the_limits_are_inclusive_of_four_words_and_69_characters() -> None:
    assert split_headline("A B C D: " + "x" * 69) == ("A B C D", "x" * 69)


@pytest.mark.parametrize("text", ["", "   \n ", "Acme: "])
def test_a_post_without_a_usable_headline_cannot_be_split(text: str) -> None:
    assert split_headline(text) is None
    assert fix.headline_problem(text.strip() or None) is not None


def test_title_and_company_come_from_the_headline() -> None:
    result = fixed(messy())

    assert result.labels.title == "Senior Backend Engineer"
    assert result.labels.company == "Acme Corp"
    assert [(c.field, c.before, c.after) for c in result.changes[:2]] == [
        ("title", "Acme Corp: Senior Backend Engineer", "Senior Backend Engineer"),
        ("company", "Acme", "Acme Corp"),
    ]
    assert {c.rule for c in result.changes[:2]} == {fix.RULE_HEADLINE}


def test_a_blank_company_label_is_filled_from_the_headline() -> None:
    assert fixed(messy(company=None)).labels.company == "Acme Corp"


def test_an_unsplittable_headline_changes_neither_field_and_needs_a_human() -> None:
    result = fixed(messy(title="Backend Dev", company="Acme"), "Backend Engineer at Acme\n\nbody")

    assert (result.labels.title, result.labels.company) == ("Backend Dev", "Acme")
    assert not [c for c in result.changes if c.rule == fix.RULE_HEADLINE]
    [review] = [r for r in result.reviews if r.field == "title, company"]
    assert review.current == "title: Backend Dev; company: Acme"
    assert "exactly one" in review.shows
    assert "headline: Backend Engineer at Acme" in review.shows  # the text is in the table


def test_a_company_the_label_and_the_post_spell_longer_is_not_shortened() -> None:
    text = "Abnormal: Engineer\n\nAbnormal AI builds email security."

    result = fixed(messy(company="Abnormal AI"), text)

    assert result.labels.company == "Abnormal AI"
    assert result.labels.title == "Engineer"


def test_a_longer_company_label_the_post_never_spells_is_replaced() -> None:
    result = fixed(messy(company="Abnormal AI"), "Abnormal: Engineer\n\nAbnormal builds things.")

    assert result.labels.company == "Abnormal"


def test_a_longer_label_that_is_a_different_word_is_replaced() -> None:
    result = fixed(messy(company="Acmeville"), "Acme: Engineer\n\nAcmeville is a town.")

    assert result.labels.company == "Acme"


def test_rules_after_the_headline_use_the_current_title_when_it_was_not_split() -> None:
    result = fixed(messy(title="Junior Dev", seniority=None), "Backend Engineer at Acme\n\nbody")

    assert result.labels.seniority == Seniority.JUNIOR


# --- rule 2: seniority ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Senior Backend Engineer", "senior"),
        ("SENIOR engineer", "senior"),
        ("Intern, Data", "intern"),
        ("Mid-level Developer", "mid"),
        ("Principal Engineer", "principal"),
        ("Lead Engineer", "lead"),
        ("Senior Senior Engineer", "senior"),
        ("Junior Developer", "junior"),
        ("Staff Engineer", None),
        ("Engineering Manager", None),
        ("Director of Engineering", None),
        ("Head of Platform", None),
        ("Senior Lead Engineer", None),
        ("Internal Tools Engineer", None),
        ("Backend Engineer", None),
        (None, None),
    ],
)
def test_seniority_is_the_one_enum_word_in_the_title(
    title: str | None, expected: str | None
) -> None:
    assert seniority_from_title(title) == expected


def test_seniority_is_computed_from_the_final_title() -> None:
    result = fixed(messy(title="Senior Backend Engineer", seniority=Seniority.LEAD))

    assert result.labels.seniority == Seniority.SENIOR
    assert ("seniority", "lead", "senior", fix.RULE_SENIORITY) in [
        (c.field, c.before, c.after, c.rule) for c in result.changes
    ]
    assert result.reviews == []


def test_a_title_edit_that_drops_a_level_word_keeps_the_label_and_needs_a_human() -> None:
    # The label title says "Senior ..."; the headline gives a title without it.
    result = fixed(
        messy(title="Senior Engineer", seniority=Seniority.SENIOR),
        "Acme: Platform Engineer\n\nbody",
    )

    assert result.labels.seniority == Seniority.SENIOR  # not blanked
    [review] = [r for r in result.reviews if r.field == "seniority"]
    assert review.current == "senior"
    assert "senior" in review.shows and "Platform Engineer" in review.shows
    assert not [c for c in result.changes if c.field == "seniority"]


@pytest.mark.parametrize("word", ["Senior", "Staff", "Principal", "Lead", "Junior"])
def test_every_guarded_level_word_is_kept_when_the_edit_drops_it(word: str) -> None:
    result = fixed(
        messy(title=f"{word} Engineer", seniority=Seniority.MID), "Acme: Platform Engineer\n\nb"
    )

    assert result.labels.seniority == Seniority.MID
    assert [r.field for r in result.reviews] == ["seniority"]


def test_a_title_without_a_level_word_still_clears_a_stale_seniority() -> None:
    result = fixed(
        messy(title="Platform Engineer", seniority=Seniority.LEAD), "Acme: Platform Engineer\n\nb"
    )

    assert result.labels.seniority is None  # the original title had no level word either
    assert result.reviews == []


def test_a_level_word_that_survives_the_edit_is_not_a_review() -> None:
    result = fixed(messy(title="Acme: Senior Platform Engineer"), "Acme: Senior Platform Engineer")

    assert result.labels.seniority == Seniority.SENIOR
    assert result.reviews == []


# --- rule 3: salary ------------------------------------------------------------------------------


def test_an_amount_not_in_the_text_is_removed_and_listed_and_the_other_is_kept() -> None:
    result = fixed(messy())

    assert (result.labels.salary_min, result.labels.salary_max) == (90000, None)
    assert (fix.SALARY_REMOVED, "salary_max is not in the text") in result.flags
    assert ("salary_max", "130000", "(none)", fix.RULE_AMOUNT) in [
        (c.field, c.before, c.after, c.rule) for c in result.changes
    ]


@pytest.mark.parametrize("amount", ["$90,000", "90k", "90 000", "90.000"])
def test_amounts_are_found_with_separators_and_a_k_suffix(amount: str) -> None:
    text = f"Acme: Engineer\n\nPay {amount} per year."

    result = fixed(messy(salary_min=90000, salary_max=None), text)

    assert result.labels.salary_min == 90000
    assert not result.flags


@pytest.mark.parametrize(
    ("phrase", "period"),
    [
        ("per hour", "hour"),
        ("hourly", "hour"),
        ("$90/hr", "hour"),
        ("per year", "year"),
        ("annual", "year"),
        ("annually", "year"),
        ("yearly", "year"),
        ("a year", "year"),
        ("$90/yr", "year"),
        ("per month", "month"),
        ("monthly", "month"),
        ("per week", "week"),
        ("weekly", "week"),
        ("per day", "day"),
        ("daily", "day"),
    ],
)
def test_the_period_comes_from_the_word_next_to_the_amount(phrase: str, period: str) -> None:
    text = f"Acme: Engineer\n\nPay 90000 {phrase}."
    if phrase.startswith("$"):
        text = f"Acme: Engineer\n\nPay 90000{phrase[3:]}."

    result = fixed(messy(salary_min=90000, salary_max=None, salary_period=None), text)

    assert result.labels.salary_period is not None
    assert result.labels.salary_period.value == period


def period_of(text: str) -> str | None:
    result = fixed(messy(salary_min=90000, salary_max=None, salary_period=None), text)
    return result.labels.salary_period.value if result.labels.salary_period else None


def test_a_period_word_is_found_within_sixty_characters_first() -> None:
    assert period_of("Acme: Engineer\n\n90000 " + "x" * 10 + " per year") == "year"


def test_a_period_word_up_to_two_hundred_characters_away_is_found_too() -> None:
    assert period_of("Acme: Engineer\n\n90000 " + "x" * 150 + " per year") == "year"
    assert period_of("Acme: Engineer\n\nper month " + "x" * 150 + " 90000") == "month"


def test_a_period_word_beyond_two_hundred_characters_is_not_used() -> None:
    assert period_of("Acme: Engineer\n\n90000. " + "x" * 300 + " per year") is None


def test_a_period_word_in_the_same_sentence_is_found_beyond_two_hundred_characters() -> None:
    sentence = "Pay is 90000 " + "and so on " * 30 + "paid yearly."

    assert period_of("Acme: Engineer\n\n" + sentence) == "year"


def test_a_period_word_in_another_sentence_or_line_is_not_the_amounts() -> None:
    far = "x" * 300
    assert period_of("Acme: Engineer\n\nPay 90000. " + far + " Paid yearly.") is None
    assert period_of("Acme: Engineer\n\nPay 90000\n" + far + " paid yearly") is None


def test_the_near_period_wins_over_a_farther_different_one() -> None:
    text = "Acme: Engineer\n\nPay 90000 per year. " + "x" * 100 + " Contractors: 50 per hour."

    assert period_of(text) == "year"


@pytest.mark.parametrize("text", ["HR Manager 90000", "90000 for 5 years", "90000 in 4 weeks"])
def test_hr_the_department_and_plural_durations_are_not_periods(text: str) -> None:
    assert periods_near({90000}, text.casefold()) == set()


def test_a_period_label_with_no_period_word_near_the_amount_is_cleared() -> None:
    result = fixed(messy(salary_min=90000, salary_max=None), "Acme: Engineer\n\nPay 90000 flat.")

    assert result.labels.salary_period is None
    assert ("salary_period", "month", "(none)", fix.RULE_PERIOD) in [
        (c.field, c.before, c.after, c.rule) for c in result.changes
    ]


def test_two_different_periods_near_the_amounts_leave_the_period_and_need_a_human() -> None:
    text = "Acme: Engineer\n\n90000 per year, or 50 per hour. Overtime 90000 daily"
    result = fixed(messy(salary_min=90000, salary_max=None, salary_period=SalaryPeriod.YEAR), text)

    assert result.labels.salary_period == SalaryPeriod.YEAR
    [review] = [r for r in result.reviews if r.field == "salary_period"]
    assert review.current == "year"
    assert "day, hour, year" in review.shows  # the periods that compete
    assert "90000 per year, or 50 per hour. Overtime 90000 daily" in review.shows  # the line


def test_with_no_amount_left_currency_and_period_go_too() -> None:
    result = fixed(messy(salary_min=70000, salary_max=80000), "Acme: Engineer\n\nCompetitive pay.")

    labels = result.labels
    assert (labels.salary_min, labels.salary_max) == (None, None)
    assert (labels.salary_currency, labels.salary_period) == (None, None)
    assert {c.field: c.rule for c in result.changes if c.field.startswith("salary")} == {
        "salary_min": fix.RULE_AMOUNT,
        "salary_max": fix.RULE_AMOUNT,
        "salary_currency": fix.RULE_NO_AMOUNTS,
        "salary_period": fix.RULE_NO_AMOUNTS,
    }


def test_a_stray_currency_without_amounts_is_cleared_without_a_listing() -> None:
    result = fixed(
        messy(salary_min=None, salary_max=None, salary_period=None), "Acme: Engineer\n\nbody"
    )

    assert result.labels.salary_currency is None
    assert not [flag for flag in result.flags if flag[0] == fix.SALARY_REMOVED]


# --- rule 4: stack -------------------------------------------------------------------------------


def test_an_entry_missing_from_the_text_is_removed_and_the_rest_keep_their_order() -> None:
    result = fixed(messy(stack=["Rust", "Python", "Kubernetes", "Go"]))

    assert result.labels.stack == ["Python", "Kubernetes"]
    assert [(c.before, c.after, c.rule) for c in result.changes if c.field == "stack"] == [
        ("Rust", "(removed; no partial match)", fix.RULE_STACK),
        ("Go", "(removed; no partial match)", fix.RULE_STACK),
    ]


@pytest.mark.parametrize(
    ("entry", "text"),
    [
        ("Next.js", "We build with NextJS."),
        ("NextJS", "We build with Next.js."),
        ("Next.js", "We build with next js."),
        ("Node.js", "We run Node on servers."),
        ("Node", "We run node.js on servers."),
        ("PostgreSQL", "We store data in Postgres."),
        ("Vue.js", "A Vue app."),
        ("GitHub Actions", "We use github-actions."),
        ("Tailwind CSS", "TailwindCSS styling."),
        ("C++", "Experience with C++."),
        ("Apache Kafka", "We run Kafka."),
        ("Scikit-Learn", "We use scikit learn."),
        ("Kubernetes", "We run on K8S."),
    ],
)
def test_broader_matching_ignores_dots_hyphens_spaces_and_case(entry: str, text: str) -> None:
    result = fixed(messy(stack=[entry]), "Acme: Engineer\n\n" + text)

    assert result.labels.stack == [entry]
    assert not [c for c in result.changes if c.field == "stack"]


def test_the_broader_matching_is_still_whole_word() -> None:
    result = fixed(
        messy(stack=["Java", "Go", "Node"]), "Acme: Engineer\n\nJavaScript, good, nodes."
    )

    assert result.labels.stack == []


def test_each_removed_entry_says_whether_a_partial_match_exists() -> None:
    text = "Acme: Engineer\n\nWe use JavaScript and the Google ecosystem."
    result = fixed(messy(stack=["Java", "Google Cloud Platform", "Rust"]), text)

    notes = {c.before: c.after for c in result.changes if c.field == "stack"}
    assert notes == {
        "Java": "(removed; partial match: 'java' inside a longer word)",
        "Google Cloud Platform": "(removed; partial match: word 'google')",
        "Rust": "(removed; no partial match)",
    }


def test_up_to_three_removals_are_applied() -> None:
    result = fixed(messy(stack=["Python", "Rust", "Zig", "Nim"]))

    assert result.labels.stack == ["Python"]
    assert [r for r in result.reviews if r.field == "stack"] == []


def test_more_than_three_removals_remove_none_and_need_a_human() -> None:
    entries = ["Python", "Rust", "Zig", "Nim", "Crystal"]
    text = "Acme: Engineer\n\nWe use Python and Rustacean tools."

    result = fixed(messy(stack=entries), text)

    assert result.labels.stack == entries  # nothing removed
    assert not [c for c in result.changes if c.field == "stack"]
    [review] = [r for r in result.reviews if r.field == "stack"]
    assert review.current == "Rust; Zig; Nim; Crystal"
    assert "4 entries are not found; none removed" in review.shows
    assert "Rust: partial match: 'rust' inside a longer word" in review.shows
    assert "Zig: no partial match" in review.shows


def test_a_dictionary_alias_keeps_the_entry() -> None:
    # TEXT says "k8s", not "Kubernetes"; "Postgres" is a dictionary spelling of PostgreSQL
    text = TEXT + " Also postgres."

    assert fixed(messy(stack=["Kubernetes", "PostgreSQL"]), text).labels.stack == [
        "Kubernetes",
        "PostgreSQL",
    ]


def test_a_policy_alias_keeps_the_entry() -> None:
    text = "Acme: Engineer\n\nWe use Acme DB daily."  # the post never writes "AcmeDatabase"
    aliases = {"acme db": ("AcmeDatabase",)}

    assert fixed(messy(stack=["AcmeDatabase"]), text, aliases).labels.stack == ["AcmeDatabase"]
    assert fixed(messy(stack=["AcmeDatabase"]), text).labels.stack == []


def test_matching_is_by_whole_word_and_ignores_case() -> None:
    text = "Acme: Engineer\n\nWe use JavaScript and PYTHON."

    assert fixed(messy(stack=["Java", "python", "Script"]), text).labels.stack == ["python"]


def test_the_fixer_never_adds_a_stack_entry() -> None:
    assert fixed(messy(stack=[])).labels.stack == []


def test_a_name_for_several_products_is_kept_when_one_of_them_is_in_the_text() -> None:
    aliases = {"react vite": ("React", "Vite")}
    text = "Acme: Engineer\n\nWe use vite."

    assert fixed(messy(stack=["React Vite"]), text, aliases).labels.stack == ["React Vite"]


def test_an_alias_of_one_product_does_not_make_its_sibling_count() -> None:
    aliases = {"react vite": ("React", "Vite")}

    assert "vite" not in entry_names("React", aliases)
    assert "vite" in entry_names("React Vite", aliases)


def test_entry_names_include_dictionary_spellings_and_policy_aliases() -> None:
    aliases = {"google cloud platform": ("GCP",)}

    names = entry_names("Google Cloud Platform", aliases)

    assert {"google cloud platform", "gcp", "google cloud"} <= names
    assert entry_names("Mystery Tool", aliases) == {"mystery tool"}


def test_the_stack_dictionary_lists_the_spellings_of_a_known_technology() -> None:
    assert {"Kubernetes", "kubernetes", "k8s"} == set(spellings("k8s"))
    assert spellings("Mystery Tool") == ()


# --- rule 5: untouched, and the whole file -------------------------------------------------------


def test_remote_policy_is_never_touched() -> None:
    assert (
        fixed(messy(remote_policy=RemotePolicy.ONSITE)).labels.remote_policy == RemotePolicy.ONSITE
    )
    assert fixed(messy(remote_policy=None)).labels.remote_policy is None


def test_labels_that_already_follow_the_rules_produce_no_changes() -> None:
    clean = messy(
        title="Senior Backend Engineer",
        company="Acme Corp",
        stack=["Python", "Kubernetes"],
        seniority=Seniority.SENIOR,
        salary_max=120000,
        salary_period=SalaryPeriod.YEAR,
    )

    result = fixed(clean)

    assert result.changes == [] and result.flags == []
    assert result.labels == clean


def entry(labels: Labels, text: str = TEXT, job: int = 1) -> LabelEntry:
    return LabelEntry(
        job_id=uuid.UUID(int=job),
        raw_text_hash=raw_text_hash(text),
        status="labelled",
        labels=labels,
    )


def skipped(job: int) -> LabelEntry:
    return LabelEntry(job_id=uuid.UUID(int=job), raw_text_hash=raw_text_hash("s"), status="skipped")


def label_file(*entries: LabelEntry) -> LabelFile:
    return LabelFile(
        source_id=uuid.UUID(int=99),
        seed=1,
        selected=[e.job_id for e in entries],
        posts=list(entries),
    )


def texts(**by_job: str) -> dict[uuid.UUID, str]:
    return {uuid.UUID(int=int(job.removeprefix("j"))): text for job, text in by_job.items()}


def test_the_plan_numbers_labelled_posts_and_keeps_skipped_ones_as_they_are() -> None:
    skip = skipped(5)
    original = label_file(skip, entry(messy(), job=1), entry(messy(), job=2))

    plan = plan_fixes(original, texts(j1=TEXT, j2=TEXT), NO_ALIASES)

    assert plan.fixed.posts[0] == skip
    assert {c.post for c in plan.changes} == {1, 2}
    assert plan.fixed.posts[1].job_id == uuid.UUID(int=1)
    verify_only_allowed_changes(original, plan.fixed)


def test_a_missing_or_stale_post_is_listed_and_not_touched() -> None:
    changed = TEXT + " (edited)"
    original = label_file(entry(messy(), job=1), entry(messy(), job=2))

    plan = plan_fixes(original, {uuid.UUID(int=2): changed}, NO_ALIASES)

    assert plan.changes == []
    assert plan.fixed.posts == original.posts
    assert [(r.post, r.field, r.shows) for r in plan.reviews] == [
        (1, "(whole post)", "post not found"),
        (2, "(whole post)", "post text changed since it was labelled"),
    ]


def test_posts_with_an_empty_stack_are_listed_as_info() -> None:
    original = label_file(entry(messy(stack=["Rust"]), job=1), entry(messy(), job=2))

    plan = plan_fixes(original, texts(j1=TEXT, j2=TEXT), NO_ALIASES)

    assert plan.lists[fix.INFO] == [(1, "stack is empty")]


def test_posts_that_end_up_with_identical_labels_are_listed_but_left_alone() -> None:
    original = label_file(
        entry(messy(), job=1), entry(messy(title="Other"), job=2), entry(messy(), job=3)
    )

    plan = plan_fixes(original, texts(j1=TEXT, j2=TEXT, j3=TEXT), NO_ALIASES)

    assert plan.lists[fix.IDENTICAL] == [(1, "same labels as posts 2, 3")]
    assert [e.labels for e in plan.fixed.posts][0] == [e.labels for e in plan.fixed.posts][1]


# --- the safety check ----------------------------------------------------------------------------


def with_labels(original: LabelFile, labels: Labels) -> LabelFile:
    post = original.posts[0].model_copy(update={"labels": labels})
    return original.model_copy(update={"posts": [post, *original.posts[1:]]})


def test_the_verification_accepts_changes_to_the_fixable_fields_only() -> None:
    original = label_file(entry(messy()))
    changed = messy(title="T", company="C", seniority=None, stack=["Python"], salary_min=None)

    verify_only_allowed_changes(original, with_labels(original, changed))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"remote_policy": RemotePolicy.HYBRID}, "outside the fixable ones"),
        ({"stack": ["Python", "Brand New"]}, "added or reordered"),
        ({"stack": ["Kubernetes", "Python"]}, "added or reordered"),
    ],
)
def test_the_verification_refuses_other_changes(changes: dict[str, object], message: str) -> None:
    original = label_file(entry(messy()))

    with pytest.raises(ScriptError, match=message):
        verify_only_allowed_changes(original, with_labels(original, messy(**changes)))


def test_the_verification_refuses_header_id_hash_status_and_count_changes() -> None:
    original = label_file(entry(messy(), job=1), skipped(2))
    first = original.posts[0]

    cases = [
        (original.model_copy(update={"seed": 2}), "header"),
        (original.model_copy(update={"posts": [first]}), "number of posts"),
        (
            original.model_copy(
                update={
                    "posts": [first.model_copy(update={"job_id": uuid.UUID(int=7)}), skipped(2)]
                }
            ),
            "id, hash or status",
        ),
        (
            original.model_copy(
                update={"posts": [first.model_copy(update={"raw_text_hash": "0" * 64}), skipped(2)]}
            ),
            "id, hash or status",
        ),
        (
            original.model_copy(
                update={"posts": [first, skipped(2).model_copy(update={"labels": messy()})]}
            ),
            "skipped post gained labels",
        ),
    ]
    for changed, message in cases:
        with pytest.raises(ScriptError, match=message):
            verify_only_allowed_changes(original, changed)


# --- the table and the lists ---------------------------------------------------------------------


def test_the_table_has_a_header_and_one_row_per_change() -> None:
    lines = format_changes([Change(3, "title", "Acme: Dev", "Dev", fix.RULE_HEADLINE)])

    assert lines[0].split(" | ")[0].strip() == "post"
    assert [cell.strip() for cell in lines[2].split(" | ")] == [
        "3",
        "title",
        "Acme: Dev",
        "Dev",
        fix.RULE_HEADLINE,
    ]


def test_counts_are_given_for_every_rule_even_when_zero() -> None:
    lines = format_counts(
        [Change(1, "stack", "Rust", "(removed; no partial match)", fix.RULE_STACK)]
    )

    assert len(lines) == len(fix.RULES)
    assert "  rule 4 not in text: 1 change(s) in 1 post(s)" in lines
    assert "  rule 1 headline: 0 change(s) in 0 post(s)" in lines


def test_the_lists_are_printed_by_name_and_post() -> None:
    lines = format_lists({fix.INFO: [(4, "stack is empty")], fix.SALARY_REMOVED: [(2, "why")]})

    assert lines == ["  SALARY_AMOUNT_REMOVED: post 2: why", "  INFO: post 4: stack is empty"]


def test_the_needs_human_table_has_the_four_columns_and_is_sorted_by_post() -> None:
    lines = format_reviews(
        [
            Review(7, "stack", "A; B", "4 entries are not found"),
            Review(2, "title, company", "title: T; company: C", "headline: X: Y: Z"),
        ]
    )

    cells = [
        [cell.strip() for cell in line.split(" | ")] for line in (lines[0], lines[2], lines[3])
    ]
    assert cells == [
        ["post", "field", "current label", "what the post shows"],
        ["2", "title, company", "title: T; company: C", "headline: X: Y: Z"],
        ["7", "stack", "A; B", "4 entries are not found"],
    ]


def test_long_lines_shown_in_the_table_are_cut() -> None:
    text = "Acme: Engineer\n\n" + "90000 per year " + "x" * 500
    amounts = fix.lines_with_amounts(text, {90000})

    assert len(amounts) == 1 and amounts[0].endswith("…") and len(amounts[0]) <= 301


def test_the_command_prints_the_needs_human_table_and_writes_nothing_from_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "Backend Engineer at Acme\n\nWe use Python. Pay $90,000 per year."
    path = tmp_path / "labels.json"
    labels = messy(title="Backend Dev", company="Acme", seniority=None, stack=["Python"])
    save_label_file(path, label_file(entry(labels, text)))
    fake_reader(monkeypatch, {1: text})
    lines: list[str] = []

    code = run_fix(path, SETTINGS, session("y", lines))

    output = "\n".join(lines)
    assert "NEEDS_HUMAN: 1 item(s); nothing in that table is written" in output
    assert "current label" in output and "headline: Backend Engineer at Acme" in output
    written = load_label_file(path).posts[0].labels
    assert written is not None and (written.title, written.company) == ("Backend Dev", "Acme")
    assert code in (0, 1)


# --- the command ---------------------------------------------------------------------------------


def fake_reader(monkeypatch: pytest.MonkeyPatch, by_job: dict[int, str]) -> list[list[uuid.UUID]]:
    asked: list[list[uuid.UUID]] = []

    async def fake(database_url: str, ids: list[uuid.UUID]) -> list[tuple[uuid.UUID, str]]:
        asked.append(ids)
        return [(uuid.UUID(int=n), text) for n, text in by_job.items()]

    monkeypatch.setattr(fix, "read_posts_by_id", fake)
    return asked


def session(answer: str | None, lines: list[str]) -> Session:
    def ask(prompt: str) -> str:
        lines.append(f"ASKED {prompt.strip()}")
        if answer is None:
            raise EOFError
        return answer

    return Session(ask=ask, out=lines.append)


def test_the_command_backs_up_shows_asks_writes_and_rechecks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    original = label_file(entry(messy()))
    save_label_file(path, original)
    before = path.read_bytes()
    asked = fake_reader(monkeypatch, {1: TEXT})
    lines: list[str] = []

    code = run_fix(path, SETTINGS, session("y", lines))

    output = "\n".join(lines)
    assert asked == [[uuid.UUID(int=1)]]
    assert (tmp_path / "labels.clean-v1.json").read_bytes() == before
    assert "Acme Corp: Senior Backend Engineer" in output  # the table may show text
    assert "ASKED Write these changes? [y/N]" in lines
    assert "secret" not in output
    written = load_label_file(path).posts[0].labels
    assert written is not None
    assert (written.title, written.company) == ("Senior Backend Engineer", "Acme Corp")
    assert written.remote_policy == RemotePolicy.REMOTE
    assert written.stack == ["Python", "Kubernetes"]
    assert code == 0  # the re-check finds nothing wrong with the corrected labels
    assert "Checked 1 labelled posts; 0 have problems." in lines


def test_the_recheck_after_writing_prints_codes_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy()), entry(messy(), job=2)))
    fake_reader(monkeypatch, {1: TEXT, 2: TEXT})
    lines: list[str] = []

    assert run_fix(path, SETTINGS, session("y", lines)) == 1

    start = lines.index("Written. Checking the labels against the post text again (codes only):")
    report = "\n".join(lines[start + 1 :])
    assert "identical_labels (with post 2)" in report
    assert "Backend" not in report and "Acme" not in report and "Python" not in report


def test_declining_changes_nothing_but_the_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy())))
    before = path.read_bytes()
    fake_reader(monkeypatch, {1: TEXT})
    lines: list[str] = []

    for answer in ("", "n", "no", "anything"):
        assert run_fix(path, SETTINGS, session(answer, lines)) == 1
        assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["labels.clean-v1.json", "labels.json"]


def test_end_of_input_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy())))
    before = path.read_bytes()
    fake_reader(monkeypatch, {1: TEXT})
    lines: list[str] = []

    assert run_fix(path, SETTINGS, session(None, lines)) == 1
    assert path.read_bytes() == before
    assert "Input ended. Nothing was changed." in "\n".join(lines)


def test_an_existing_backup_is_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy())))
    backup = tmp_path / "labels.clean-v1.json"
    backup.write_text("older copy", encoding="utf-8")
    fake_reader(monkeypatch, {1: TEXT})
    lines: list[str] = []

    run_fix(path, SETTINGS, session("n", lines))

    assert backup.read_text(encoding="utf-8") == "older copy"
    assert any("already there, kept as it is" in line for line in lines)


def test_nothing_to_change_asks_nothing_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    clean = messy(
        title="Senior Backend Engineer",
        company="Acme Corp",
        stack=["Python"],
        seniority=Seniority.SENIOR,
        salary_max=120000,
        salary_period=SalaryPeriod.YEAR,
    )
    save_label_file(path, label_file(entry(clean)))
    before = path.read_bytes()
    fake_reader(monkeypatch, {1: TEXT})
    lines: list[str] = []

    assert run_fix(path, SETTINGS, session("y", lines)) == 0
    assert path.read_bytes() == before
    assert "Nothing to change." in lines and not any(line.startswith("ASKED") for line in lines)


def test_the_wrong_database_is_refused_before_anything_is_read_or_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy())))
    asked = fake_reader(monkeypatch, {1: TEXT})
    wrong = settings_with(database_url="postgresql+asyncpg://u:secret@db:5432/jobradar_test")

    with pytest.raises(ScriptError, match="refusing to read database"):
        run_fix(path, wrong, session("y", []))

    assert asked == []
    assert [p.name for p in tmp_path.iterdir()] == ["labels.json"]


def test_a_missing_labels_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ScriptError, match="does not exist"):
        run_fix(tmp_path / "nope.json", SETTINGS, session("y", []))


def test_main_runs_the_fixer_and_reports_errors_with_exit_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(fix, "get_settings", lambda: SETTINGS)

    assert fix.main(["--labels", str(tmp_path / "nope.json")]) == 2
    assert "error:" in capsys.readouterr().err

    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy())))
    fake_reader(monkeypatch, {1: TEXT})
    assert fix.main(["--labels", str(path)], session("n", [])) == 1


# --- per-post overrides --------------------------------------------------------------------------


def override(**values: object) -> Override:
    return Override.model_validate(values)


def test_an_override_knows_which_fields_it_sets_and_which_it_keeps() -> None:
    result = override(title="T", seniority=None, keep=["company"])

    assert result.values() == {"title": "T", "seniority": None}  # an explicit null still counts
    assert result.locked() == {"title", "seniority", "company"}
    assert override(keep=["stack"]).values() == {}


@pytest.mark.parametrize(
    "bad",
    [
        {"title": "T", "keep": ["title"]},  # set and kept
        {"salary_min": 5},  # not an overridable field
        {"keep": ["salary_min"]},
        {"stack": "Python"},
    ],
)
def test_an_invalid_override_is_refused(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        override(**bad)


def write_overrides(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_overrides_are_loaded_by_post_number(tmp_path: Path) -> None:
    path = write_overrides(
        tmp_path / "o.json",
        {"2": {"title": "T", "company": "C"}, "1": {"keep": ["company"], "seniority": None}},
    )

    loaded = load_overrides(path, 2)

    assert set(loaded) == {1, 2}
    assert loaded[2].values() == {"title": "T", "company": "C"}


def test_a_missing_overrides_file_means_no_overrides(tmp_path: Path) -> None:
    assert load_overrides(tmp_path / "nope.json", 5) == {}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([1, 2], "not a valid overrides file"),
        ({"x": {"title": "T"}}, "not a valid overrides file"),
        ({"1": {"surprise": 1}}, "not a valid overrides file"),
        ({"1": {"title": "T", "keep": ["title"]}}, "not a valid overrides file"),
        ({"0": {"title": "T"}}, "no labelled post 0"),
        ({"4": {"title": "T"}, "9": {"title": "T"}}, "no labelled post 4, 9"),
    ],
)
def test_a_bad_overrides_file_is_an_error(tmp_path: Path, data: object, message: str) -> None:
    with pytest.raises(ScriptError, match=message):
        load_overrides(write_overrides(tmp_path / "o.json", data), 3)


def test_an_unreadable_overrides_file_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "o.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ScriptError, match="not a valid overrides file"):
        load_overrides(path, 3)


def test_an_override_sets_the_fields_and_is_shown_with_the_rule_override() -> None:
    result = fixed_with(messy(), override(title="Chosen Title", company="Chosen Co"))

    assert (result.labels.title, result.labels.company) == ("Chosen Title", "Chosen Co")
    rows = [(c.field, c.before, c.after, c.rule) for c in result.changes if c.rule == "override"]
    assert rows == [
        ("title", "Acme Corp: Senior Backend Engineer", "Chosen Title", "override"),
        ("company", "Acme", "Chosen Co", "override"),
    ]
    assert not [c for c in result.changes if c.rule == fix.RULE_HEADLINE]  # the rule skipped them


def fixed_with(labels: Labels, chosen: Override, text: str = TEXT) -> PostFix:
    return fix_post(1, labels, text, NO_ALIASES, chosen)


def test_an_override_wins_over_the_headline_for_only_the_fields_it_sets() -> None:
    result = fixed_with(messy(), override(title="Chosen Title"))

    assert result.labels.title == "Chosen Title"
    assert result.labels.company == "Acme Corp"  # the headline rule still ran for the company


def test_kept_fields_are_left_exactly_as_labelled_and_get_no_row() -> None:
    result = fixed_with(messy(), override(keep=["company"]))

    assert result.labels.company == "Acme"
    assert result.labels.title == "Senior Backend Engineer"  # the other field still follows rules
    assert not [c for c in result.changes if c.field == "company"]


def test_keeping_title_and_company_skips_the_headline_check_and_its_review() -> None:
    bad_headline = "Backend Engineer at Acme\n\nbody"

    kept = fixed_with(messy(), override(keep=["title", "company"]), bad_headline)
    not_kept = fixed_with(messy(), override(keep=["title"]), bad_headline)

    assert kept.reviews == []
    assert [r.field for r in not_kept.reviews] == ["title, company"]  # company is still undecided


def test_a_set_seniority_is_final_even_null() -> None:
    result = fixed_with(
        messy(seniority=Seniority.LEAD), override(title="Applied AI Engineer", seniority=None)
    )

    assert result.labels.seniority is None
    assert ("seniority", "lead", "(none)", "override") in [
        (c.field, c.before, c.after, c.rule) for c in result.changes
    ]
    assert not [c for c in result.changes if c.rule == fix.RULE_SENIORITY]


def test_seniority_follows_the_overridden_title_when_it_is_not_set() -> None:
    result = fixed_with(
        messy(title="Engineer", seniority=None), override(title="Senior Independent Developer")
    )

    assert result.labels.seniority == Seniority.SENIOR


def test_the_seniority_guard_still_sees_a_level_word_an_override_drops() -> None:
    result = fixed_with(
        messy(title="Senior Engineer", seniority=Seniority.SENIOR), override(title="Engineer")
    )

    assert result.labels.seniority == Seniority.SENIOR
    assert [r.field for r in result.reviews] == ["seniority"]


def test_a_stack_override_replaces_the_stack_and_the_rules_skip_it() -> None:
    result = fixed_with(messy(stack=["Python", "Rust"]), override(stack=[]))

    assert result.labels.stack == []
    assert [(c.field, c.before, c.after, c.rule) for c in result.changes if c.field == "stack"] == [
        ("stack", "Python, Rust", "(none)", "override")  # one row for the whole stack
    ]
    assert not [c for c in result.changes if c.rule == fix.RULE_STACK]


def test_a_kept_stack_is_not_trimmed_even_when_entries_are_missing() -> None:
    result = fixed_with(messy(stack=["Rust", "Zig"]), override(keep=["stack"]))

    assert result.labels.stack == ["Rust", "Zig"]


def test_an_override_that_adds_a_stack_entry_is_refused_by_the_write_guard() -> None:
    original = label_file(entry(messy(stack=["Python"])))
    plan = plan_fixes(original, texts(j1=TEXT), NO_ALIASES, {1: override(stack=["Python", "Go"])})

    with pytest.raises(ScriptError, match="added or reordered"):
        verify_only_allowed_changes(original, plan.fixed)


def test_overrides_are_looked_up_by_the_posts_position_among_labelled_posts() -> None:
    skip = skipped(5)
    original = label_file(skip, entry(messy(), job=1), entry(messy(), job=2))

    plan = plan_fixes(original, texts(j1=TEXT, j2=TEXT), NO_ALIASES, {2: override(title="Second")})

    first, second = (e.labels for e in plan.fixed.posts if e.labels)
    assert first is not None and second is not None
    assert (first.title, second.title) == ("Senior Backend Engineer", "Second")
    assert {c.post for c in plan.changes if c.rule == "override"} == {2}


def test_the_counts_include_the_override_rule() -> None:
    lines = format_counts([Change(2, "title", "a", "b", fix.RULE_OVERRIDE)])

    assert "  rule override: 1 change(s) in 1 post(s)" in lines


def test_kept_fields_are_listed() -> None:
    lines = overrides_kept({3: override(keep=["title", "company"]), 1: override(title="T")})

    assert lines == ["  KEPT as labelled: post 3: title, company"]


# --- company spelling ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        ("A.Team", "A·Team: Engineer\n\nA·Team builds.", ["A·Team"]),
        ("A·Team", "A.Team: Engineer", ["A.Team"]),
        ("A·Team", "A·Team: Engineer", []),  # the label is in the text exactly
        ("Acme Corp", "ACME CORP hires", ["ACME CORP"]),
        ("Acme", "Beta hires", []),
        ("...", "anything", []),
    ],
)
def test_the_posts_own_spelling_of_the_company_is_reported(
    label: str, text: str, expected: list[str]
) -> None:
    assert company_spellings(label, text) == expected


def test_spelling_notes_appear_in_the_plan_and_the_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "A·Team: Engineer\n\nWe use Python. A·Team is remote."
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy(company="A.Team", stack=["Python"]), text)))
    fake_reader(monkeypatch, {1: text})
    lines: list[str] = []

    run_fix(path, SETTINGS, session("n", lines), dry_run=True)

    output = "\n".join(lines)
    assert "COMPANY SPELLING" in output
    assert "post 1: label 'A.Team'; the post spells it ['A·Team']" in output
    assert "A·Team" in [
        c.after for c in plan_fixes(load_label_file(path), texts(j1=text), {}).changes
    ]


# --- the command with overrides ------------------------------------------------------------------


def test_the_command_reads_overrides_beside_the_labels_and_prints_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy())))
    write_overrides(tmp_path / "label_overrides.json", {"1": {"title": "Chosen Title"}})
    fake_reader(monkeypatch, {1: TEXT})
    lines: list[str] = []

    run_fix(path, SETTINGS, session("n", lines))

    output = "\n".join(lines)
    assert "Overrides: 1 post(s) from" in output
    assert "Chosen Title" in output and "override" in output


def test_an_overrides_file_outside_the_posts_is_refused_before_anything_is_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy())))
    other = write_overrides(tmp_path / "mine.json", {"7": {"title": "T"}})
    asked = fake_reader(monkeypatch, {1: TEXT})

    with pytest.raises(ScriptError, match="no labelled post 7"):
        run_fix(path, SETTINGS, session("y", []), overrides_path=other)

    assert asked == []
    assert not (tmp_path / "labels.clean-v1.json").exists()


def test_a_dry_run_shows_the_table_and_touches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy())))
    write_overrides(tmp_path / "label_overrides.json", {"1": {"title": "Chosen Title"}})
    before = path.read_bytes()
    fake_reader(monkeypatch, {1: TEXT})
    lines: list[str] = []

    code = run_fix(path, SETTINGS, session("y", lines), dry_run=True)

    assert code == 0
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["label_overrides.json", "labels.json"]
    assert not any(line.startswith("ASKED") for line in lines)
    assert any("Chosen Title" in line for line in lines)


def test_main_accepts_the_overrides_and_dry_run_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "labels.json"
    save_label_file(path, label_file(entry(messy())))
    chosen = write_overrides(tmp_path / "x.json", {"1": {"company": "Chosen Co"}})
    fake_reader(monkeypatch, {1: TEXT})
    monkeypatch.setattr(fix, "get_settings", lambda: SETTINGS)
    lines: list[str] = []

    code = fix.main(
        ["--labels", str(path), "--overrides", str(chosen), "--dry-run"], session("y", lines)
    )

    assert code == 0 and any("Chosen Co" in line for line in lines)


# --- seniority evidence in the post body ---------------------------------------------------------

FAR = "x" * 450 + ". "  # puts what follows beyond the first 400 characters, in its own sentence


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Acme: Engineer\n\nSenior role on our team. More.", "Senior role on our team"),
        ("Acme: Engineer\n\nWe need a JUNIOR dev. Apply now.", "We need a JUNIOR dev"),
        (
            FAR + "We are looking for a Lead to guide us. Bye.",
            "We are looking for a Lead to guide us",
        ),
        (FAR + "You will be a principal contributor.", "You will be a principal contributor."),
        (
            FAR + "Honestly, you are a mid engineer who ships. Bye",
            "Honestly, you are a mid engineer who ships",
        ),
    ],
)
def test_seniority_evidence_is_the_sentence_with_the_level_word(text: str, expected: str) -> None:
    assert seniority_evidence(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Acme: Engineer\n\nNo level is named here.",
        "Internal tools and a seniority-free team.",  # not whole words
        FAR + "The senior person on the team will guide you.",  # late, no phrase
        FAR + "We hire a lead. We are looking for a builder.",  # phrase in another sentence
        FAR + "Mid-sized company.\nYou are a builder.",  # phrase on another line
    ],
)
def test_no_evidence_without_a_whole_level_word_early_or_in_a_phrase_sentence(text: str) -> None:
    assert seniority_evidence(text) is None


def test_a_level_word_must_start_inside_the_first_400_characters() -> None:
    assert seniority_evidence("x" * 398 + " senior") is not None  # starts at index 399
    assert seniority_evidence("x" * 399 + " senior") is None  # starts at index 400


def test_long_evidence_sentences_are_cut() -> None:
    sentence = seniority_evidence("Senior " + "word " * 200)

    assert sentence is not None and sentence.endswith("…") and len(sentence) <= 301


@pytest.mark.parametrize(
    "text",
    [
        "Acme needs someone to lead the design of our API.",
        "You will lead a small team.",
        "You can lead an effort.",
        "Help lead our platform work.",
        "Come lead your own squad.",
        "Plan, lead and ship features.",
        "We want to lead development of the app.",
        "Please lead, then the team follows.",  # "the" is the second word after "lead"
        "We hire to lead product design.",
        "LEAD THE way.",  # case-insensitive
    ],
)
def test_lead_followed_by_a_verb_use_word_within_two_words_is_not_evidence(text: str) -> None:
    assert seniority_evidence(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "Lead Engineer wanted.",
        "We need a lead backend engineer today.",  # "backend engineer" is not in the list
        "Lead: three years of Python.",  # "three years" is not in the list
    ],
)
def test_lead_as_a_level_still_counts(text: str) -> None:
    assert seniority_evidence(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "You will work with senior engineers daily.",
        "Report to our Senior Leaders.",
        "Backed by senior leadership.",
        "SENIOR ENGINEERS mentor you.",
    ],
)
def test_senior_about_other_people_is_not_evidence(text: str) -> None:
    assert seniority_evidence(text) is None


@pytest.mark.parametrize(
    "text",
    ["Senior engineer wanted.", "Senior Backend Engineers wanted.", "A senior role."],
)
def test_senior_not_followed_by_a_people_word_still_counts(text: str) -> None:
    assert seniority_evidence(text) is not None


# --- no provider, no extraction ------------------------------------------------------------------


def test_the_fixer_cannot_reach_a_provider_or_an_extraction() -> None:
    source = (ROOT / "scripts" / "eval_fix_labels.py").read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)

    forbidden = ("app.llm", "httpx", "requests", "aiohttp", "app.models", "app.extraction.rules")
    assert not {i for i in imported if i.startswith(forbidden)}
    assert not {i for i in imported if i.startswith(("scripts.eval_run", "scripts.try_extraction"))}
    assert not {i for i in imported if "build_client" in i or "merge" in i}
    for word in ("read_stored_jobs", "rules_values", "extraction_method", "llm_enrichment"):
        assert word not in source
