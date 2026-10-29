"""Title, company, seniority, remote policy, stack, apply email and link."""

import pytest

from app.extraction.rules import extract_with_rules
from app.extraction.stack import canonical_name, find_stack
from app.extraction.types import Confidence, Found
from app.models import RemotePolicy, Seniority


def post(title: str, body: str = "") -> str:
    """The shape the RSS fetcher stores: title, blank line, body."""
    return f"{title}\n\n{body}" if body else title


# --- title and company ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headline", "title", "company"),
    [
        ("Senior Python Developer at Acme Corp", "Senior Python Developer", "Acme Corp"),
        ("Acme Corp: Backend Engineer", "Backend Engineer", "Acme Corp"),
        ("Backend Engineer | Acme Corp", "Backend Engineer", "Acme Corp"),
        ("Backend Engineer - Acme Corp", "Backend Engineer", "Acme Corp"),
        ("Backend Engineer @ Acme", "Backend Engineer", "Acme"),
        ("Backend Engineer [Remote]", "Backend Engineer", None),
        ("Backend Engineer (Remote)", "Backend Engineer", None),
        ("Backend Engineer - Remote", "Backend Engineer", None),
        ("Hiring: Backend Engineer", "Backend Engineer", None),
        ("Full Stack Developer", "Full Stack Developer", None),
    ],
)
def test_headline_shapes_give_a_title_and_sometimes_a_company(
    headline: str, title: str, company: str | None
) -> None:
    result = extract_with_rules(post(headline, "Details follow."))

    assert result.title is not None
    assert result.title.value == title
    assert result.title.confidence is Confidence.HIGH  # a headline followed by a blank line
    assert (result.company.value if result.company else None) == company


def test_a_first_line_without_a_role_word_is_not_a_title() -> None:
    assert extract_with_rules(post("Big announcement today", "We are growing.")).title is None


def test_a_role_word_in_a_line_that_is_not_a_headline_is_only_a_low_confidence_title() -> None:
    result = extract_with_rules("Senior Engineer wanted\nWe build things.")

    assert result.title is not None
    assert result.title.confidence is Confidence.LOW


def test_labelled_fields_are_high_confidence() -> None:
    result = extract_with_rules("Great news\nCompany: Globex Ltd\nPosition: Data Analyst\nMore.")

    assert result.company is not None
    assert (result.company.value, result.company.confidence) == ("Globex Ltd", Confidence.HIGH)
    assert result.title is not None
    assert (result.title.value, result.title.confidence) == ("Data Analyst", Confidence.HIGH)


def test_the_company_can_come_from_a_hiring_sentence() -> None:
    result = extract_with_rules(post("QA Engineer", "Initech Systems is hiring a tester."))

    assert result.company is not None
    assert (result.company.value, result.company.confidence) == ("Initech Systems", Confidence.LOW)


def test_a_sentence_starting_with_we_is_not_a_company() -> None:
    assert extract_with_rules(post("QA Engineer", "We are hiring a tester.")).company is None


# --- seniority --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "level"),
    [
        ("Senior Backend Engineer", Seniority.SENIOR),
        ("Sr. Backend Engineer", Seniority.SENIOR),
        ("Junior Developer", Seniority.JUNIOR),
        ("Jr Developer", Seniority.JUNIOR),
        ("Entry-Level Analyst", Seniority.JUNIOR),
        ("Software Engineering Intern", Seniority.INTERN),
        ("Mid-Level Designer", Seniority.MID),
        ("Tech Lead", Seniority.LEAD),
        ("Staff Engineer", Seniority.PRINCIPAL),
        ("Principal Architect", Seniority.PRINCIPAL),
    ],
)
def test_seniority_comes_from_the_title(title: str, level: Seniority) -> None:
    result = extract_with_rules(post(title, "x"))

    assert result.seniority is not None
    assert result.seniority.value is level


@pytest.mark.parametrize(
    "title", ["Senior/Lead Engineer", "Backend Engineer", "Lead Generation Specialist"]
)
def test_no_seniority_when_the_title_is_silent_or_unclear(title: str) -> None:
    assert extract_with_rules(post(title, "x")).seniority is None


def test_seniority_in_the_body_is_not_used() -> None:
    assert (
        extract_with_rules(post("Backend Engineer", "We want a senior person.")).seniority is None
    )


# --- remote policy ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "policy"),
    [
        ("This is a fully remote role.", RemotePolicy.REMOTE),
        ("100% remote, work from anywhere", RemotePolicy.REMOTE),
        ("We are remote-first.", RemotePolicy.REMOTE),
        ("Hybrid: 3 days in the office a week", RemotePolicy.HYBRID),
        ("On-site in Berlin", RemotePolicy.ONSITE),
        ("Office-based role", RemotePolicy.ONSITE),
        ("This position is not remote.", RemotePolicy.ONSITE),
        ("No remote work available.", RemotePolicy.ONSITE),
    ],
)
def test_explicit_phrases_give_a_high_confidence_policy(body: str, policy: RemotePolicy) -> None:
    result = extract_with_rules(post("Engineer", body))

    assert result.remote_policy is not None
    assert result.remote_policy.value is policy
    assert result.remote_policy.confidence is Confidence.HIGH


def test_a_bare_remote_is_only_a_low_confidence_guess() -> None:
    result = extract_with_rules(post("Engineer (Remote)", "Join the team."))

    assert result.remote_policy is not None
    assert result.remote_policy.confidence is Confidence.LOW


@pytest.mark.parametrize(
    "body",
    [
        "Hybrid or fully remote",
        "Remote or on-site, your choice",
        "Remote or hybrid",
        "Nothing about location.",
    ],
)
def test_mixed_or_missing_signals_give_no_policy(body: str) -> None:
    assert extract_with_rules(post("Engineer", body)).remote_policy is None


def test_a_negated_onsite_is_not_onsite() -> None:
    result = extract_with_rules(post("Engineer", "Fully remote. No on-site requirement."))

    assert result.remote_policy is not None
    assert result.remote_policy.value is RemotePolicy.REMOTE


# --- stack ------------------------------------------------------------------------------------


def test_technologies_are_found_by_whole_word_in_order_without_duplicates() -> None:
    text = "We use Python and Django on PostgreSQL, plus python scripts, Node.js, C++ and C#."

    assert find_stack(text) == ["Python", "Django", "PostgreSQL", "Node.js", "C++", "C#"]


def test_java_is_not_found_inside_javascript_and_go_needs_golang() -> None:
    assert find_stack("JavaScript and TypeScript, let's go!") == ["JavaScript", "TypeScript"]
    assert find_stack("Golang services") == ["Go"]


def test_multi_word_names_win_over_their_first_word() -> None:
    assert find_stack("Ruby on Rails and React Native, GitHub Actions") == [
        "Ruby on Rails",
        "React Native",
        "GitHub Actions",
    ]


def test_names_are_matched_in_punctuation_and_dotted_forms() -> None:
    assert find_stack("(Python), Next.js; .NET, k8s.") == [
        "Python",
        "Next.js",
        ".NET",
        "Kubernetes",
    ]


def test_the_stack_is_capped_at_30() -> None:
    names = (
        "python java rust ruby php dart scala kotlin swift elixir erlang haskell clojure sql react "
        "vue svelte angular django flask fastapi laravel symfony flutter graphql redis mysql "
        "mongodb kafka rabbitmq celery pandas numpy pytorch tensorflow docker"
    )

    assert len(names.split()) > 30
    assert len(find_stack(names)) == 30


def test_canonical_name_maps_spellings_to_the_dictionary_name() -> None:
    assert canonical_name("postgres") == "PostgreSQL"
    assert canonical_name("PostgreSQL") == "PostgreSQL"
    assert canonical_name("Some Obscure Tool") is None


# --- apply email and link ---------------------------------------------------------------------


def test_the_email_next_to_apply_words_is_chosen() -> None:
    text = post("Engineer", "Questions? press@acme.com. To apply send your CV to jobs@acme.com.")

    result = extract_with_rules(text)

    assert result.apply_email is not None
    assert result.apply_email.value == "jobs@acme.com"


def test_two_equally_likely_emails_give_nothing() -> None:
    text = post("Engineer", "Write to anna@acme.com or to bob@other.org")

    assert extract_with_rules(text).apply_email is None


def test_an_address_that_only_looks_like_an_email_is_rejected_by_validation() -> None:
    assert extract_with_rules(post("Engineer", "Apply: a..b@acme.com")).apply_email is None


def test_noreply_addresses_are_never_the_apply_email() -> None:
    assert extract_with_rules(post("Engineer", "From noreply@acme.com")).apply_email is None


def test_a_link_near_apply_wording_or_on_a_known_ats_is_the_apply_link() -> None:
    text = post(
        "Engineer", "Our site https://acme.com/about. Apply here: https://acme.com/join/42."
    )

    result = extract_with_rules(text)

    assert result.apply_url is not None
    assert result.apply_url.value == "https://acme.com/join/42"


def test_an_ats_link_is_high_confidence_without_apply_wording() -> None:
    result = extract_with_rules(post("Engineer", "See https://jobs.lever.co/acme/123 for more."))

    assert result.apply_url is not None
    assert result.apply_url.confidence is Confidence.HIGH


@pytest.mark.parametrize(
    "body",
    [
        "Follow us https://twitter.com/acme and https://www.linkedin.com/company/acme",
        "Unsubscribe: https://acme.com/unsubscribe?id=1",
        "Homepage https://acme.com/",
        "Local http://localhost/apply",
        "Apply: http://10.0.0.1/job",
    ],
)
def test_links_that_are_not_application_links_are_ignored(body: str) -> None:
    assert extract_with_rules(post("Engineer", body)).apply_url is None


# --- the whole post ---------------------------------------------------------------------------


def test_an_empty_post_gives_an_empty_result() -> None:
    result = extract_with_rules("")

    assert result.title is None
    assert result.stack == ()
    assert result.apply_email is None


def test_found_reports_whether_it_is_high_confidence() -> None:
    assert Found("x", Confidence.HIGH).is_high is True
    assert Found("x", Confidence.LOW).is_high is False
