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


# A job board puts "Headquarters: ..." in the first lines. The header is read only when the text
# itself gave mixed signals (a remote word next to a hybrid or on-site word).


def hq_post(header: str, body: str) -> str:
    return post("Acme: Backend Engineer", f"Headquarters: {header}\nSalary: n/a\n\n{body}")


def test_a_remote_headquarters_settles_a_generic_hybrid_mention() -> None:
    body = "A remote team. Some colleagues enjoy hybrid arrangements."

    result = extract_with_rules(hq_post("Remote - US", body))

    assert result.remote_policy == Found(RemotePolicy.REMOTE, Confidence.HIGH)


@pytest.mark.parametrize("header", ["Anywhere", "Worldwide", "Germany (Remote)"])
def test_a_headquarters_with_anywhere_or_worldwide_or_remote_counts(header: str) -> None:
    body = "Our remote team. An on-site day is possible for the odd workshop."

    result = extract_with_rules(hq_post(header, body))

    assert result.remote_policy == Found(RemotePolicy.REMOTE, Confidence.HIGH)


@pytest.mark.parametrize(
    "obligation",
    [
        "Hybrid: 3 days a week in the office.",
        "Remote, but you must live within 50 miles of Austin. Hybrid weeks apply.",
        "Hybrid role, in the office two days each week.",
        "Remote, hybrid if you relocate to Denver.",
    ],
)
def test_a_body_obligation_is_not_overruled_by_the_header(obligation: str) -> None:
    assert extract_with_rules(hq_post("Remote - US", obligation)).remote_policy is None


def test_a_place_only_headquarters_decides_nothing() -> None:
    body = "A remote team. Some colleagues enjoy hybrid arrangements."

    assert extract_with_rules(hq_post("Austin, Texas", body)).remote_policy is None


def test_a_headquarters_naming_hybrid_decides_nothing() -> None:
    body = "A remote team. Some colleagues enjoy hybrid arrangements."

    assert extract_with_rules(hq_post("Remote or hybrid", body)).remote_policy is None


def test_a_post_without_a_headquarters_line_is_unaffected() -> None:
    body = "A remote team. Some colleagues enjoy hybrid arrangements."

    assert extract_with_rules(post("Backend Engineer", body)).remote_policy is None


def test_the_headquarters_is_only_read_in_the_first_lines() -> None:
    late = "\n".join(["line"] * 10) + "\nHeadquarters: Remote - US\nRemote, some hybrid."

    assert extract_with_rules(post("Backend Engineer", late)).remote_policy is None


def test_a_remote_headquarters_does_not_change_a_clear_text_signal() -> None:
    result = extract_with_rules(hq_post("Remote - US", "This position is not remote."))

    assert result.remote_policy == Found(RemotePolicy.ONSITE, Confidence.HIGH)


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


def test_a_post_naming_more_than_12_technologies_reports_none() -> None:
    twelve = "python java rust ruby php dart scala kotlin swift elixir erlang haskell"
    capitalised = " ".join(word.capitalize() for word in (twelve + " clojure").split())

    assert len(find_stack(twelve.title())) == 12  # the limit itself is still a stack
    assert find_stack(capitalised) == []  # one more is a recruiter's category list


def test_the_same_technology_named_many_times_counts_once_towards_the_limit() -> None:
    assert find_stack("Python " * 40) == ["Python"]


def test_a_customer_list_does_not_count_as_stack() -> None:
    text = "Trusted by OpenAI, PayPal and Stripe. We build with Python and Docker."

    assert find_stack(text) == ["Python", "Docker"]


def test_a_customer_list_phrase_ends_with_its_sentence() -> None:
    assert find_stack("Used by millions. Our backend is Django.") == ["Django"]
    assert find_stack("Clients include Shopify and Stripe.") == []


def test_an_alternative_phrase_does_not_count_as_stack() -> None:
    assert find_stack("A Shopify alternative for small shops, built in Rust.") == ["Rust"]
    assert find_stack("An alternative to Firebase, written in golang.") == ["Go"]
    assert find_stack("We chose Postgres instead of MongoDB.") == ["PostgreSQL"]


def test_a_technology_named_in_a_customer_list_and_elsewhere_still_counts() -> None:
    assert find_stack("Trusted by OpenAI. You will use OpenAI APIs daily.") == ["OpenAI"]


@pytest.mark.parametrize(
    "text",
    [
        "We value swift execution.",
        "iron rust is orange and a rust belt town.",
        "a ruby red gem, celery sticks, a flask of tea and unreal beauty.",
        "no jest in the dart game on the rails.",
    ],
)
def test_ambiguous_english_words_are_not_technologies_in_lower_case(text: str) -> None:
    assert find_stack(text) == []


def test_a_capitalised_ambiguous_word_at_the_start_of_a_sentence_still_counts() -> None:
    # Known limit: a capital letter cannot tell "Swift fingers" from the language.
    assert find_stack("Swift fingers wanted.") == ["Swift"]


def test_ambiguous_words_count_when_capitalised() -> None:
    text = "Experience with Swift, Rust and Ruby required. Celery and Flask are a plus."

    assert find_stack(text) == ["Swift", "Rust", "Ruby", "Celery", "Flask"]


def test_unambiguous_names_still_count_in_lower_case() -> None:
    assert find_stack("python and django") == ["Python", "Django"]


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


# --- title cleanup ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("headline", "title", "company"),
    [
        ("Acme: Backend Engineer ($90-$170/hr)", "Backend Engineer", "Acme"),
        ("Acme: Web Developer (Fresh graduate welcome, WFH policy)", "Web Developer", "Acme"),
        ("Acme: Developer (on-job-training provided)", "Developer", "Acme"),
        ("Acme: FULL TIME: Software Engineer", "Software Engineer", "Acme"),
        ("FULL TIME: Software Engineer", "Software Engineer", None),
        ("Part-time - Data Analyst", "Data Analyst", None),
        ("Backend Engineer ($120k/yr) (Remote)", "Backend Engineer", None),
        ("Acme: AI Engineer (Conversational AI)", "AI Engineer (Conversational AI)", "Acme"),
    ],
)
def test_a_price_or_perk_parenthetical_and_an_employment_prefix_are_not_the_title(
    headline: str, title: str, company: str | None
) -> None:
    result = extract_with_rules(post(headline, "Details follow."))

    assert result.title is not None and result.title.value == title
    assert (result.company.value if result.company else None) == company


def test_a_title_made_only_of_noise_is_kept_as_it_was() -> None:
    result = extract_with_rules(post("Engineer (welcome)", "Details follow."))

    assert result.title is not None and result.title.value == "Engineer"


def test_seniority_is_read_from_the_cleaned_title() -> None:
    graduate = extract_with_rules(post("Acme: Web Developer (Fresh graduate welcome)", "Hi."))
    junior = extract_with_rules(post("Acme: Junior Web Developer (graduates welcome)", "Hi."))

    assert graduate.seniority is None  # "graduate" lived in the perk parenthetical
    assert junior.seniority == Found(Seniority.JUNIOR, Confidence.HIGH)


# --- WFH ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "Work From Home (WFH) option available one day per week.",
        "We have a WFH policy.",
        "WFH is optional.",
        "Occasional WFH.",
        "WFH allowance of a few dollars.",
        "WFH two days a week.",
    ],
)
def test_an_occasional_wfh_is_not_a_remote_job(body: str) -> None:
    assert extract_with_rules(post("Developer", body)).remote_policy is None


def test_a_plain_wfh_is_remote() -> None:
    result = extract_with_rules(post("Developer", "This is a WFH role."))

    assert result.remote_policy == Found(RemotePolicy.REMOTE, Confidence.HIGH)


def test_one_firm_wfh_mention_wins_over_an_occasional_one() -> None:
    body = "WFH policy applies.\nThe job itself is WFH."

    assert extract_with_rules(post("Developer", body)).remote_policy == Found(
        RemotePolicy.REMOTE, Confidence.HIGH
    )


def test_a_link_with_remote_in_its_address_is_not_a_remote_signal() -> None:
    link = "Apply: https://weworkremotely.com/remote-jobs/acme-web-developer"

    assert extract_with_rules(post("Web Developer", link)).remote_policy is None
    assert extract_with_rules(
        post("Web Developer", link + "\nFully remote.")
    ).remote_policy == Found(RemotePolicy.REMOTE, Confidence.HIGH)
