"""CV skills: the extractor is the job-post dictionary scan without the job-post safeguards."""

import time

import pytest

from app.cv.skills import MAX_SKILL_CHARS, MAX_SKILLS, SkillError, find_skills, normalize_skills
from app.extraction.stack import MAX_LISTED, find_stack
from tests.cv_helpers import synthetic_cv_text

# --- finding skills in a CV ------------------------------------------------------------------


def test_the_synthetic_cv_yields_its_technologies_in_order_of_appearance() -> None:
    skills = find_skills(synthetic_cv_text())

    assert skills[:3] == ["Python", "FastAPI", "PostgreSQL"]
    for expected in (
        "Celery",
        "Redis",
        "Docker",
        "GitHub Actions",
        "React",
        "TypeScript",
        "Node.js",
    ):
        assert expected in skills
    assert len(skills) == len(set(skills))  # no duplicates, though the SKILLS list repeats most


def test_a_cv_may_name_more_technologies_than_a_job_post_is_allowed_to() -> None:
    names = [f"{name}" for name in ("Python", "Java", "Kotlin", "Scala", "Elixir", "Erlang")]
    names += ["Haskell", "Clojure", "PHP", "Laravel", "Django", "Flask", "FastAPI", "Redis"]
    text = "Skills: " + ", ".join(names)
    assert len(names) > MAX_LISTED

    assert find_stack(text) == []  # a job post that long is a recruiter's list
    assert len(find_skills(text)) == len(names)  # a CV is the person's own list


def test_a_technology_in_a_customer_list_still_counts_on_a_cv() -> None:
    text = "Built an integration used by Shopify merchants, trusted by OpenAI."

    assert find_stack(text) == []
    assert find_skills(text) == ["Shopify", "OpenAI"]


def test_ambiguous_words_still_need_a_capital_letter() -> None:
    assert find_skills("I move swift and rust-free; also ruby") == []
    assert find_skills("Languages: Swift, Rust, Ruby") == ["Swift", "Rust", "Ruby"]


def test_aliases_and_longest_spelling_are_reused_from_the_dictionary() -> None:
    assert find_skills("postgres, k8s and ruby on rails") == [
        "PostgreSQL",
        "Kubernetes",
        "Ruby on Rails",
    ]


def test_a_cv_without_technologies_gives_an_empty_list() -> None:
    assert find_skills("I enjoy hiking and cooking.") == []
    assert find_skills("") == []


@pytest.mark.parametrize(
    "text",
    ["python " * 4_000, "a." * 15_000, "é字🙂" * 10_000, "c++ " * 7_500, "x" * 30_000],
    ids=["repeats", "dotted", "unicode", "plusses", "one word"],
)
def test_hostile_cv_text_is_scanned_quickly(text: str) -> None:
    started = time.perf_counter()
    find_skills(text)

    assert time.perf_counter() - started < 0.5


# --- cleaning the user's own list --------------------------------------------------------------


def test_known_names_get_the_dictionarys_spelling_and_others_are_kept_as_typed() -> None:
    result = normalize_skills(["postgres", "  Hexagonal   architecture ", "TYPESCRIPT"])

    assert result == ["PostgreSQL", "Hexagonal architecture", "TypeScript"]


def test_duplicates_ignoring_case_and_blanks_are_dropped_and_order_is_kept() -> None:
    result = normalize_skills(["Docker", "", "  ", "docker", "Team leadership", "TEAM LEADERSHIP"])

    assert result == ["Docker", "Team leadership"]


def test_control_characters_and_line_breaks_cannot_split_or_hide_a_skill() -> None:
    assert normalize_skills(["Py​thon", "Go\nlang"]) == ["Python", "Go lang"]


def test_a_skill_longer_than_the_limit_is_refused_with_a_fixed_phrase() -> None:
    with pytest.raises(SkillError) as error:
        normalize_skills(["SECRET-" + "x" * MAX_SKILL_CHARS])

    assert "SECRET" not in str(error.value)


def test_more_than_the_maximum_number_of_skills_is_refused() -> None:
    with pytest.raises(SkillError):
        normalize_skills([f"skill {n}" for n in range(MAX_SKILLS + 1)])

    assert len(normalize_skills([f"skill {n}" for n in range(MAX_SKILLS)])) == MAX_SKILLS


# --- ambiguous words count only inside a skills list --------------------------------------------


def test_go_in_a_skills_line_is_a_skill_with_its_neighbours() -> None:
    assert sorted(find_skills("Skills: Python, Go, SQL")) == ["Go", "Python", "SQL"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Skills\nPython, Go, R", {"Python", "Go", "R"}),
        ("TECHNICAL SKILLS\n- Go\n- R\n- Docker", {"Go", "R", "Docker"}),
        ("Tech stack: go, r", {"Go", "R"}),  # lower case is fine inside a list
        ("Languages: Python / Go / Rust", {"Python", "Go", "Rust"}),
        ("Technologies\nBackend: Go, Redis\nData: R, SQL", {"Go", "Redis", "R", "SQL"}),
        ("Programming languages: C, C++, Swift", {"C", "C++", "Swift"}),
    ],
)
def test_a_heading_such_as_skills_makes_the_ambiguous_words_below_it_skills(
    text: str, expected: set[str]
) -> None:
    assert expected <= set(find_skills(text))


@pytest.mark.parametrize(
    "text",
    [
        "Built services in Python, Docker and Go",
        "Python / Docker / SQL / Go",
        "Python, Go, SQL",  # two other known technologies, no heading needed
        "Comfortable with Python, Redis, rust",
    ],
)
def test_a_line_that_lists_two_known_technologies_counts_its_ambiguous_items(text: str) -> None:
    skills = find_skills(text)

    assert {"Go", "Rust"} & set(skills)


@pytest.mark.parametrize(
    "text",
    [
        "Led the Go-To-Market plan for the launch",
        "Skills: Python, Go-To-Market, SQL",  # the item is not "Go"
        "We had to go to production before Friday, then fix bugs",
        "Helped the team go to production using Python, Docker",  # a clause, not an item
        "I like to go, run and swim",  # a list, but of no technologies
        "Python, go",  # two items: not enough to call it a list without a heading
        "Experience with R&D and Python, then the go-live",
        "Education\nBSc, Go Wild University, 2019",  # not a skills heading
        "Python was my first language; I go home at five, rust happens, and so on.",
        "Plan A, plan C, plan R",
    ],
)
def test_ambiguous_words_in_ordinary_sentences_are_not_skills(text: str) -> None:
    skills = find_skills(text)

    assert not {"Go", "R", "C", "Rust", "Ruby", "Swift"} & set(skills), skills


def test_the_list_rule_does_not_leak_into_the_job_post_dictionary() -> None:
    # Job posts keep the strict rule: a bare "Go" or "R" is never read from a post.
    assert find_stack("Skills: Python, Go, SQL, R") == ["Python", "SQL"]


def test_a_skills_section_ends_at_a_blank_line_or_the_next_section() -> None:
    text = "Skills\nPython, Docker\n\nHobbies: go\n\nEXPERIENCE\nWorked on R&D, go-live"

    skills = find_skills(text)

    assert skills == ["Python", "Docker"]


def test_a_word_inside_an_item_is_not_the_item() -> None:
    assert "Go" not in find_skills("Skills: Python, Go programming, SQL, Docker")


@pytest.mark.parametrize(
    "text",
    [
        "Skills: " + "go, " * 10_000,
        "\n".join(["Skills", *["go, r, c"] * 2_000]),
        "python, " * 4_000 + "go",
        ("a, " * 100 + "\n") * 300,
        "Skills:" * 4_000,
    ],
    ids=["one long list", "many lines", "huge list", "many short lists", "repeated heading"],
)
def test_hostile_skills_text_is_scanned_quickly(text: str) -> None:
    started = time.perf_counter()
    find_skills(text)

    assert time.perf_counter() - started < 0.5
