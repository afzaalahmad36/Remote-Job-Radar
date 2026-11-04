"""The accuracy scoring, on small fixture labels. Pure functions: no database, no network.

The rule being tested is written in CLAUDE.md (section 6, "Extraction accuracy evaluation").
"""

import pytest

from scripts.eval_scoring import (
    FieldValues,
    Outcome,
    PostScore,
    combine,
    compare,
    field_table,
    headline,
    normalise_company,
    normalise_currency,
    normalise_enum,
    normalise_title,
    score_post,
    score_stack,
    stack_key,
    summarise,
    wilson_interval,
)

LABEL = FieldValues(
    title="Senior Python Developer",
    company="Acme Corp",
    stack=("Python", "Django", "PostgreSQL"),
    remote_policy="remote",
    seniority="senior",
    salary_min=90000,
    salary_max=120000,
    salary_currency="USD",
    salary_period="year",
)


def changed(**fields: object) -> FieldValues:
    """The label with some fields replaced: a system answer that is right except for these."""
    return FieldValues(**{**LABEL.__dict__, **fields})  # type: ignore[arg-type]


# --- normalisation -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Senior Backend Engineer (m/f/d)", "senior backend engineer"),
        ("Backend Engineer [w/m/d]", "backend engineer"),
        ("Backend  Engineer\n", "backend engineer"),
        ("Backend Engineer - Remote", "backend engineer remote"),
        ("Backend Engineer – Remote", "backend engineer remote"),
        ("Backend/Platform Engineer, Payments", "backend platform engineer payments"),
        ("\U0001f680 Backend Engineer \U0001f680", "backend engineer"),
        ("C++ Developer", "c++ developer"),
        ("Senior C#", "senior c#"),
        ("ＰＹＴＨＯＮ Developer", "python developer"),  # full-width
        ("  ", None),
        ("!!!", None),
    ],
)
def test_titles_are_compared_after_normalisation(raw: str, expected: str | None) -> None:
    assert normalise_title(raw) == expected


def test_a_title_is_matched_exactly_not_by_containment() -> None:
    assert normalise_title("Senior Backend Engineer") != normalise_title("Backend Engineer")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Acme, Inc.", "acme"),
        ("ACME GmbH", "acme"),
        ("Acme Corp Co.", "acme"),
        ("Acme Ltd", "acme"),
        ("Inc", "inc"),  # a name that is only a suffix stays
        ("Co-op Bank", "co-op bank"),
        ("  Acme  ", "acme"),
        ("", None),
    ],
)
def test_companies_lose_their_legal_suffix(raw: str, expected: str | None) -> None:
    assert normalise_company(raw) == expected


def test_enums_and_currency_are_normalised() -> None:
    assert normalise_enum("On-site") == "onsite" == normalise_enum("onsite")
    assert normalise_enum(" Remote ") == "remote"
    assert normalise_enum(None) is None
    assert normalise_currency(" usd ") == "USD"
    assert normalise_currency("") is None


@pytest.mark.parametrize(
    ("a", "b"),
    [("js", "JavaScript"), ("JS", "javascript"), ("python3", "Python"), ("FIGMA", " figma ")],
)
def test_stack_entries_are_compared_by_the_dictionary_spelling(a: str, b: str) -> None:
    assert stack_key(a) == stack_key(b)


# --- the five outcomes -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "system", "expected"),
    [
        (None, None, Outcome.ABSENT),
        ("x", "x", Outcome.CORRECT),
        ("x", "y", Outcome.WRONG),
        ("x", None, Outcome.MISSING),
        (None, "y", Outcome.INVENTED),
    ],
)
def test_the_five_outcomes(label: str | None, system: str | None, expected: Outcome) -> None:
    assert compare(label, system) is expected


def test_a_group_takes_its_worst_outcome() -> None:
    assert combine([Outcome.CORRECT, Outcome.WRONG, Outcome.MISSING]) is Outcome.WRONG
    assert combine([Outcome.CORRECT, Outcome.INVENTED, Outcome.MISSING]) is Outcome.INVENTED
    assert combine([Outcome.CORRECT, Outcome.MISSING]) is Outcome.MISSING
    assert combine([Outcome.CORRECT, Outcome.ABSENT]) is Outcome.CORRECT
    assert combine([Outcome.ABSENT, Outcome.ABSENT]) is Outcome.ABSENT


# --- the stack ---------------------------------------------------------------------------------


def test_stack_counts_are_per_entry_and_ignore_case_aliases_and_repeats() -> None:
    score = score_stack(["Python", "Django", "Node"], ["python3", "PYTHON", "django", "Redis"])

    assert (score.true_positives, score.false_positives, score.false_negatives) == (2, 1, 1)
    assert score.label_size == 3


@pytest.mark.parametrize(("found", "ok"), [(7, True), (6, False), (10, True)])
def test_the_stack_must_find_at_least_70_percent_of_the_label(found: int, ok: bool) -> None:
    label = [f"Tool{i}" for i in range(10)]

    assert score_stack(label, label[:found]).ok is ok


def test_an_entry_outside_the_label_fails_the_stack_even_with_full_recall() -> None:
    score = score_stack(["Python"], ["Python", "Redis"])

    assert score.true_positives == 1 and score.false_positives == 1 and not score.ok


def test_an_empty_label_passes_recall_trivially_but_any_entry_fails() -> None:
    assert score_stack([], []).ok
    assert not score_stack([], ["Python"]).ok


def test_label_entries_the_dictionary_does_not_know_are_counted_as_gaps() -> None:
    score = score_stack(["Python", "Penpot", "Obscure Tool"], ["Python"])

    assert score.dictionary_gaps == 2
    assert score.false_negatives == 2


# --- one post ----------------------------------------------------------------------------------


def test_a_fully_correct_post_passes() -> None:
    score = score_post(LABEL, LABEL)

    assert score.passed and score.failures == () and not score.invented
    assert set(score.outcomes.values()) == {Outcome.CORRECT}


@pytest.mark.parametrize(
    ("system", "failures"),
    [
        (changed(title=None), ("title_missing",)),
        (changed(title="Junior Python Developer"), ("title_wrong",)),
        (changed(title="Senior Python Developer (m/f/d)"), ()),  # normalised away
        (changed(company=None), ("company_missing",)),
        (changed(company="Other Co"), ("company_wrong",)),
        (changed(company="ACME corporation"), ("company_wrong",)),
        (changed(company="Acme, Corp."), ()),
        (changed(stack=("Python", "Django", "PostgreSQL", "Redis")), ("stack_extra_entries",)),
        (changed(stack=("Python", "Django")), ("stack_low_recall",)),
        (changed(remote_policy=None), ()),  # missing is allowed
        (changed(remote_policy="hybrid"), ("remote_policy_wrong",)),
        (changed(seniority=None), ()),
        (changed(seniority="junior"), ("seniority_wrong",)),
        (changed(salary_min=None, salary_max=None), ()),  # missing amounts are allowed
        (changed(salary_period=None), ()),
        (changed(salary_min=80000), ("salary_wrong",)),
        (changed(salary_currency="EUR"), ("salary_wrong",)),
        (changed(salary_period="month"), ("salary_wrong",)),
    ],
)
def test_the_pass_rule_field_by_field(system: FieldValues, failures: tuple[str, ...]) -> None:
    assert score_post(LABEL, system).failures == failures


def test_a_blank_company_label_accepts_any_company_but_the_invention_is_counted() -> None:
    label = changed(company=None)

    score = score_post(label, changed(company="Whoever"))

    assert score.passed
    assert score.outcomes["company"] is Outcome.INVENTED
    assert score.invented


@pytest.mark.parametrize(
    ("label", "system", "code"),
    [
        (changed(remote_policy=None), changed(remote_policy="remote"), "remote_policy_invented"),
        (changed(seniority=None), changed(seniority="senior"), "seniority_invented"),
        (changed(salary_min=None, salary_max=None), LABEL, "salary_invented"),
    ],
)
def test_an_invented_value_fails_the_post_and_counts_for_the_gate(
    label: FieldValues, system: FieldValues, code: str
) -> None:
    score = score_post(label, system)

    assert code in score.failures
    assert score.invented


def test_salary_parts_are_judged_one_by_one() -> None:
    no_salary = changed(salary_min=None, salary_max=None, salary_currency=None, salary_period=None)

    only_a_currency = score_post(
        no_salary, changed(salary_min=None, salary_max=None, salary_period=None)
    )
    one_amount_right = score_post(LABEL, changed(salary_max=None, salary_period=None))

    assert only_a_currency.outcomes["salary"] is Outcome.INVENTED and only_a_currency.invented
    assert one_amount_right.outcomes["salary"] is Outcome.MISSING and one_amount_right.passed


def test_single_amount_labels_follow_the_rules_conventions() -> None:
    label = changed(salary_min=90000, salary_max=90000)

    assert score_post(label, changed(salary_min=90000, salary_max=90000)).passed
    assert not score_post(label, changed(salary_min=90000, salary_max=None)).failures
    assert score_post(label, changed(salary_min=90000, salary_max=120000)).failures == (
        "salary_wrong",
    )


def test_a_post_without_a_title_label_is_never_scored() -> None:
    with pytest.raises(ValueError, match="blank title"):
        score_post(changed(title=None), LABEL)
    with pytest.raises(ValueError, match="blank title"):
        score_post(changed(title="  "), LABEL)


# --- many posts --------------------------------------------------------------------------------


def posts(passing: int, failing: int = 0, invented: int = 0) -> list[PostScore]:
    """`invented` passing posts that also contain an invented company."""
    good = score_post(LABEL, LABEL)
    bad = score_post(LABEL, changed(title="Something else"))
    made_up = score_post(changed(company=None), changed(company="Whoever"))
    assert good.passed and not bad.passed and made_up.passed and made_up.invented
    return [good] * (passing - invented) + [made_up] * invented + [bad] * failing


def test_27_of_30_meets_the_target() -> None:
    summary = summarise(posts(27, 3))

    assert (summary.scored, summary.passed) == (30, 27)
    assert summary.target_met and not summary.gate_failed
    assert summary.verdict == "PASS"
    assert summary.fail_reasons == ()


def test_26_of_30_misses_the_target() -> None:
    summary = summarise(posts(26, 4))

    assert not summary.target_met
    assert summary.verdict == "FAIL"
    assert any("fewer than 90%" in reason for reason in summary.fail_reasons)


def test_the_invented_gate_fails_the_evaluation_whatever_the_parse_share() -> None:
    summary = summarise(posts(30, 0, invented=4))  # 4 of 30 is 13%: every post parses

    assert summary.passed == 30 and summary.target_met
    assert summary.gate_failed
    assert summary.verdict == "FAIL"
    assert any("invented" in reason for reason in summary.fail_reasons)


def test_exactly_10_percent_invented_is_still_allowed() -> None:
    assert not summarise(posts(30, 0, invented=3)).gate_failed
    assert summarise(posts(30, 0, invented=3)).verdict == "PASS"
    assert summarise(posts(30, 0, invented=4)).gate_failed


def test_both_conditions_can_fail_together() -> None:
    summary = summarise(posts(20, 10, invented=5))

    assert len(summary.fail_reasons) == 2


def test_an_empty_sample_has_no_verdict() -> None:
    summary = summarise([])

    assert summary.verdict == "NO DATA"
    assert summary.pass_share is None and summary.interval is None
    assert "NO DATA" in headline("Stored", summary)


def test_the_wilson_interval_around_27_of_30() -> None:
    low, high = wilson_interval(27, 30) or (0.0, 0.0)

    assert 0.73 < low < 0.76
    assert 0.96 < high < 0.98
    assert wilson_interval(0, 0) is None
    assert wilson_interval(30, 30) == pytest.approx((0.886, 1.0), abs=0.01)


def test_field_accuracy_and_recall_on_labelled_values_are_separate() -> None:
    def remote(label: str | None, system: str | None) -> PostScore:
        return score_post(changed(remote_policy=label), changed(remote_policy=system))

    summary = summarise(
        [
            remote("remote", "remote"),  # correct
            remote("remote", None),  # missing
            remote(None, None),  # absent
            remote("hybrid", "remote"),  # wrong
        ]
    )
    field = summary.fields["remote_policy"]

    assert field.accuracy == pytest.approx(2 / 4)  # correct + correctly absent
    assert field.recall_on_labelled == pytest.approx(1 / 3)  # correct / (correct+wrong+missing)
    assert field.counts[Outcome.WRONG] == field.counts[Outcome.MISSING] == 1


def test_a_field_nobody_labelled_has_no_recall() -> None:
    summary = summarise([score_post(changed(seniority=None), changed(seniority=None))])

    assert summary.fields["seniority"].recall_on_labelled is None
    assert summary.fields["seniority"].accuracy == 1.0


def test_stack_precision_recall_and_f1_are_micro_averaged() -> None:
    summary = summarise(
        [
            score_post(
                changed(stack=("Python", "Django")), changed(stack=("Python", "Django", "Redis"))
            ),
            score_post(changed(stack=("Go",)), changed(stack=())),
        ]
    )
    stack = summary.stack

    assert (stack.true_positives, stack.false_positives, stack.false_negatives) == (2, 1, 1)
    assert stack.precision == pytest.approx(2 / 3)
    assert stack.recall == pytest.approx(2 / 3)
    assert stack.f1 == pytest.approx(2 / 3)
    assert stack.posts_ok == 0 and stack.posts == 2


def test_stack_metrics_with_nothing_to_measure_are_none() -> None:
    stack = summarise([score_post(changed(stack=()), changed(stack=()))]).stack

    assert stack.precision is None and stack.recall is None and stack.f1 is None
    assert stack.posts_ok == 1


def test_failure_codes_are_counted_across_posts() -> None:
    summary = summarise(posts(5, 3))

    assert summary.failure_counts == {"title_wrong": 3}


# --- what is printed ---------------------------------------------------------------------------


def test_the_headline_shows_the_share_the_gate_and_the_dictionary_gaps_together() -> None:
    label = changed(stack=("Python", "Penpot"))
    scores = [score_post(label, changed(stack=("Python", "Penpot")))] * 27 + [
        score_post(label, changed(title="Other"))
    ] * 3

    text = headline("Stored", summarise(scores))

    assert "Stored: PASS (on this sample)" in text
    assert "27/30 = 90.0%" in text and "95% interval" in text and "target 90%" in text
    assert "invented gate" in text and "0/30" in text
    assert "stack dictionary: 30 label entries unknown to the dictionary" in text


def test_a_failing_headline_says_why() -> None:
    text = headline("Rules", summarise(posts(30, 0, invented=5)))

    assert "Rules: FAIL" in text
    assert "FAIL because" in text and "invented value" in text


def test_the_field_table_lists_every_field_and_the_recall_footnote() -> None:
    table = field_table(summarise(posts(3, 1)))

    for name in ("title", "company", "remote_policy", "seniority", "salary", "stack"):
        assert name in table
    assert "recall" in table and "precision" in table
