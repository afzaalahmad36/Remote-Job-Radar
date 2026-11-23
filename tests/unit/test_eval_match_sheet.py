"""The sheet builder of the matching evaluation (scripts/match_eval.py, eval_match_sheet.py).

Invented jobs, invented CVs, the fake embedder: no database, no real model, no network.
"""

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.matching.fake_embedder import FakeEmbedder
from app.matching.scoring import PreferenceFacts, ScoringConfig
from app.models.job import RemotePolicy, Seniority
from scripts import eval_match_sheet as sheet
from scripts import match_eval as ev
from scripts.calibrate_similarity import cosine, percentile
from scripts.db_guard import ScriptError
from tests.unit.test_embedding_isolation import forbidden_imports

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG = ScoringConfig(
    weight_skills=0.6,
    weight_embedding=0.4,
    similarity_floor=0.715,
    similarity_ceiling=0.86,
    failed_filter_cap=30,
)
EMBEDDER = FakeEmbedder()
START = datetime(2026, 1, 1, tzinfo=UTC)

# Ids that start like the documented worked examples, so they resolve.
EXAMPLE_PREFIXES = [
    "4cf4674d",
    "14a2b4d7",
    "ce090770",
    "ac5106d6",
    "3c85f760",
    "0c02a211",
    "7b9c2242",
    "0922063b",
]
STACKS = [("Python", "SQL"), ("React", "TypeScript"), ("Python", "Airflow"), (), ("Go",)]


def make_job(index: int, job_id: str | None = None, title: str | None = None) -> ev.PoolJob:
    name = title or f"Invented role {index}"
    return ev.PoolJob(
        id=job_id or f"{index:08x}-0000-4000-8000-{index:012x}",
        title=name,
        company=f"Company {index}",
        seniority=[Seniority.SENIOR, Seniority.JUNIOR, None][index % 3],
        remote_policy=[RemotePolicy.REMOTE, RemotePolicy.ONSITE, None][index % 3],
        stack=STACKS[index % len(STACKS)],
        salary_min=None,
        salary_max=None,
        salary_currency=None,
        salary_period=None,
        posted_at=START + timedelta(days=index),
        created_at=START,
        text=f"The team builds software for customers and we are hiring for the role {index}.",
        raw_text_hash=f"hash-{index}",
        rules_version="rules_v3",
        vector=EMBEDDER.embed([name])[0],
    )


def make_pool(count: int = 60) -> list[ev.PoolJob]:
    pool = [make_job(i) for i in range(count - len(EXAMPLE_PREFIXES))]
    for offset, prefix in enumerate(EXAMPLE_PREFIXES):
        index = 1000 + offset
        pool.append(make_job(index, job_id=f"{prefix}-0000-4000-8000-{index:012x}"))
    return pool


def persona(
    name: str, kind: ev.PersonaKind = "synthetic", skills: tuple[str, ...] = ("Python",)
) -> ev.Persona:
    return ev.Persona(
        name=name,
        kind=kind,
        skills=skills,
        vector=EMBEDDER.embed([f"cv of {name}"])[0],
        preferences=PreferenceFacts(),
    )


def three() -> list[ev.Persona]:
    return [persona("alex"), persona("sam", skills=("React",)), persona("jordan", skills=("SQL",))]


def plans_for(pool: list[ev.PoolJob], seed: int = 1) -> list[ev.PersonaPlan]:
    return ev.plan_all(three(), pool, CONFIG, seed)


# --- the frozen pool ---------------------------------------------------------------------------


def test_the_pool_hash_ignores_order_and_changes_with_content() -> None:
    assert ev.pool_hash(["b", "a", "c"]) == ev.pool_hash(["c", "a", "b"])
    assert ev.pool_hash(["a", "b"]) != ev.pool_hash(["a", "c"])


def test_a_pool_that_is_not_the_frozen_one_is_refused() -> None:
    with pytest.raises(ScriptError, match="not the frozen one"):
        ev.check_pool(make_pool())


def test_the_protocol_document_records_the_ids_and_their_hash() -> None:
    text = (ROOT / "docs" / "MATCHING.md").read_text(encoding="utf-8")
    block = re.search(r"The ids:\n\n```text\n(.*?)\n```", text, re.DOTALL)
    assert block is not None
    ids = block.group(1).split("\n")

    assert len(ids) == ev.PROTOCOL_POOL_SIZE == 108
    assert ev.pool_hash(ids) == ev.PROTOCOL_POOL_HASH
    assert ev.PROTOCOL_POOL_HASH in text


def test_every_worked_example_of_the_protocol_is_in_the_recorded_pool() -> None:
    text = (ROOT / "docs" / "MATCHING.md").read_text(encoding="utf-8")
    block = re.search(r"The ids:\n\n```text\n(.*?)\n```", text, re.DOTALL)
    assert block is not None
    ids = block.group(1).split("\n")

    for name, examples in ev.SEEN_EXAMPLES.items():
        assert len(ev.resolve_examples(name, ids)) == len(examples)


# --- rankings ----------------------------------------------------------------------------------


def test_every_method_ranks_every_job_once() -> None:
    pool = make_pool()
    plan = plans_for(pool)[0]

    for method in ev.METHODS:
        assert sorted(plan.rankings[method]) == sorted(j.id for j in pool)


def test_ties_go_to_the_newer_job_then_to_the_smaller_id() -> None:
    older = make_job(1, job_id="bbbb0000-0000-4000-8000-000000000001")
    newer = make_job(2, job_id="bbbb0000-0000-4000-8000-000000000002")
    same_time_a = make_job(3, job_id="aaaa0000-0000-4000-8000-000000000003")
    same_time_b = make_job(3, job_id="cccc0000-0000-4000-8000-000000000003")

    order = ev._order(
        {j.id: 1.0 for j in (older, newer, same_time_a, same_time_b)},
        [older, newer, same_time_a, same_time_b],
    )

    assert order == [same_time_a.id, same_time_b.id, newer.id, older.id]


def test_newest_first_orders_by_date() -> None:
    pool = make_pool()
    plan = plans_for(pool)[0]
    dates = {j.id: j.recency for j in pool}
    ranked = [dates[i] for i in plan.rankings["newest_first"]]

    assert ranked == sorted(ranked, reverse=True)


def test_embedding_only_orders_by_cosine() -> None:
    pool = make_pool()
    me = three()[0]
    plan = plans_for(pool)[0]
    cosines = {j.id: cosine(me.vector, j.vector) for j in pool}
    ranked = [cosines[i] for i in plan.rankings["embedding_only"]]

    assert ranked == sorted(ranked, reverse=True)


# --- calibration B -----------------------------------------------------------------------------


def test_b_uses_only_the_other_personas_and_never_the_held_out_cv() -> None:
    pool = make_pool()
    alex, sam, jordan = three()
    others = sorted(ev.cv_job_cosines(sam, pool) + ev.cv_job_cosines(jordan, pool))
    expected = (percentile(others, 50.0), percentile(others, 99.0))

    assert ev.b_constants(alex, [alex, sam, jordan], pool) == expected
    other_alex = ev.Persona(
        "alex", "synthetic", ("Python",), EMBEDDER.embed(["another cv"])[0], alex.preferences
    )
    assert ev.b_constants(other_alex, [other_alex, sam, jordan], pool) == expected


def test_a_real_cv_is_calibrated_from_the_synthetic_ones_and_calibrates_none_of_them() -> None:
    pool = make_pool()
    alex, sam, jordan = three()
    real = persona("me", kind="real")

    without = ev.b_constants(alex, [alex, sam, jordan], pool)
    with_real = ev.b_constants(alex, [alex, sam, jordan, real], pool)
    synthetic = sorted(c for p in (alex, sam, jordan) for c in ev.cv_job_cosines(p, pool))

    assert with_real == without
    assert ev.b_constants(real, [alex, sam, jordan, real], pool) == (
        percentile(synthetic, 50.0),
        percentile(synthetic, 99.0),
    )


def test_a_single_persona_cannot_be_calibrated() -> None:
    alex = three()[0]

    with pytest.raises(ScriptError, match="no other synthetic persona"):
        ev.b_constants(alex, [alex], make_pool())


# --- the sheet ---------------------------------------------------------------------------------


def test_a_sheet_holds_the_tops_the_examples_and_ten_random_jobs_without_repeats() -> None:
    pool = make_pool()
    for plan in plans_for(pool):
        tops = {i for m in ev.METHODS for i in plan.rankings[m][: ev.TOP_N]}

        assert len(plan.items) == len(set(plan.items))
        assert tops <= set(plan.items)
        assert set(plan.examples) <= set(plan.items)
        assert len(plan.random_ids) == ev.RANDOM_N
        assert not set(plan.random_ids) & (tops | set(plan.examples))
        assert set(plan.items) == tops | set(plan.examples) | set(plan.random_ids)


def test_each_persona_gets_its_documented_examples() -> None:
    plans = {p.name: p for p in plans_for(make_pool())}

    assert sorted(plans["alex"].examples.values()) == ["maybe", "no", "relevant"]
    assert {i[:8] for i in plans["alex"].examples} == {"4cf4674d", "14a2b4d7", "ce090770"}
    assert {i[:8] for i in plans["sam"].examples} == {"ac5106d6", "3c85f760", "0c02a211"}


def test_the_plan_is_repeatable_for_a_seed_and_changes_with_another() -> None:
    pool = make_pool()

    assert [p.items for p in plans_for(pool, 1)] == [p.items for p in plans_for(pool, 1)]
    assert [p.items for p in plans_for(pool, 1)] != [p.items for p in plans_for(pool, 2)]


def test_the_order_of_the_items_does_not_follow_a_ranking() -> None:
    plan = plans_for(make_pool())[0]

    assert plan.items != sorted(plan.items)
    assert plan.items[: ev.TOP_N] != plan.rankings["hybrid_a"][: ev.TOP_N]


def all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in all_keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in all_keys(v)}
    return set()


def test_a_sheet_item_has_exactly_the_public_fields_and_no_rank_or_score() -> None:
    pool = make_pool()
    document = ev.sheet_document(plans_for(pool)[0], pool)
    items = document["items"]
    assert isinstance(items, list)

    assert all(tuple(item) == ev.SHEET_ITEM_FIELDS for item in items)
    assert [item["item"] for item in items] == list(range(1, len(items) + 1))
    forbidden = {"rank", "score", "method", "rankings", "union", "random", "seen_examples", "label"}
    assert not all_keys(document) & forbidden


def test_the_key_says_where_every_job_stood_and_marks_the_examples() -> None:
    pool = make_pool()
    plans = plans_for(pool)
    key = ev.key_document(plans, pool, CONFIG, seed=1)
    personas = key["personas"]
    assert isinstance(personas, dict)

    alex = personas["alex"]
    assert set(alex["rankings"]) == set(ev.METHODS)
    assert len(alex["seen_examples"]) == 3
    assert alex["items"]["1"] == plans[0].items[0]
    pool_info = key["pool"]
    assert isinstance(pool_info, dict)
    assert pool_info["count"] == len(pool)
    assert pool_info["technical"] == sorted(j.id for j in pool if j.stack)
    assert key["scorer_version"] == "score_v1"


# --- writing -----------------------------------------------------------------------------------


def test_sheets_and_key_go_to_different_directories_and_are_never_overwritten(
    tmp_path: Path,
) -> None:
    pool = make_pool()
    plans = plans_for(pool)

    written = sheet.write_outputs(tmp_path, plans, pool, CONFIG, seed=1)

    assert {p.parent.name for p in written} == {"sheets", "key"}
    assert sorted(p.name for p in (tmp_path / "sheets").iterdir()) == [
        "alex.json",
        "jordan.json",
        "sam.json",
    ]
    assert [p.name for p in (tmp_path / "key").iterdir()] == ["key.json"]
    assert "rankings" not in (tmp_path / "sheets" / "alex.json").read_text(encoding="utf-8")
    with pytest.raises(ScriptError, match="refusing to overwrite"):
        sheet.write_outputs(tmp_path, plans, pool, CONFIG, seed=1)


# --- personas ----------------------------------------------------------------------------------


def write_personas(tmp_path: Path, body: str) -> Path:
    (tmp_path / "cv.txt").write_text("Python developer. Skills: Python, SQL", encoding="utf-8")
    path = tmp_path / "personas.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_persona_file_is_read_with_its_preferences(tmp_path: Path) -> None:
    path = write_personas(
        tmp_path,
        '[[persona]]\nname = "alex"\ncv = "cv.txt"\nseniority_levels = ["senior", "lead"]\n'
        'remote_policies = ["remote"]\nmin_salary = 100000\nsalary_currency = "usd"\n'
        'salary_period = "year"\n',
    )

    (spec,) = ev.load_personas(path)

    assert spec.name == "alex" and spec.kind == "synthetic"
    assert spec.preferences.seniority_levels == (Seniority.SENIOR, Seniority.LEAD)
    assert spec.preferences.salary_currency == "USD"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('[[persona]]\nname = "../evil"\ncv = "cv.txt"\n', "persona name"),
        ('[[persona]]\nname = "a"\ncv = "cv.txt"\nkind = "other"\n', "kind"),
        ('[[persona]]\nname = "a"\ncv = "cv.txt"\nmin_salary = 5\n', "needs min_salary"),
        ('[[persona]]\nname = "a"\ncv = "cv.txt"\nseniority_levels = ["boss"]\n', "not allowed"),
        ('[[persona]]\nname = "a"\ncv = "missing.txt"\n', "cannot read the CV"),
        (
            '[[persona]]\nname = "a"\ncv = "cv.txt"\n[[persona]]\nname = "a"\ncv = "cv.txt"\n',
            "unique",
        ),
    ],
)
def test_a_bad_persona_file_is_refused_with_a_fixed_message(
    tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(ScriptError, match=message):
        ev.load_personas(write_personas(tmp_path, body))


# --- the dry run and the guards ----------------------------------------------------------------


def test_the_dry_run_description_has_counts_only(tmp_path: Path) -> None:
    pool = [make_job(i, title=f"CANARYTITLE{i}") for i in range(52)] + make_pool()[-8:]
    lines = "\n".join(ev.describe(plans_for(pool), pool))

    assert "items" in lines and "random" in lines
    assert "CANARY" not in lines
    assert not any(job.id in lines for job in pool)
    assert "Python" not in lines


def test_build_checks_the_pool_before_anything_else() -> None:
    spec = ev.PersonaSpec("alex", "synthetic", "CANARYCV text", PreferenceFacts())

    with pytest.raises(ScriptError, match="not the frozen one") as raised:
        sheet.build(make_pool(), [spec], EMBEDDER, CONFIG, seed=1)

    assert "CANARYCV" not in str(raised.value)


def test_the_script_refuses_allow_provider() -> None:
    assert sheet.main(["--allow-provider", "--dry-run"]) == 1


def test_a_persona_is_prepared_from_masked_text_and_keeps_no_cv_text() -> None:
    spec = ev.PersonaSpec(
        "alex",
        "synthetic",
        "Skills: Python, SQL\nmail me at someone@example.com",
        PreferenceFacts(),
    )

    (prepared,) = sheet.prepare_personas([spec], EMBEDDER)

    assert "Python" in prepared.skills
    assert "someone@example.com" not in json.dumps(prepared.skills)
    assert len(prepared.vector) == EMBEDDER.dimension
    assert not hasattr(prepared, "cv_text")


def test_the_evaluation_code_reaches_no_llm_and_no_http_client() -> None:
    for name in ("match_eval.py", "eval_match_sheet.py", "calibrate_similarity.py"):
        assert forbidden_imports(ROOT / "app", ROOT / "scripts" / name) == []


# --- a real CV: only under .local/, and never in any output --------------------------------------


def real_file(root: Path, cv_name: str = "real_cv.txt", cv_text: str = "Skills: Elixir") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / cv_name).write_text(cv_text, encoding="utf-8")
    path = root / "real.toml"
    path.write_text(f'[[persona]]\nname = "me"\ncv = "{cv_name}"\n', encoding="utf-8")
    return path


def test_a_real_persona_file_outside_local_is_refused(tmp_path: Path) -> None:
    local = tmp_path / ".local"
    local.mkdir()
    outside = real_file(tmp_path / "elsewhere")

    with pytest.raises(ScriptError, match="inside .local"):
        ev.load_personas(outside, local_root=local)


def test_a_real_persona_file_may_not_name_a_cv_outside_local(tmp_path: Path) -> None:
    local = tmp_path / ".local"
    local.mkdir()
    (tmp_path / "outside.txt").write_text("Skills: Python", encoding="utf-8")
    path = local / "real.toml"
    path.write_text('[[persona]]\nname = "me"\ncv = "../outside.txt"\n', encoding="utf-8")

    with pytest.raises(ScriptError, match="CV file must be inside .local"):
        ev.load_personas(path, local_root=local)


def test_a_real_persona_inside_local_is_read_and_is_a_real_kind(tmp_path: Path) -> None:
    local = tmp_path / ".local"

    (spec,) = ev.load_personas(real_file(local / "cv"), local_root=local)

    assert spec.kind == "real" and spec.name == "me"


def test_a_real_kind_is_refused_in_the_ordinary_persona_file(tmp_path: Path) -> None:
    path = write_personas(tmp_path, '[[persona]]\nname = "a"\ncv = "cv.txt"\nkind = "real"\n')

    with pytest.raises(ScriptError, match="kind must be synthetic"):
        ev.load_personas(path)


def test_the_output_directory_must_be_inside_local(tmp_path: Path) -> None:
    local = tmp_path / ".local"
    local.mkdir()

    ev.check_out_dir(local / "match_eval", local)
    with pytest.raises(ScriptError, match="inside .local"):
        ev.check_out_dir(tmp_path / "out", local)


def test_the_real_cv_text_and_skills_appear_in_no_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    canary = "CANARYREALCV9137"
    cv_text = f"Skills: Elixir, Python\nLine that must stay private: {canary}\n"
    local = tmp_path / ".local"
    (spec,) = ev.load_personas(real_file(local, cv_text=cv_text), local_root=local)
    synthetic = [
        ev.PersonaSpec("alex", "synthetic", "Skills: Python, SQL", PreferenceFacts()),
        ev.PersonaSpec("sam", "synthetic", "Skills: React, TypeScript", PreferenceFacts()),
    ]
    pool = make_pool()

    personas = sheet.prepare_personas([*synthetic, spec], EMBEDDER)
    plans = ev.plan_all(personas, pool, CONFIG, seed=1)
    for line in ev.describe(plans, pool):
        print(line)
    written = sheet.write_outputs(local / "match_eval", plans, pool, CONFIG, seed=1)

    shown = capsys.readouterr().out
    stored = "".join(path.read_text(encoding="utf-8") for path in written)
    for output in (shown, stored):
        assert canary not in output
        assert "Elixir" not in output
        assert "Line that must stay private" not in output
    assert "me (real)" in shown
