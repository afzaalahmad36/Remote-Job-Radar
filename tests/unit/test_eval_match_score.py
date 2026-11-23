"""The offline scorer of the matching evaluation (scripts/eval_match_score.py).

Invented pool, personas and labels: no database, no model, no network. The frozen pool's hash is
patched to the hash of the invented pool, because the real pool is not available in a test.
"""

import hashlib
import json
from collections.abc import Collection
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts import eval_match_label as label
from scripts import eval_match_score as score
from scripts import match_eval as ev
from scripts.db_guard import ScriptError
from tests.unit.test_embedding_isolation import forbidden_imports
from tests.unit.test_eval_match_sheet import CONFIG, make_pool, persona, three

ROOT = Path(__file__).resolve().parent.parent.parent
WHEN = datetime(2026, 12, 1, 10, 0, 0, tzinfo=UTC)


class Setup:
    """An output folder with a key, the sheets and (once `label_all` ran) the labels."""

    def __init__(self, out: Path, plans: list[ev.PersonaPlan], pool: list[ev.PoolJob]) -> None:
        self.out, self.plans, self.pool = out, plans, pool

    def plan(self, name: str) -> ev.PersonaPlan:
        return next(p for p in self.plans if p.name == name)

    def label_all(self, name: str, relevant: Collection[str], maybe: Collection[str] = ()) -> None:
        plan = self.plan(name)
        items = [{"item": n, "job_id": j} for n, j in enumerate(plan.items, start=1)]
        labels = {
            n: "relevant" if j in relevant else "maybe" if j in maybe else "no"
            for n, j in enumerate(plan.items, start=1)
        }
        label.save_labels(self.out / "labels" / f"{name}.json", name, items, labels)


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Setup:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / ".local" / "match_eval"
    pool = make_pool()
    personas = [*three(), persona("me", "real")]
    plans = ev.plan_all(personas, pool, CONFIG, seed=1)
    frozen = ev.pool_hash([j.id for j in pool])
    for module in (score, label, ev):
        monkeypatch.setattr(module, "PROTOCOL_POOL_HASH", frozen)
    (out / "key").mkdir(parents=True)
    (out / "key" / "key.json").write_text(
        json.dumps(ev.key_document(plans, pool, CONFIG, seed=1)), encoding="utf-8"
    )
    (out / "sheets").mkdir()
    for plan in plans:
        (out / "sheets" / f"{plan.name}.json").write_text(
            json.dumps(ev.sheet_document(plan, pool)), encoding="utf-8"
        )
    return Setup(out, plans, pool)


def top(setup: Setup, name: str, method: str, count: int = 5) -> list[str]:
    return setup.plan(name).rankings[method][:count]


def label_synthetic(setup: Setup, per_persona_relevant: int) -> None:
    for name in ("alex", "sam", "jordan"):
        setup.label_all(name, top(setup, name, "hybrid_a")[:per_persona_relevant])


def run(setup: Setup) -> tuple[int, str]:
    lines: list[str] = []
    code = score.score(setup.out, lines.append, now=lambda: WHEN)
    return code, "\n".join(lines)


# --- the metrics -------------------------------------------------------------------------------


def test_precision_counts_the_relevant_jobs_in_the_top_k() -> None:
    labels = {"a": "relevant", "b": "maybe", "c": "no", "d": "relevant", "e": "no", "f": "relevant"}
    ranking = ["a", "b", "c", "d", "e", "f"]

    assert score.hits(ranking[:5], labels, frozenset({"relevant"})) == (2, 5)
    assert score.hits(ranking[:5], labels, frozenset({"relevant", "maybe"})) == (3, 5)


def test_a_top_k_with_an_unlabelled_job_is_not_measured() -> None:
    assert score.top_labelled(["a", "b", "c"], {"a": "no", "b": "no"}, 3) is None
    assert score.top_labelled(["a", "b"], {"a": "no", "b": "no"}, 3) is None
    assert score.top_labelled(["a", "b", "c"], {"a": "no", "b": "no", "c": "no"}, 3) == [
        "a",
        "b",
        "c",
    ]


def test_the_technical_stratum_takes_the_top_k_of_the_technical_jobs_only() -> None:
    labels = {i: "relevant" for i in "abcd"}

    assert score.top_labelled(list("abcd"), labels, 2, frozenset({"b", "d"})) == ["b", "d"]


def test_p_at_10_is_measured_only_when_the_whole_top_10_is_labelled(setup: Setup) -> None:
    ranking = top(setup, "alex", "hybrid_a", 10)
    labelled = {j: "no" for j in ranking[:7]}

    partial = score.method_result(ranking, labelled, {}, frozenset())
    full = score.method_result(ranking, {j: "no" for j in ranking}, {}, frozenset())

    assert partial.p10_relevant is None and partial.p10_relevant_maybe is None
    assert full.p10_relevant == (0, 10)


def test_a_key_whose_top_5_is_not_labelled_is_refused(setup: Setup) -> None:
    with pytest.raises(ScriptError, match="do not belong together"):
        score.method_result(top(setup, "alex", "hybrid_a"), {}, {}, frozenset())


def test_the_seen_examples_are_reported_with_the_documented_label_and_without(
    setup: Setup,
) -> None:
    plan = setup.plan("alex")
    example = next(iter(plan.examples))
    ranking = [example, *[i for i in plan.items if i != example]][:5]
    labels = dict.fromkeys(plan.items, "no")  # the labeller said no to everything
    documented = {example: "relevant"}

    result = score.method_result(ranking, labels, documented, frozenset())

    assert result.p5_relevant == (0, 5)
    assert result.p5_documented == (1, 5)
    assert result.p5_without_examples == (0, 4)


# --- the pass rule and calibration B -----------------------------------------------------------


def fake_result(
    name: str, a: int, b: int, newest: int, kind: str = "synthetic"
) -> score.PersonaResult:
    def method(hits_: int) -> score.MethodResult:
        h = (hits_, 5)
        return score.MethodResult(h, h, None, None, None, h, h)

    methods = {m: method(0) for m in ev.METHODS}
    methods.update(hybrid_a=method(a), hybrid_b=method(b), newest_first=method(newest))
    return score.PersonaResult(name, kind, methods, (0, 10), (0, 10), 0, 0)


def test_the_rule_needs_nine_of_fifteen_and_strictly_more_than_newest_first() -> None:
    def verdict(a: int, newest: int) -> bool:
        results = [fake_result(n, a, a, newest) for n in "xyz"]
        return score.verdict(results)[0]

    assert verdict(3, 0)  # 9 of 15
    assert not verdict(2, 0)  # 6 of 15
    assert not verdict(3, 3)  # equal to newest first
    assert verdict(4, 3)


def test_b_is_recommended_only_with_a_gain_of_two_and_no_loss_in_any_persona() -> None:
    gain = [fake_result("x", 3, 5, 0), fake_result("y", 3, 3, 0), fake_result("z", 3, 3, 0)]
    one = [fake_result("x", 3, 4, 0), fake_result("y", 3, 3, 0), fake_result("z", 3, 3, 0)]
    loss = [fake_result("x", 3, 5, 0), fake_result("y", 3, 5, 0), fake_result("z", 3, 1, 0)]

    assert score.verdict(gain)[1]["b_recommended"] is True
    assert score.verdict(one)[1]["b_recommended"] is False
    assert score.verdict(loss)[1]["b_recommended"] is False
    assert score.verdict(loss)[1]["b_gain"] == 2


def test_only_three_synthetic_personas_can_be_pooled() -> None:
    with pytest.raises(ScriptError, match="exactly three"):
        score.verdict([fake_result("x", 3, 3, 0)])


# --- the whole run -----------------------------------------------------------------------------


def test_a_run_that_meets_the_rule_passes_and_exits_zero(setup: Setup) -> None:
    label_synthetic(setup, 4)  # 4 of the top 5 of A are relevant, in each persona: 12 of 15

    code, text = run(setup)

    assert code == 0
    assert "VERDICT: PASS" in text
    assert "hybrid_a" in text and "12/15" in text
    assert score.FOOTING_NOTE in text


def test_a_run_below_the_rule_fails_and_says_so(setup: Setup) -> None:
    label_synthetic(setup, 2)  # 6 of 15

    code, text = run(setup)

    assert code == 3
    assert "VERDICT: FAIL" in text


def test_a_run_that_only_ties_newest_first_fails(setup: Setup) -> None:
    for name in ("alex", "sam", "jordan"):
        setup.label_all(name, setup.plan(name).items)  # everything relevant: A 15, newest 15

    code, text = run(setup)

    assert code == 3 and "VERDICT: FAIL" in text


def test_a_synthetic_persona_must_be_completely_labelled(setup: Setup) -> None:
    label_synthetic(setup, 4)
    plan = setup.plan("sam")
    items = [{"item": n, "job_id": j} for n, j in enumerate(plan.items, start=1)]
    label.save_labels(setup.out / "labels" / "sam.json", "sam", items, {1: "no"})

    with pytest.raises(ScriptError, match="not completely labelled"):
        run(setup)


def test_a_real_persona_without_labels_is_skipped_and_with_labels_is_reported_apart(
    setup: Setup,
) -> None:
    label_synthetic(setup, 4)
    _, without = run(setup)
    setup.label_all("me", top(setup, "me", "hybrid_a")[:5])

    lines: list[str] = []
    score.score(setup.out, lines.append, now=lambda: WHEN.replace(minute=5))

    assert "me (real): no labels file, skipped" in without
    assert any("Real CV, reported apart" in line for line in lines)
    pooled_lines = [line for line in lines if line.startswith("  hybrid_a") and "/15" in line]
    assert pooled_lines and "12/15" in pooled_lines[0]  # the real CV did not enter the pool


def test_a_key_of_another_pool_is_refused(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(score, "PROTOCOL_POOL_HASH", "0" * 64)

    with pytest.raises(ScriptError, match="not for the frozen pool"):
        run(setup)


def test_labels_that_do_not_belong_to_the_key_are_refused(setup: Setup) -> None:
    label_synthetic(setup, 4)
    path = setup.out / "labels" / "alex.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["labels"][0]["job_id"] = "another-job"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ScriptError, match="does not match the sheet"):
        run(setup)


def test_an_output_folder_outside_local_is_refused(setup: Setup, tmp_path: Path) -> None:
    with pytest.raises(ScriptError, match="inside .local"):
        score.score(tmp_path / "elsewhere", lambda _line: None)


def test_a_changed_worked_example_is_counted(setup: Setup) -> None:
    plan = setup.plan("alex")
    example, documented = next(iter(plan.examples.items()))
    other = "no" if documented != "no" else "relevant"
    items = [{"item": n, "job_id": j} for n, j in enumerate(plan.items, start=1)]
    labels = {
        n: other if j == example else plan.examples.get(j, "no")
        for n, j in enumerate(plan.items, start=1)
    }
    label.save_labels(setup.out / "labels" / "alex.json", "alex", items, labels)
    for name in ("sam", "jordan"):
        setup.label_all(name, ())

    _, text = run(setup)

    assert "worked examples: 1 relabelled by the labeller" in text


# --- what is printed and recorded --------------------------------------------------------------


def test_output_and_run_record_hold_counts_only(setup: Setup) -> None:
    label_synthetic(setup, 4)
    setup.label_all("me", top(setup, "me", "hybrid_a")[:3])

    _, text = run(setup)

    record_text = next((setup.out / "runs").iterdir()).read_text(encoding="utf-8")
    for output in (text, record_text):
        assert not any(job.id in output for job in setup.pool)
        assert not any(job.title in output for job in setup.pool)
        assert "job_id" not in output and '"skills"' not in output
    assert "Python" not in record_text


def test_the_run_record_has_the_constants_the_versions_and_the_hashes_of_the_files(
    setup: Setup,
) -> None:
    label_synthetic(setup, 4)

    run(setup)

    (path,) = list((setup.out / "runs").iterdir())
    record = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "run-20261201T100000Z.json"
    assert record["scorer_version"] == "score_v1"
    assert record["config"]["failed_filter_cap"] == 30
    assert record["pool"]["hash"] == ev.pool_hash([j.id for j in setup.pool])
    assert record["pass_rule"]["verdict"] == "PASS"
    assert record["footing"] == score.FOOTING_NOTE
    key_bytes = (setup.out / "key" / "key.json").read_bytes()
    assert record["file_sha256"]["key/key.json"] == hashlib.sha256(key_bytes).hexdigest()
    assert {"labels/alex.json", "labels/sam.json", "labels/jordan.json"} <= set(
        record["file_sha256"]
    )
    assert set(record["personas"]) == {"alex", "sam", "jordan"}
    assert {"b_floor", "b_ceiling", "methods"} <= set(record["personas"]["alex"])


def test_a_run_record_is_never_overwritten(setup: Setup) -> None:
    label_synthetic(setup, 4)
    run(setup)

    with pytest.raises(FileExistsError):
        run(setup)


def test_the_real_cv_never_appears_in_the_printed_report_or_the_run_record(
    setup: Setup,
) -> None:
    label_synthetic(setup, 4)
    setup.label_all("me", ())
    canary_cv = "CANARYREALCV4421"  # never given to anything here: the scorer has no CV text at all

    _, text = run(setup)

    record_text = next((setup.out / "runs").iterdir()).read_text(encoding="utf-8")
    assert canary_cv not in text + record_text
    assert "me (real)" in text
    key_text = (setup.out / "key" / "key.json").read_text(encoding="utf-8")
    assert "cv_text" not in key_text and '"skills"' not in key_text


def test_the_scorer_is_offline_and_reaches_no_llm_and_no_http_client() -> None:
    source = (ROOT / "scripts" / "eval_match_score.py").read_text(encoding="utf-8")

    assert forbidden_imports(ROOT / "app", ROOT / "scripts" / "eval_match_score.py") == []
    for name in ("get_settings", "read_only_engine", "build_embedder", "create_async_engine"):
        assert name not in source
