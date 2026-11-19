"""Score the matching evaluation from the answer key and the labels (run by hand, never by CI).

The protocol is docs/MATCHING.md, section 10: the metrics, the pass rule and the rule for the
CV-job calibration were fixed before any label existed, and this script only applies them.

It is OFFLINE: it reads `<out>/key/key.json` and `<out>/labels/<persona>.json` (and only hashes the
sheets). No database, no model, no provider, no network. It prints counts and opaque numbers only
(never a job id, a title, a CV, a skill or a note), and writes a run record, also counts only, to
`<out>/runs/` with the constants, the versions and the SHA-256 of every file it read.

Every persona of kind `synthetic` must be completely labelled. A `real` persona (the owner's own CV)
is optional, is reported apart and never takes part in the pass rule.

The baselines are NOT on equal footing (docs/MATCHING.md, note of 2026-11-19): the hybrids and
skills-only apply the persona's hard filters, embedding-only, newest-first and random do not. The
report says so.

    uv run python -m scripts.eval_match_score [--out .local/match_eval]
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.db_guard import ScriptError
from scripts.eval_match_label import load_labels
from scripts.match_eval import (
    LOCAL_DIR,
    METHODS,
    PROTOCOL_POOL_HASH,
    TOP_N,
    check_out_dir,
    valid_persona_name,
)

DEFAULT_OUT = LOCAL_DIR / "match_eval"
PASS_POOLED_P5 = 9  # of 15: at least this many relevant jobs in the pooled top 5 of method A
B_MIN_GAIN = 2  # B is recommended only if it adds at least this many relevant jobs, losing in none
FOOTING_NOTE = (
    "Not on equal footing: hybrid A, hybrid B and skills only apply the persona's hard filters "
    "(and the cap of 30); embedding only, newest first and random do not."
)

Hits = tuple[int, int]  # (relevant jobs, slots)


@dataclass(frozen=True)
class MethodResult:
    p5_relevant: Hits
    p5_relevant_maybe: Hits
    p10_relevant: Hits | None  # None: some of the top 10 were not labelled
    p10_relevant_maybe: Hits | None
    technical_p5_relevant: Hits | None  # top 5 of the technical jobs only; None if unlabelled
    p5_documented: Hits  # seen examples counted with their documented label
    p5_without_examples: Hits  # seen examples left out (fewer slots)


@dataclass(frozen=True)
class PersonaResult:
    name: str
    kind: str
    methods: Mapping[str, MethodResult]
    random_relevant: Hits  # relevant among the labelled random jobs
    random_relevant_maybe: Hits
    changed_examples: int  # seen examples whose label the labeller changed
    seen_examples_in_top5_of_a: int


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hits(ids: Sequence[str], labels: Mapping[str, str], wanted: frozenset[str]) -> Hits:
    return sum(1 for i in ids if labels[i] in wanted), len(ids)


def top_labelled(
    ranking: Sequence[str], labels: Mapping[str, str], k: int, only: frozenset[str] | None = None
) -> list[str] | None:
    """The top k of a ranking (restricted to `only` if given), or None if any is unlabelled."""
    chosen = [i for i in ranking if only is None or i in only][:k]
    if len(chosen) < k or any(i not in labels for i in chosen):
        return None
    return chosen


def method_result(
    ranking: Sequence[str],
    labels: Mapping[str, str],
    documented: Mapping[str, str],
    technical: frozenset[str],
) -> MethodResult:
    relevant, relevant_maybe = frozenset({"relevant"}), frozenset({"relevant", "maybe"})
    top5 = top_labelled(ranking, labels, TOP_N)
    if (
        top5 is None
    ):  # the sheet puts every top 5 on the labelled items; a key that disagrees is bad
        raise ScriptError("a top-5 job has no label: the key and the labels do not belong together")
    top10 = top_labelled(ranking, labels, 2 * TOP_N)
    tech5 = top_labelled(ranking, labels, TOP_N, technical)
    with_documented = {**labels, **documented}
    kept = [i for i in top5 if i not in documented]
    return MethodResult(
        p5_relevant=hits(top5, labels, relevant),
        p5_relevant_maybe=hits(top5, labels, relevant_maybe),
        p10_relevant=hits(top10, labels, relevant) if top10 else None,
        p10_relevant_maybe=hits(top10, labels, relevant_maybe) if top10 else None,
        technical_p5_relevant=hits(tech5, labels, relevant) if tech5 else None,
        p5_documented=hits(top5, with_documented, relevant),
        p5_without_examples=hits(kept, labels, relevant),
    )


def persona_result(
    name: str, entry: Mapping[str, Any], labels: Mapping[str, str], technical: frozenset[str]
) -> PersonaResult:
    documented = {str(k): str(v) for k, v in dict(entry["seen_examples"]).items()}
    rankings = dict(entry["rankings"])
    methods = {m: method_result(rankings[m], labels, documented, technical) for m in METHODS}
    random_ids = [str(i) for i in entry["random"]]
    top_a = rankings["hybrid_a"][:TOP_N]
    return PersonaResult(
        name=name,
        kind=str(entry["kind"]),
        methods=methods,
        random_relevant=hits(random_ids, labels, frozenset({"relevant"})),
        random_relevant_maybe=hits(random_ids, labels, frozenset({"relevant", "maybe"})),
        changed_examples=sum(1 for i, label in documented.items() if labels[i] != label),
        seen_examples_in_top5_of_a=sum(1 for i in top_a if i in documented),
    )


def pooled(results: Sequence[PersonaResult], method: str) -> Hits:
    return (
        sum(r.methods[method].p5_relevant[0] for r in results),
        sum(r.methods[method].p5_relevant[1] for r in results),
    )


def verdict(synthetic: Sequence[PersonaResult]) -> tuple[bool, dict[str, object]]:
    """The pass rule of the protocol, and the rule for recommending calibration B."""
    if len(synthetic) != 3:
        raise ScriptError("the protocol pools exactly three synthetic personas")
    a, newest = pooled(synthetic, "hybrid_a"), pooled(synthetic, "newest_first")
    b = pooled(synthetic, "hybrid_b")
    b_loses_somewhere = any(
        r.methods["hybrid_b"].p5_relevant[0] < r.methods["hybrid_a"].p5_relevant[0]
        for r in synthetic
    )
    passed = a[0] >= PASS_POOLED_P5 and a[0] > newest[0]
    b_recommended = (b[0] - a[0]) >= B_MIN_GAIN and not b_loses_somewhere
    return passed, {
        "pooled_p5_hybrid_a": list(a),
        "pooled_p5_newest_first": list(newest),
        "rule": f"pooled P@5 of A >= {PASS_POOLED_P5}/15 and strictly more than newest first",
        "b_gain": b[0] - a[0],
        "b_loses_in_a_persona": b_loses_somewhere,
        "b_recommended": b_recommended,
    }


# --- reading -----------------------------------------------------------------------------------


def load_key(path: Path) -> dict[str, Any]:
    try:
        key: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ScriptError("cannot read the answer key: build the sheets first") from None
    pool = key.get("pool")
    if not isinstance(pool, dict) or pool.get("hash") != PROTOCOL_POOL_HASH:
        raise ScriptError("the answer key is not for the frozen pool")
    if not isinstance(key.get("personas"), dict):
        raise ScriptError("the answer key has no personas")
    return key


def read_persona_labels(
    out: Path, name: str, kind: str, entry: Mapping[str, Any]
) -> dict[str, str] | None:
    """Job id -> label, complete. A real persona with no labels file is skipped (None)."""
    if not valid_persona_name(name):
        raise ScriptError("the key holds an invalid persona name")
    path = out / "labels" / f"{name}.json"
    if not path.exists() and kind == "real":
        return None
    items = [{"item": int(n), "job_id": job_id} for n, job_id in dict(entry["items"]).items()]
    by_item = load_labels(path, name, items)
    if len(by_item) != len(items):
        raise ScriptError(
            f"persona {name} is not completely labelled ({len(by_item)} of {len(items)})"
        )
    return {str(item["job_id"]): by_item[int(item["item"])] for item in items}


# --- reporting ---------------------------------------------------------------------------------


def fmt(value: Hits | None) -> str:
    return "not measured" if value is None else f"{value[0]}/{value[1]}"


def persona_lines(result: PersonaResult) -> list[str]:
    lines = [
        f"{result.name} ({result.kind})",
        f"  {'method':<16}{'P@5 rel':>9}{'rel+maybe':>11}{'P@10 rel':>14}{'tech P@5':>14}"
        f"{'documented':>12}{'no examples':>13}",
    ]
    for method in METHODS:
        m = result.methods[method]
        lines.append(
            f"  {method:<16}{fmt(m.p5_relevant):>9}{fmt(m.p5_relevant_maybe):>11}"
            f"{fmt(m.p10_relevant):>14}{fmt(m.technical_p5_relevant):>14}"
            f"{fmt(m.p5_documented):>12}{fmt(m.p5_without_examples):>13}"
        )
    lines.append(
        f"  random: {fmt(result.random_relevant)} relevant, "
        f"{fmt(result.random_relevant_maybe)} relevant+maybe among the labelled random jobs"
    )
    lines.append(
        f"  worked examples: {result.changed_examples} relabelled by the labeller; "
        f"{result.seen_examples_in_top5_of_a} in the top 5 of A"
    )
    return lines


def report(
    synthetic: Sequence[PersonaResult],
    real: Sequence[PersonaResult],
    passed: bool,
    detail: Mapping[str, object],
) -> list[str]:
    lines = ["Matching evaluation (docs/MATCHING.md section 10)", FOOTING_NOTE, ""]
    for result in synthetic:
        lines += persona_lines(result) + [""]
    lines.append("Pooled P@5, relevant only, over the synthetic personas (out of 15)")
    for method in METHODS:
        lines.append(f"  {method:<16}{fmt(pooled(synthetic, method)):>8}")
    lines.append(
        f"Calibration B vs A: gain {detail['b_gain']}, loses in a persona: "
        f"{detail['b_loses_in_a_persona']} -> "
        + ("recommended" if detail["b_recommended"] else "no evidence that it helps; A stays")
    )
    lines.append(f"VERDICT: {'PASS' if passed else 'FAIL'} ({detail['rule']})")
    for result in real:
        lines += ["", "Real CV, reported apart, not part of the rule (counts only)"]
        lines += persona_lines(result)
    return lines


def method_json(result: MethodResult) -> dict[str, object]:
    return {
        "p5_relevant": list(result.p5_relevant),
        "p5_relevant_maybe": list(result.p5_relevant_maybe),
        "p10_relevant": None if result.p10_relevant is None else list(result.p10_relevant),
        "technical_p5_relevant": (
            None if result.technical_p5_relevant is None else list(result.technical_p5_relevant)
        ),
        "p5_documented": list(result.p5_documented),
        "p5_without_examples": list(result.p5_without_examples),
    }


def run_record(
    key: Mapping[str, Any],
    results: Sequence[PersonaResult],
    passed: bool,
    detail: Mapping[str, object],
    hashes: Mapping[str, str],
    when: datetime,
) -> dict[str, object]:
    """Constants, versions, file hashes and counts. No job id, label, text, skill or CV."""
    personas = dict(key["personas"])
    return {
        "written_at": when.isoformat(),
        "protocol": key["protocol"],
        "scorer_version": key["scorer_version"],
        "seed": key["seed"],
        "config": key["config"],
        "pool": {"count": dict(key["pool"])["count"], "hash": PROTOCOL_POOL_HASH},
        "pass_rule": {"verdict": "PASS" if passed else "FAIL", **detail},
        "footing": FOOTING_NOTE,
        "file_sha256": dict(hashes),
        "personas": {
            r.name: {
                "kind": r.kind,
                "b_floor": personas[r.name]["b_floor"],
                "b_ceiling": personas[r.name]["b_ceiling"],
                "methods": {m: method_json(r.methods[m]) for m in METHODS},
                "random_relevant": list(r.random_relevant),
                "random_relevant_maybe": list(r.random_relevant_maybe),
                "changed_examples": r.changed_examples,
            }
            for r in results
        },
    }


def write_run_record(out: Path, record: Mapping[str, object], when: datetime) -> Path:
    path = out / "runs" / f"run-{when.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return path


def score(
    out: Path, show: Callable[[str], None], now: Callable[[], datetime] = lambda: datetime.now(UTC)
) -> int:
    check_out_dir(out)
    key_path = out / "key" / "key.json"
    key = load_key(key_path)
    technical = frozenset(str(i) for i in dict(key["pool"])["technical"])
    hashes = {"key/key.json": sha256_of(key_path)}
    synthetic: list[PersonaResult] = []
    real: list[PersonaResult] = []
    for name, entry in dict(key["personas"]).items():
        kind = str(entry["kind"])
        labels = read_persona_labels(out, name, kind, entry)
        if labels is None:
            show(f"{name} ({kind}): no labels file, skipped")
            continue
        hashes[f"labels/{name}.json"] = sha256_of(out / "labels" / f"{name}.json")
        if (out / "sheets" / f"{name}.json").exists():
            hashes[f"sheets/{name}.json"] = sha256_of(out / "sheets" / f"{name}.json")
        (synthetic if kind == "synthetic" else real).append(
            persona_result(name, entry, labels, technical)
        )
    passed, detail = verdict(synthetic)
    for line in report(synthetic, real, passed, detail):
        show(line)
    when = now()
    path = write_run_record(
        out, run_record(key, [*synthetic, *real], passed, detail, hashes, when), when
    )
    show(f"run record: {path.relative_to(out)}")
    return 0 if passed else 3


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score the matching evaluation (offline).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return score(parse_args(argv).out, print)
    except ScriptError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
