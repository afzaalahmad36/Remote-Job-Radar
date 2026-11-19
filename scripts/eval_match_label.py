"""Label one persona's blind sheet by hand (run by hand, never by CI or app code).

The protocol is docs/MATCHING.md, section 10. Read the relevance scale there first. This tool shows
each job of the persona's sheet, in the sheet's shuffled order, with its public fields and the start
of its post, and asks for a label: relevant, maybe or no.

It is **blind by construction**: it reads only `sheets/<persona>.json`, which holds no rank, no
score and no method, and it never opens the answer key (a test checks the source). It also never
asks the database, the model or a provider. While you label a persona do not open `GET /matches`.

One file per persona: `<out>/labels/<persona>.json`, holding the item number, the job id and the
label, nothing else. The file is saved after every label (written aside and renamed, so a crash
cannot leave half a file), and running the tool again resumes at the first unlabelled item.

Commands at the prompt: `r` relevant, `m` maybe, `n` no, `b` go back and relabel the previous item,
`q` quit (everything is already saved).

    uv run python -m scripts.eval_match_label alex [--out .local/match_eval]
"""

import argparse
import json
import os
import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from scripts.db_guard import ScriptError
from scripts.match_eval import (
    DEFAULT_PERSONAS,
    LABELS,
    LOCAL_DIR,
    PROTOCOL_POOL_HASH,
    SHEET_ITEM_FIELDS,
    check_out_dir,
    valid_persona_name,
)

DEFAULT_OUT = LOCAL_DIR / "match_eval"
ANSWERS = {"r": "relevant", "m": "maybe", "n": "no"}
SCALE = (
    "relevant = you would apply (main work matches skills and seniority, no preference violated)\n"
    "maybe    = adjacent (domain or part of the stack matches, or one level off, or one key gap)\n"
    "no       = a different field or role, or a stated preference is violated"
)


def _number(item: Mapping[str, object]) -> int:
    return int(str(item["item"]))


def load_sheet(path: Path, persona: str) -> list[dict[str, object]]:
    """The items of a sheet, after checking it is this persona's sheet of the frozen pool."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ScriptError(
            f"cannot read the sheet {path.name}: build it with eval_match_sheet"
        ) from None
    if document.get("persona") != persona or document.get("pool_hash") != PROTOCOL_POOL_HASH:
        raise ScriptError("the sheet is not this persona's sheet of the frozen pool")
    items = document.get("items")
    if not isinstance(items, list) or any(tuple(i) != SHEET_ITEM_FIELDS for i in items):
        raise ScriptError("the sheet has an unexpected shape")
    return items


def load_labels(path: Path, persona: str, items: Sequence[Mapping[str, object]]) -> dict[int, str]:
    """The labels made so far (item number -> label); an unknown or mismatched entry is refused."""
    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        entries = document["labels"]
        if document["persona"] != persona:
            raise ScriptError("the labels file belongs to another persona")
        job_of = {int(i["item"]): str(i["job_id"]) for i in items}  # type: ignore[call-overload]
        labels = {}
        for entry in entries:
            number, label = int(entry["item"]), str(entry["label"])
            if job_of.get(number) != entry["job_id"] or label not in LABELS:
                raise ScriptError("the labels file does not match the sheet")
            labels[number] = label
    except (OSError, ValueError, KeyError, TypeError):
        raise ScriptError(f"cannot read the labels file {path.name}") from None
    return labels


def save_labels(
    path: Path, persona: str, items: Sequence[Mapping[str, object]], labels: Mapping[int, str]
) -> None:
    """Write the labels aside and rename, so the file is always whole."""
    job_of = {int(i["item"]): str(i["job_id"]) for i in items}  # type: ignore[call-overload]
    document = {
        "persona": persona,
        "pool_hash": PROTOCOL_POOL_HASH,
        "labels": [{"item": n, "job_id": job_of[n], "label": labels[n]} for n in sorted(labels)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def show_item(
    item: Mapping[str, object], position: int, total: int, show: Callable[[str], None]
) -> None:
    """The public fields and the start of the post; nothing that says how a method ranked it."""
    salary = [
        item["salary_min"],
        item["salary_max"],
        item["salary_currency"],
        item["salary_period"],
    ]
    stack = item["stack"]
    show("")
    show("=" * 78)
    show(f"item {position} of {total}")
    show(f"title:    {item['title']}")
    show(f"company:  {item['company'] or '-'}")
    show(f"level:    {item['seniority'] or '-'}    remote: {item['remote_policy'] or '-'}")
    show(f"stack:    {', '.join(stack) if isinstance(stack, list) and stack else '-'}")
    show("salary:   " + (" ".join(str(s) for s in salary if s is not None) or "-"))
    show("-" * 78)
    show(str(item["text"]))
    show("-" * 78)


def label_session(
    items: Sequence[Mapping[str, object]],
    labels: dict[int, str],
    save: Callable[[Mapping[int, str]], None],
    ask: Callable[[str], str],
    show: Callable[[str], None],
) -> bool:
    """Ask for each unlabelled item in sheet order. True when every item is labelled."""
    numbers = [int(i["item"]) for i in items]  # type: ignore[call-overload]
    by_number = {int(i["item"]): i for i in items}  # type: ignore[call-overload]
    index = next((k for k, n in enumerate(numbers) if n not in labels), len(numbers))
    while index < len(numbers):
        number = numbers[index]
        show_item(by_number[number], index + 1, len(numbers), show)
        answer = ask(f"[{len(labels)}/{len(numbers)} labelled] r / m / n / b(ack) / q(uit): ")
        answer = answer.strip().lower()
        if answer == "q":
            return False
        if answer == "b":
            index = max(0, index - 1)
            continue
        if answer not in ANSWERS:
            show("not understood: type r, m, n, b or q")
            continue
        labels[number] = ANSWERS[answer]
        save(labels)
        index += 1
    return True


def preference_line(persona: str, *files: Path | None) -> str | None:
    """The persona's stated preferences (not the CV), read from a persona file if it names them."""
    for path in files:
        if path is None or not path.is_file():
            continue
        try:
            entries = tomllib.loads(path.read_text(encoding="utf-8")).get("persona", [])
        except (OSError, tomllib.TOMLDecodeError):
            continue
        for entry in entries:
            if isinstance(entry, dict) and entry.get("name") == persona:
                parts = [
                    f"seniority {', '.join(entry.get('seniority_levels', [])) or 'any'}",
                    f"remote policy {', '.join(entry.get('remote_policies', [])) or 'any'}",
                ]
                if entry.get("min_salary") is not None:
                    parts.append(
                        f"minimum salary {entry['min_salary']} {entry.get('salary_currency')}"
                        f" per {entry.get('salary_period')}"
                    )
                return "; ".join(parts)
    return None


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label one persona's blind sheet.")
    parser.add_argument("persona")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--personas", type=Path, default=DEFAULT_PERSONAS)
    parser.add_argument("--real-personas", type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace, ask: Callable[[str], str], show: Callable[[str], None]) -> int:
    check_out_dir(args.out)
    if not valid_persona_name(args.persona):
        raise ScriptError("a persona name is lower case letters, digits, - or _")
    items = load_sheet(args.out / "sheets" / f"{args.persona}.json", args.persona)
    labels_path = args.out / "labels" / f"{args.persona}.json"
    labels = load_labels(labels_path, args.persona, items)
    show(f"persona: {args.persona}   ({len(labels)} of {len(items)} already labelled)")
    preferences = preference_line(args.persona, args.personas, args.real_personas)
    show("preferences: " + (preferences or "(see your persona file)"))
    show(SCALE)
    finished = label_session(
        items,
        labels,
        lambda current: save_labels(labels_path, args.persona, items, current),
        ask,
        show,
    )
    show("every item is labelled" if finished else "stopped; run again to resume")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv), input, print)
    except (ScriptError, EOFError) as error:
        print(
            f"error: {error}"
            if isinstance(error, ScriptError)
            else "input closed; progress is saved",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
