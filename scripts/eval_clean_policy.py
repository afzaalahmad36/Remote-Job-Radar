"""The keep-or-drop rule for stack-label entries, used by `eval_clean_labels.py --policy`.

One rule: keep only a named product, language, framework, library, database, cloud service or
protocol that someone would list on a CV. Drop practices, methods, domains, standards and abstract
nouns, and fragments that are not an entry at all.

The signals, in order (the first that applies decides, and its name is shown as the reason):

1. fragment     the entry starts with "and " / "or ", or is on the `fragments` list
2. deny-list    a technique, not a product (LLMs, RAG, MLOps, CI/CD, ...)
3. allow-list   a known product the stack dictionary lacks
4. standard     SOC 2, ISO 27001, PCI DSS and the like: a standard, not a technology
5. dictionary   the stack dictionary (canonical names and aliases) knows it
6. concept rule an entry already flagged as not looking like a product (a concept, a description,
                too long, a sentence, an unknown lowercase word) is dropped

An entry that no signal decides, and that was not flagged, is left as it is. The lists live in a
TOML file (`scripts/eval_clean_policy.toml`) so they can be read and changed without touching code.
Pure functions plus one file read; no database, no network.
"""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from app.extraction.stack import canonical_name
from scripts.db_guard import ScriptError

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "eval_clean_policy.toml"
_KEYS = frozenset({"allow", "deny", "fragments"})
_STANDARD = re.compile(r"\b(soc\s*\d|iso\s*\d+|pci|gdpr|hipaa|nist)\b", re.IGNORECASE)


def normal(name: str) -> str:
    """The form names are compared in: case-folded, single spaces."""
    return " ".join(name.casefold().split())


@dataclass(frozen=True)
class Policy:
    allow: frozenset[str]
    deny: frozenset[str]
    fragments: frozenset[str]


@dataclass(frozen=True)
class Decision:
    keep: bool
    reason: str  # fragment | deny-list | allow-list | concept rule: standard | dictionary | ...


def load_policy(path: Path) -> Policy:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ScriptError(f"cannot read {path}: {type(error).__name__}") from None
    except tomllib.TOMLDecodeError:
        raise ScriptError(f"{path} is not valid TOML") from None
    unknown = set(data) - _KEYS
    if unknown:
        raise ScriptError(f"{path}: unknown key(s) {', '.join(sorted(unknown))}")
    lists: dict[str, frozenset[str]] = {}
    for key in sorted(_KEYS):
        values = data.get(key, [])
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ScriptError(f"{path}: '{key}' must be a list of strings")
        lists[key] = frozenset(normal(v) for v in values)
    clash = lists["allow"] & (lists["deny"] | lists["fragments"])
    if clash:
        raise ScriptError(
            f"{path}: on both the allow-list and a drop list: {', '.join(sorted(clash))}"
        )
    return Policy(lists["allow"], lists["deny"], lists["fragments"])


def decide(name: str, flag: str | None, policy: Policy) -> Decision | None:
    """Keep or drop `name`, with the reason. None = no signal applies and nothing was flagged.

    `flag` is why the cleaner would otherwise ask about the entry (None if it would not).
    """
    key = normal(name)
    if key.startswith(("and ", "or ")) or key in policy.fragments:
        return Decision(False, "fragment")
    if key in policy.deny:
        return Decision(False, "deny-list")
    if key in policy.allow:
        return Decision(True, "allow-list")
    if _STANDARD.search(name):
        return Decision(False, "concept rule: standard")
    if flag is None:
        return None
    if canonical_name(name) is not None:
        return Decision(True, "dictionary")
    return Decision(False, f"concept rule: {flag}")
