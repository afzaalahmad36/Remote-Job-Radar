"""The keep-or-drop rule for stack-label entries, used by `eval_clean_labels.py --policy`.

One rule: keep only a named product, language, framework, library, database, cloud service or
protocol that someone would list on a CV. Drop practices, methods, domains, standards and abstract
nouns, and fragments that are not an entry at all. The default is DROP: an entry is kept only when
something positively says it is a known product.

Before deciding, a leading "and " / "or " and a leading skill adjective (strong, solid, deep,
hands-on, expert, proficient) are stripped, so "and AWS" is "AWS" and "strong SQL" is "SQL". The
lists are checked first on the entry as written, so "Deep Learning" is caught before "deep" could be
stripped from it.

An entry that glues several products together is split first (`expand`), and each part is decided
on its own: an alias that names two products ("React Vite"), a slash without spaces when at least
one side is a known product ("AWS/Cloud Infrastructure" is "AWS" plus a dropped "Cloud
Infrastructure"), and " or " between two known products ("AWS or GCP"). A "known" product is on
the allow-list, an alias, or in the stack dictionary.

The signals, in order (the first that applies decides, and its name is shown as the reason):

1. fragment     on the `fragments` list, or nothing is left after stripping
2. deny-list    a technique, method or non-engineering tool (LLMs, RAG, Agile, CRM, ...)
3. allow-list   a known product the stack dictionary lacks
4. alias        another spelling of a known product (Google Cloud Platform is GCP): kept under the
                canonical name, so it dedupes against that name in the same post
5. standard     SOC 2, ISO 27001, PCI DSS and the like: a standard, not a technology
6. dictionary   the stack dictionary (canonical names and aliases) knows it
7. otherwise    DROP, reason "not a known product"

The lists live in a TOML file (`scripts/eval_clean_policy.toml`) so they can be read and changed
without touching code. Pure functions plus one file read; no database, no network.
"""

import re
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from app.extraction.stack import canonical_name
from scripts.db_guard import ScriptError

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "eval_clean_policy.toml"
_LISTS = ("allow", "deny", "fragments")
_KEYS = frozenset({*_LISTS, "aliases"})
_STANDARD = re.compile(r"\b(soc\s*\d|iso\s*\d+|pci|gdpr|hipaa|nist)\b", re.IGNORECASE)
_OR = re.compile(r"\s+or\s+", re.IGNORECASE)
_CONNECTORS = frozenset({"and", "or"})
SKILL_ADJECTIVES = frozenset({"strong", "solid", "deep", "hands-on", "expert", "proficient"})


def normal(name: str) -> str:
    """The form names are compared in: case-folded, single spaces."""
    return " ".join(name.casefold().split())


@dataclass(frozen=True)
class Policy:
    allow: frozenset[str]
    deny: frozenset[str]
    fragments: frozenset[str]
    # normal(alias) -> the product(s) it stands for; more than one means "split into these"
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    keep: bool
    reason: str  # fragment | deny-list | allow-list | alias of X | concept rule: standard | ...
    name: str  # the entry after stripping (and after an alias became its canonical name)
    rules: tuple[str, ...] = ()  # what was stripped from the front


def _string_list(path: Path, key: str, values: object) -> list[str]:
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ScriptError(f"{path}: '{key}' must be a list of strings")
    return values


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
    lists = {
        key: frozenset(normal(v) for v in _string_list(path, key, data.get(key, [])))
        for key in _LISTS
    }
    aliases = _load_aliases(path, data.get("aliases", {}))
    clash = (lists["allow"] | set(aliases)) & (lists["deny"] | lists["fragments"])
    if clash:
        names = ", ".join(sorted(clash))
        raise ScriptError(f"{path}: on both a keep list and a drop list: {names}")
    return Policy(lists["allow"], lists["deny"], lists["fragments"], aliases)


def _load_aliases(path: Path, raw: object) -> dict[str, tuple[str, ...]]:
    message = f"{path}: 'aliases' must map names to a name or a non-empty list of names"
    if not isinstance(raw, dict):
        raise ScriptError(message)
    aliases = {}
    for alias, target in raw.items():
        targets = [target] if isinstance(target, str) else target
        if (
            not isinstance(targets, list)
            or not targets
            or not all(isinstance(t, str) for t in targets)
        ):
            raise ScriptError(message)
        aliases[normal(alias)] = tuple(targets)
    return aliases


def strip_leading(name: str) -> tuple[str, tuple[str, ...]]:
    """Remove leading "and" / "or" and skill adjectives. Returns the rest and what was removed."""
    words = name.split()
    removed = []
    while words and (words[0].casefold() in _CONNECTORS | SKILL_ADJECTIVES):
        removed.append(f"stripped '{words.pop(0)}'")
    return " ".join(words), tuple(removed)


def _by_lists(name: str, policy: Policy) -> Decision | None:
    key = normal(name)
    if key in policy.fragments:
        return Decision(False, "fragment", name)
    if key in policy.deny:
        return Decision(False, "deny-list", name)
    if key in policy.allow:
        return Decision(True, "allow-list", name)
    if len(policy.aliases.get(key, ())) == 1:  # an alias for several products is split first
        canonical = policy.aliases[key][0]
        return Decision(True, f"alias of {canonical}", canonical)
    return None


def _by_knowledge(name: str) -> Decision:
    if _STANDARD.search(name):
        return Decision(False, "concept rule: standard", name)
    if canonical_name(name) is not None:
        return Decision(True, "dictionary", name)
    return Decision(False, "not a known product", name)


def decide(name: str, policy: Policy) -> Decision:
    """Keep or drop `name`, with the reason. Every entry gets a decision."""
    decision = _by_lists(name, policy)  # as written first: "Deep Learning" must not lose "Deep"
    if decision is not None:
        return decision
    stripped, rules = strip_leading(name)
    if not stripped:
        return Decision(False, "fragment", name, rules)
    decision = _by_lists(stripped, policy) or _by_knowledge(stripped)
    return replace(decision, rules=rules)


def is_known(name: str, policy: Policy) -> bool:
    """On the allow-list, an alias, or in the stack dictionary."""
    key = normal(name)
    return key in policy.allow or key in policy.aliases or canonical_name(name) is not None


def _split_on_slash(name: str, policy: Policy) -> list[str]:
    parts = [part.strip() for part in name.split("/")]
    if len(parts) > 1 and all(parts) and any(is_known(part, policy) for part in parts):
        return parts
    return []


def _split_on_or(name: str, policy: Policy) -> list[str]:
    parts = [part.strip() for part in _OR.split(name)]
    if len(parts) > 1 and all(parts) and all(is_known(part, policy) for part in parts):
        return parts
    return []


def expand(name: str, policy: Policy) -> list[tuple[str, tuple[str, ...]]]:
    """Split an entry that glues several products together; each part is decided on its own.

    Returns (part, rules) pairs; an entry that is not split comes back as itself, untouched, with no
    rules. An entry the lists or the dictionary already know as a whole is never split ("CI/CD").
    """
    if _by_lists(name, policy) is not None or canonical_name(name) is not None:
        return [(name, ())]
    rest, stripped = strip_leading(name)
    key = normal(rest)
    if len(policy.aliases.get(key, ())) > 1:
        return [(part, (*stripped, "split alias")) for part in policy.aliases[key]]
    for rule, parts in (
        ("split '/'", _split_on_slash(rest, policy)),
        ("split 'or'", _split_on_or(rest, policy)),
    ):
        if parts:
            return [(part, (*stripped, rule)) for part in parts]
    return [(name, ())]
