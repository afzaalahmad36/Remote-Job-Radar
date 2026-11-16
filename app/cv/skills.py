"""Skills of a CV: found with the job-post technology dictionary, and cleaned when the user edits.

The extractor is `find_technologies` with the two job-post safeguards switched off, because a CV is
written by the person themselves: there is no "recruiter's list" (a CV may name 30 technologies)
and no "customer list" ("trusted by OpenAI"). The dictionary, the aliases ("postgres" is
PostgreSQL) and the rule for ambiguous words ("swift", "rust" count only when capitalised) are the
same ones the job rules use, so a CV skill and a job's stack entry are spelled the same way, which
is what the later matching step compares.

On top of that a CV gets one more rule, for words too ambiguous to count in prose: "Go", "R", "C"
and the ambiguous words of the dictionary ("swift", "rust", "ruby", ...). They count only inside a
skills-style list (see `_list_items`), as a whole list item, so "Go-To-Market" and "go to
production" are never skills while "Skills: Python, Go, SQL" gives all three. This lives here and
not in the dictionary because a job post is written by someone else and prose there is common.

Pure: no I/O, and nothing here ever puts CV text into an error message.
"""

import re

from app.cv.text import clean_cv_text
from app.extraction.stack import AMBIGUOUS_SPELLINGS, canonical_name, find_technologies

MAX_SKILLS = 100  # most skills one CV can hold (also a database check)
MAX_SKILL_CHARS = 50
# A CV is at most 30,000 characters (about 5,000 words); the job-post scan stops at 4,000 words.
_MAX_CV_TOKENS = 12_000

# Words that are a skill only inside a list, with the name they get. The dictionary's ambiguous
# spellings ("swift", "rust", ...) are added in `_list_only_name`.
_LIST_ONLY: dict[str, str] = {"go": "Go", "r": "R", "c": "C"}

_MAX_LINES = 2_000  # lines examined for list context
_MAX_ITEMS_PER_LINE = 200
_MAX_ITEM_WORDS = 3  # a list item is a name, not a clause: "Led go to market" is not an item
_SECTION_LINES = 8  # lines a heading's list can run for, if no blank line comes first
_MIN_KNOWN_IN_LIST = 2  # outside a heading, a list must hold this many other known technologies

_HEADING = re.compile(
    r"^\W{0,3}(?:(?:technical|key|core|programming)\s+)?"
    r"(?:skills|technologies|tech\s+stack|technology\s+stack|languages|tools|stack)"
    r"(?:\s+(?:&|and)\s+technologies)?\W{0,3}(?P<rest>:.*)?$",
    re.IGNORECASE,
)
_SECTION_TITLE = re.compile(r"^[A-Z][A-Z &/]{2,29}$")  # "EXPERIENCE", "EDUCATION": a new section
_SEPARATORS = re.compile(r"[,;/|•·&]|\band\b", re.IGNORECASE)
_BULLET = "-*•–—"
_PARENTHETICAL = re.compile(r"\([^()]{0,40}\)")
_LABEL_WORDS = 3  # "Backend: Python, Go" - at most this many words before the colon


def _list_only_name(item: str) -> str | None:
    """The skill a list item names when it is one of the words that count only in a list."""
    folded = item.casefold()
    if folded in _LIST_ONLY:
        return _LIST_ONLY[folded]
    if folded in AMBIGUOUS_SPELLINGS:
        return canonical_name(folded)
    return None


def _strip_label(line: str) -> str:
    """ "Backend: Python, Go" -> "Python, Go" (a short label before a colon is not an item)."""
    label, colon, rest = line.partition(":")
    return rest if colon and len(label.split()) <= _LABEL_WORDS else line


def _items(line: str) -> list[str]:
    """The items of a comma/slash/"and" separated line: trimmed, parentheses and bullets removed."""
    line = _PARENTHETICAL.sub("", _strip_label(line.lstrip(_BULLET + " \t")))
    parts = _SEPARATORS.split(line)[:_MAX_ITEMS_PER_LINE]
    items = [part.strip(" \t.:") for part in parts if part.strip(" \t.:")]
    if len(items) > 1 and len(items[0].split()) > 1:
        # "Built services in Python, Docker and Go": a lead-in clause sticks to the first item,
        # so only its last word is the first item. (The other items must be short names.)
        items[0] = items[0].split()[-1]
    return items


def _is_list(items: list[str]) -> bool:
    return len(items) >= _MIN_KNOWN_IN_LIST + 1 and all(
        len(item.split()) <= _MAX_ITEM_WORDS for item in items
    )


def _known_count(items: list[str]) -> int:
    """Items that are an unambiguous dictionary technology ("Python"), not a list-only word."""
    return sum(
        1 for item in items if _list_only_name(item) is None and canonical_name(item) is not None
    )


def _section_indexes(lines: list[str]) -> set[int]:
    """Indexes of the lines inside a skills section: a heading line that carries a list after its
    colon ("Skills: Python, Go"), and the lines after a heading until a blank line, a new
    section title or the section limit."""
    inside: set[int] = set()
    for index, line in enumerate(lines):
        match = _HEADING.match(line.strip())
        if match is None:
            continue
        if match["rest"]:
            inside.add(index)
        for follower in range(index + 1, min(index + 1 + _SECTION_LINES, len(lines))):
            stripped = lines[follower].strip()
            if not stripped or _SECTION_TITLE.match(stripped):
                break
            inside.add(follower)
    return inside


def _list_context_skills(text: str) -> list[str]:
    """Ambiguous words that sit as a whole item in a skills list.

    A list is either a line in a skills section (under "Skills", "Technologies", "Tech stack",
    "Languages"...), or any line of short comma/slash separated items that names at least two
    other known technologies. Prose never qualifies: its items are clauses, not names.
    """
    lines = text.splitlines()[:_MAX_LINES]
    inside = _section_indexes(lines)
    found: dict[str, None] = {}
    for index, line in enumerate(lines):
        items = _items(line)
        in_list = index in inside or (_is_list(items) and _known_count(items) >= _MIN_KNOWN_IN_LIST)
        if not in_list:
            continue
        for item in items:
            name = _list_only_name(item)
            if name is not None and len(item.split()) == 1:
                found.setdefault(name, None)
    return list(found)


def find_skills(text: str) -> list[str]:
    """The technologies a CV names, in order of first appearance, without duplicates.

    Dictionary technologies first, then the ambiguous words that sit in a skills list.
    """
    skills = find_technologies(
        text, max_listed=None, skip_references=False, max_tokens=_MAX_CV_TOKENS
    )
    for name in _list_context_skills(text):
        if name not in skills:
            skills.append(name)
    return skills[:MAX_SKILLS]


class SkillError(ValueError):
    """A skill the user sent cannot be stored. The message is a fixed phrase."""


def normalize_skills(skills: list[str]) -> list[str]:
    """What the user's edited list is stored as.

    Whitespace is collapsed and blanks dropped. A name the dictionary knows is spelled the way
    the dictionary spells it ("postgres" becomes "PostgreSQL"), so it matches job stacks; any
    other name is kept as typed, because the dictionary is small and the user knows their skills.
    Duplicates (ignoring case) are dropped, first spelling wins. Order is kept.
    """
    result: list[str] = []
    seen: set[str] = set()
    for raw in skills:
        name = " ".join(clean_cv_text(raw).split())
        if not name:
            continue
        if len(name) > MAX_SKILL_CHARS:
            raise SkillError(f"A skill must be at most {MAX_SKILL_CHARS} characters.")
        name = canonical_name(name) or name
        if name.casefold() not in seen:
            seen.add(name.casefold())
            result.append(name)
    if len(result) > MAX_SKILLS:
        raise SkillError(f"A CV can hold at most {MAX_SKILLS} skills.")
    return result
