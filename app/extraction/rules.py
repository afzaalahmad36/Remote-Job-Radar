"""Deterministic extraction: regexes, keyword lists and a technology dictionary, no I/O.

It runs first, costs nothing, and gives the same answer for the same text. Each field is either
found with a confidence or left empty; an empty field is the signal that a language model may be
worth asking (see merge.py). Posts are untrusted and up to 20,000 characters, so every pattern is
linear: bounded quantifiers, literal anchors, and `str.find` scans (tests feed it hostile input
under a time limit).
"""

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from pydantic import EmailStr, TypeAdapter, ValidationError

from app.core.url_safety import UnsafeUrlError, check_url_syntax
from app.extraction.contacts import deobfuscate, find_emails
from app.extraction.salary import Salary, find_salary
from app.extraction.stack import find_stack
from app.extraction.types import Confidence, Found
from app.ingestion.text import MAX_TEXT_CHARS
from app.models import RemotePolicy, Seniority

MAX_HEADLINE_CHARS = 250
MAX_TITLE_CHARS = 150
MAX_URLS_SCANNED = 100
MAX_EMAIL_CANDIDATES = 20

_EMAIL = TypeAdapter(EmailStr)

# --- title and company ----------------------------------------------------------------------

_ROLE_WORDS = frozenset(
    "engineer engineers developer developers programmer architect designer manager analyst "
    "scientist administrator admin consultant specialist lead director devops sre qa tester "
    "writer marketer recruiter representative officer coordinator assistant intern head "
    "researcher strategist editor accountant agent technician support advocate evangelist "
    "producer generalist associate executive swe dev".split()
)
_NOISE_WORDS = frozenset(
    "remote hybrid onsite on-site worldwide anywhere full-time fulltime part-time contract "
    "freelance hiring urgent new wanted vacancy opening job jobs careers apply europe usa us "
    "uk emea apac global".split()
)
_NOT_COMPANIES = frozenset("we our the this who a an you they it remote".split())
_NOISE_TAG = re.compile(
    r"\[[^\]]{0,40}\]"
    r"|\((?:remote|hybrid|on-?site|hiring|urgent|new|full[- ]time|part[- ]time|contract"
    r"|worldwide|anywhere)\)",
    re.IGNORECASE,
)
_COMPANY_COLON = re.compile(r"(?P<company>[^:|]{2,60}):\s+(?P<title>.{3,200})")
_HEADLINE_SPLIT = re.compile(r"\s+(?:\||–|—|-|@|at)\s+")
_LABELLED_TITLE = re.compile(
    r"^(?:job\s+title|position|role|title)\s*[:\-]\s*(?P<v>.{3,150})$", re.IGNORECASE | re.MULTILINE
)
_LABELLED_COMPANY = re.compile(
    r"^(?:company(?:\s+name)?|employer|organi[sz]ation)\s*[:\-]\s*(?P<v>.{2,80})$",
    re.IGNORECASE | re.MULTILINE,
)
_HIRING = re.compile(
    r"\b(?P<v>[A-Z][\w&.'-]{1,30}(?: [A-Z][\w&.'-]{1,30}){0,3}) (?:is|are) (?:now )?"
    r"(?:hiring|looking for|seeking)\b"
)
_WORD = re.compile(r"[a-z]+")


def _has_role(text: str) -> bool:
    return any(word in _ROLE_WORDS for word in _WORD.findall(text.lower()))


def _tidy(text: str) -> str:
    text = _NOISE_TAG.sub("", text)
    return re.sub(r"\s+", " ", text).strip(" -–—|:,")


def _title_and_company(text: str) -> tuple[Found[str] | None, Found[str] | None]:
    head, _, rest = text.partition("\n")
    # The RSS fetcher stores "title\n\nbody": a blank line after the first line means it is a
    # title, not just the first sentence of a body.
    is_headline = not rest.strip() or rest.startswith("\n")
    title, company = _from_headline(_tidy(head[:MAX_HEADLINE_CHARS]))
    title_found = None
    if title and len(title) <= MAX_TITLE_CHARS:
        title_found = Found(title, Confidence.HIGH if is_headline else Confidence.LOW)
    if title_found is None and (labelled := _LABELLED_TITLE.search(text)):
        title_found = Found(_tidy(labelled["v"]), Confidence.HIGH)
    company_found = Found(company, Confidence.LOW) if company else None
    if labelled_company := _LABELLED_COMPANY.search(text):
        company_found = Found(_tidy(labelled_company["v"]), Confidence.HIGH)
    elif company_found is None:
        company_found = _company_from_body(text)
    return title_found, company_found


def _from_headline(head: str) -> tuple[str | None, str | None]:
    """Split headlines like "Acme: Senior Engineer", "Engineer at Acme", "Engineer | Acme"."""
    if not head:
        return None, None
    colon = _COMPANY_COLON.fullmatch(head)
    if colon and _has_role(colon["title"]):
        if _plausible_company(colon["company"]):
            return _tidy(colon["title"]), _tidy(colon["company"])
        if _tidy(colon["company"]).lower() in _NOISE_WORDS:  # "Hiring: Backend Engineer"
            return _tidy(colon["title"]), None
    parts = [part.strip() for part in _HEADLINE_SPLIT.split(head, maxsplit=2) if part.strip()]
    titles = [part for part in parts if _has_role(part)]
    if not titles:
        return None, None
    others = [part for part in parts if part not in titles and _plausible_company(part)]
    return titles[0], others[0] if others else None


def _plausible_company(name: str) -> bool:
    name = _tidy(name)
    return bool(
        2 <= len(name) <= 60
        and not _has_role(name)
        and name.lower() not in _NOISE_WORDS
        and name.lower() not in _NOT_COMPANIES
        and (name[0].isupper() or name[0].isdigit())
    )


def _company_from_body(text: str) -> Found[str] | None:
    for match in _HIRING.finditer(text[:5000]):
        if match["v"].split()[0].lower() not in _NOT_COMPANIES:
            return Found(match["v"], Confidence.LOW)
    return None


# --- seniority ------------------------------------------------------------------------------

_SENIORITY_WORDS: dict[Seniority, re.Pattern[str]] = {
    Seniority.INTERN: re.compile(r"\b(?:intern(?:ship)?|trainee|apprentice)\b"),
    Seniority.JUNIOR: re.compile(r"\b(?:junior|jr|entry[- ]level|graduate)\b"),
    Seniority.MID: re.compile(r"\b(?:mid[- ]?level|mid|intermediate|midweight)\b"),
    Seniority.SENIOR: re.compile(r"\b(?:senior|sr)\b"),
    Seniority.LEAD: re.compile(r"\blead\b(?!\s+gen)|\bhead of\b"),
    Seniority.PRINCIPAL: re.compile(r"\b(?:principal|staff|distinguished)\b"),
}


def _seniority(title: str | None) -> Found[Seniority] | None:
    """From the title only, and only when it names exactly one level ("Senior/Lead" is unclear)."""
    if not title:
        return None
    levels = {level for level, pattern in _SENIORITY_WORDS.items() if pattern.search(title.lower())}
    return Found(levels.pop(), Confidence.HIGH) if len(levels) == 1 else None


# --- remote policy --------------------------------------------------------------------------

_REMOTE = re.compile(
    r"\b(?:fully|100\s?%|completely|entirely)\s+(?:remote|distributed)\b"
    r"|\bremote[- ](?:first|only|friendly)\b"
    r"|\bwork\s+(?:from\s+anywhere|remotely)\b"
    r"|\bremote\s+(?:position|job|role|work|opportunity)\b"
    r"|\b(?:is|are|be)\s+remote\b|\bwfh\b|\banywhere\s+in\s+the\s+world\b",
    re.IGNORECASE,
)
_HYBRID = re.compile(r"\bhybrid\b", re.IGNORECASE)
_ONSITE = re.compile(r"\bon[- ]?site\b|\bin[- ]office\b|\boffice[- ]based\b", re.IGNORECASE)
_NOT_REMOTE = re.compile(
    r"\b(?:not|no|non)[- ]remote\b|\bremote\s+(?:work\s+)?(?:is\s+)?not\s+(?:possible|available)\b",
    re.IGNORECASE,
)
_NOT_ONSITE = re.compile(r"\b(?:no|not|without)\s+(?:an?\s+)?(?:on[- ]?site|in[- ]office)\b", re.I)
_ANY_REMOTE = re.compile(r"\bremote\b", re.IGNORECASE)


def _remote_policy(text: str) -> Found[RemotePolicy] | None:
    """Explicit phrases give HIGH; a bare "remote" gives LOW; mixed signals give nothing."""
    not_remote = _NOT_REMOTE.search(text) is not None
    signals: set[RemotePolicy] = set()
    if _REMOTE.search(text) and not not_remote:
        signals.add(RemotePolicy.REMOTE)
    if _HYBRID.search(text):
        signals.add(RemotePolicy.HYBRID)
    if not_remote or (_ONSITE.search(text) and not _NOT_ONSITE.search(text)):
        signals.add(RemotePolicy.ONSITE)
    mentions_remote = not not_remote and _ANY_REMOTE.search(text) is not None
    if not signals:
        return Found(RemotePolicy.REMOTE, Confidence.LOW) if mentions_remote else None
    if len(signals) > 1 or (mentions_remote and RemotePolicy.REMOTE not in signals):
        return None  # "hybrid or remote", "remote or on-site": the post does not decide
    return Found(signals.pop(), Confidence.HIGH)


# --- apply email and link -------------------------------------------------------------------

_APPLY_WORDS = (
    "apply", "resume", "cv", "send", "email", "e-mail", "contact", "careers", "hiring",
    "application", "submit",
)  # fmt: skip
_GOOD_LOCAL_PARTS = frozenset(
    "jobs careers hr recruiting recruitment talent hiring apply join".split()
)
_BAD_LOCAL_PARTS = frozenset(
    "noreply no-reply donotreply do-not-reply unsubscribe privacy abuse postmaster".split()
)


def _apply_email(text: str) -> Found[str] | None:
    """The address a candidate is told to write to. Ties between different addresses give None."""
    scored: dict[str, int] = {}
    for span in find_emails(text)[:MAX_EMAIL_CANDIDATES]:
        local = span.address.split("@", 1)[0]
        if local in _BAD_LOCAL_PARTS or span.address in scored:
            continue
        try:
            address = str(_EMAIL.validate_python(span.address)).lower()
        except ValidationError:
            continue
        context = text[max(0, span.start - 60) : span.end + 60].lower()
        scored[address] = sum(word in context for word in _APPLY_WORDS) + 2 * (
            local in _GOOD_LOCAL_PARTS
        )
    if not scored:
        return None
    top = max(scored.values())
    best = [address for address, score in scored.items() if score == top]
    if len(best) > 1:
        return None
    return Found(best[0], Confidence.HIGH if top > 0 else Confidence.LOW)


_ATS_HOSTS = (
    "greenhouse.io", "lever.co", "ashbyhq.com", "workable.com", "smartrecruiters.com",
    "myworkdayjobs.com", "bamboohr.com", "recruitee.com", "teamtailor.com", "breezy.hr",
    "jobs.personio.com", "apply.workable.com", "workatastartup.com", "wellfound.com",
)  # fmt: skip
_NOT_APPLY_HOSTS = (
    "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com", "youtube.com",
    "t.me", "tiktok.com", "reddit.com", "medium.com", "github.com",
)  # fmt: skip
_NOT_APPLY_WORDS = ("unsubscribe", "privacy", "terms", "/cdn-cgi/", ".png", ".jpg", ".gif", ".svg")
_URL_STOP = frozenset(" \t\n\r<>\"'")
_SCHEME = re.compile(r"https?://", re.IGNORECASE)


def _host_in(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def _apply_url(text: str) -> Found[str] | None:
    best: tuple[int, str] | None = None
    position = 0
    for _ in range(MAX_URLS_SCANNED):
        match = _SCHEME.search(text, position)
        if match is None:
            break
        end = match.start()
        while end < len(text) and text[end] not in _URL_STOP and end - match.start() < 2048:
            end += 1
        position = max(end, match.end())
        url = text[match.start() : end].rstrip(".,;:!?)]}")
        score = _url_score(text, match.start(), url)
        if score >= 2 and (best is None or score > best[0]):
            best = (score, url)
    if best is None:
        return None
    return Found(best[1], Confidence.HIGH if best[0] >= 3 else Confidence.LOW)


def _url_score(text: str, start: int, url: str) -> int:
    try:
        host = check_url_syntax(url)
    except UnsafeUrlError:
        return 0
    lowered = url.lower()
    if _host_in(host, _NOT_APPLY_HOSTS) or any(word in lowered for word in _NOT_APPLY_WORDS):
        return 0
    path = urlsplit(url).path.lower()
    return (
        3 * _host_in(host, _ATS_HOSTS)
        + 2 * ("apply" in text[max(0, start - 60) : start].lower())
        + any(word in path for word in ("/apply", "/careers", "/jobs", "/job/"))
    )


# --- everything -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleResult:
    title: Found[str] | None = None
    company: Found[str] | None = None
    seniority: Found[Seniority] | None = None
    remote_policy: Found[RemotePolicy] | None = None
    salary: Found[Salary] | None = None
    apply_email: Found[str] | None = None
    apply_url: Found[str] | None = None
    stack: tuple[str, ...] = ()


def extract_with_rules(text: str) -> RuleResult:
    """Everything the rules can read from one post. Never raises, whatever the text."""
    text = deobfuscate(text[:MAX_TEXT_CHARS]).strip()
    title, company = _title_and_company(text)
    return RuleResult(
        title=title,
        company=company,
        seniority=_seniority(title.value if title else None),
        remote_policy=_remote_policy(text),
        salary=find_salary(text),
        apply_email=_apply_email(text),
        apply_url=_apply_url(text),
        stack=tuple(find_stack(text)),
    )
