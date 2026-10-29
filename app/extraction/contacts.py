"""Finding and hiding contact details (emails, phone numbers) in a post.

Everything here scans with `str.find` and bounded loops or with regexes whose every quantifier is
bounded. Posts are untrusted and up to 20,000 characters long, and a regex like `[\\w.]+@` is
quadratic on a long run of word characters with no `@` after it.
"""

import re
import string
from dataclasses import dataclass

MAX_EMAIL_LENGTH = 254
MAX_LOCAL_PART = 64
MAX_EMAILS_SCANNED = 200  # `@` signs examined per text

_LOCAL_CHARS = frozenset(string.ascii_letters + string.digits + "._%+-")
_DOMAIN_CHARS = frozenset(string.ascii_letters + string.digits + ".-")
_DOMAIN = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.){1,8}[a-z]{2,24}")

# "name [at] example [dot] com" and friends. Bounded spaces keep the scan linear.
_AT = re.compile(r"\s{0,3}[\[({]\s{0,3}at\s{0,3}[\])}]\s{0,3}", re.IGNORECASE)
_DOT = re.compile(r"\s{0,3}[\[({]\s{0,3}dot\s{0,3}[\])}]\s{0,3}", re.IGNORECASE)

_PHONE = re.compile(
    r"\+\d[\d\s().-]{7,18}\d"  # +44 20 7946 0958
    r"|\(\d{2,4}\)[\s.-]?\d[\d\s.-]{5,12}\d"  # (415) 555-0100
    r"|\b\d{3}[\s.-]\d{3}[\s.-]\d{4}\b"  # 415-555-0100
)

EMAIL_PLACEHOLDER = "[email]"
PHONE_PLACEHOLDER = "[phone]"


@dataclass(frozen=True)
class EmailSpan:
    start: int
    end: int
    address: str  # lowercase


def deobfuscate(text: str) -> str:
    """Turn `jobs [at] acme [dot] com` into `jobs@acme.com`."""
    return _DOT.sub(".", _AT.sub("@", text))


def find_emails(text: str) -> list[EmailSpan]:
    """Every email-shaped address in the text, in order. Shape only; EmailStr judges the rest."""
    spans: list[EmailSpan] = []
    position = 0
    for _ in range(MAX_EMAILS_SCANNED):
        at = text.find("@", position)
        if at == -1:
            break
        position = at + 1
        span = _email_around(text, at)
        if span is not None:
            spans.append(span)
    return spans


def _email_around(text: str, at: int) -> EmailSpan | None:
    left = at
    while left > 0 and at - left < MAX_LOCAL_PART and text[left - 1] in _LOCAL_CHARS:
        left -= 1
    right = at + 1
    while right < len(text) and right - at <= MAX_EMAIL_LENGTH and text[right] in _DOMAIN_CHARS:
        right += 1
    local = text[left:at].lstrip("._%+-")
    domain = text[at + 1 : right].rstrip(".-").lower()  # a trailing dot is end-of-sentence
    if not local or not _DOMAIN.fullmatch(domain):
        return None
    start = at - len(local)
    end = at + 1 + len(domain)
    return EmailSpan(start, end, f"{local.lower()}@{domain}")


def mask_contact_details(text: str) -> str:
    """The text with emails and phone numbers replaced by placeholders.

    Used before a post is sent to a hosted provider: a free tier may let people read what it is
    sent, and a recruiter's address is personal data even when it is public. Best effort for
    phone numbers (formats vary a lot); emails are matched by shape.
    """
    text = deobfuscate(text)
    pieces: list[str] = []
    cursor = 0
    for span in find_emails(text):
        pieces.append(text[cursor : span.start])
        pieces.append(EMAIL_PLACEHOLDER)
        cursor = span.end
    pieces.append(text[cursor:])
    return _PHONE.sub(PHONE_PLACEHOLDER, "".join(pieces))
