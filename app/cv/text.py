"""Cleaning and masking CV text before it is stored. Pure functions: no I/O, no logging.

A CV is personal data. We keep only its text, and we hide what is easiest to abuse: email
addresses, phone numbers and links (a LinkedIn or GitHub URL identifies a person on its own).
A name or an employer's name cannot be found reliably, so they stay; the privacy rules (nothing
reaches a hosted provider, nothing is logged) do not depend on masking being perfect.

Every regex quantifier here is bounded, because CVs are untrusted input.
"""

import re
import unicodedata

from app.extraction.contacts import mask_contact_details

LINK_PLACEHOLDER = "[link]"

# http(s)://..., www....: up to 2,000 characters of the address (a bounded `{1,2000}`, so a very
# long run of non-space characters cannot make the scan slow).
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]{1,2000}", re.IGNORECASE)
# Profile pages are often written without a scheme: "linkedin.com/in/alex-example".
_PROFILE = re.compile(
    r"\b(?:linkedin\.com|github\.com|gitlab\.com|bitbucket\.org)/[^\s<>\"')\]]{1,200}",
    re.IGNORECASE,
)
_BLANK_LINES = re.compile(r"\n{3,}")
_SPACE_RUNS = re.compile(r"[ \t ]{2,}")


def clean_cv_text(text: str) -> str:
    """Normalise text from a PDF or a paste: NFC, \\n line ends, no control characters, no runs of
    spaces, at most one blank line in a row. Line breaks are kept: section headings need them."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    # Drop control characters (category Cc) except newline and tab, and the invisible ones that
    # PDF extraction sometimes leaves inside words (category Cf: soft hyphen, zero-width space).
    text = "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch) not in {"Cc", "Cf"}
    )
    text = _SPACE_RUNS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def mask_cv_text(text: str) -> str:
    """The text with emails, phone numbers and links replaced by placeholders."""
    text = mask_contact_details(text)
    text = _URL.sub(LINK_PLACEHOLDER, text)
    return _PROFILE.sub(LINK_PLACEHOLDER, text)
