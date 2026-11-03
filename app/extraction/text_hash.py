"""Fingerprint of a post's text, used so identical posts reuse one extraction."""

import hashlib


def normalize_post_text(text: str) -> str:
    """Trim and collapse every run of whitespace to one space. Case is NOT folded.

    `str.split()` with no argument splits on any Unicode whitespace (spaces, tabs, newlines,
    non-breaking spaces) and drops empty parts, so joining with " " does both steps at once.
    Case stays, because "Go" and "GO" can differ in meaning and a wrong reuse costs more than a
    missed one.
    """
    return " ".join(text.split())


def raw_text_hash(text: str) -> str:
    """SHA-256 (64 hex characters) of the normalized text: the `jobs.raw_text_hash` value."""
    return hashlib.sha256(normalize_post_text(text).encode("utf-8")).hexdigest()
