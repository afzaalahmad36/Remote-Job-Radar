"""Reading a salary out of free text.

A number only counts as money when a currency (a symbol like `$`, or a code like `EUR`) sits next
to it, so "5 years", "10k users" and phone numbers are never read as salaries. The period
(per year, month, ...) is stored exactly as the post states it; nothing is converted.
"""

import re
from dataclasses import dataclass

from app.extraction.types import Confidence, Found
from app.models import SalaryPeriod

MAX_AMOUNTS_SCANNED = 200

_SYMBOL_CURRENCY = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}
_LETTER_PREFIX_CURRENCY = {
    "C": "CAD", "CA": "CAD", "A": "AUD", "AU": "AUD", "NZ": "NZD", "US": "USD", "HK": "HKD",
    "S": "SGD",
}  # fmt: skip
_CODES = (
    "USD EUR GBP CAD AUD NZD CHF SEK NOK DKK PLN CZK HUF RON BGN INR PKR BDT LKR JPY CNY KRW HKD "
    "SGD THB MYR IDR PHP VND AED SAR QAR ILS TRY EGP ZAR NGN KES MAD BRL MXN ARS CLP COP UAH"
).split()
_CODE_ALTERNATIVES = "|".join(_CODES)
_NUMBER = r"(?:\d{1,3}(?:,\d{3}){1,3}|\d{1,3}(?:\.\d{3}){1,3}|\d{1,7}(?:\.\d{1,2})?)"
_MONEY = re.compile(
    rf"(?:(?P<letters>[A-Z]{{1,2}})?(?P<symbol>[$€£₹])"
    rf"|(?<![A-Za-z])(?P<code>{_CODE_ALTERNATIVES})(?![A-Za-z]))?"
    rf"\s?(?P<number>{_NUMBER})(?P<k>\s?[kK](?![A-Za-z]))?"
    rf"(?:\s?(?P<suffix>(?<![A-Za-z])(?:{_CODE_ALTERNATIVES})(?![A-Za-z])))?"
)
_THOUSANDS_COMMA = re.compile(r"\d{1,3}(?:,\d{3})+")
_THOUSANDS_DOT = re.compile(r"\d{1,3}(?:\.\d{3})+")
_RANGE_GAP = re.compile(r"\s{0,2}(?:-|–|—|~|to|and)\s{0,2}")

_PERIOD_WORDS: dict[SalaryPeriod, re.Pattern[str]] = {
    SalaryPeriod.YEAR: re.compile(
        r"\bper\s+(?:year|annum|yr)\b|/\s?(?:yr|year)\b|\ba\s+year\b|\bannual(?:ly)?\b|\byearly\b"
        r"|\bp\.a\.",
        re.IGNORECASE,
    ),
    SalaryPeriod.MONTH: re.compile(
        r"\bper\s+month\b|/\s?(?:mo|month)\b|\ba\s+month\b|\bmonthly\b|\bpcm\b", re.IGNORECASE
    ),
    SalaryPeriod.WEEK: re.compile(
        r"\bper\s+week\b|/\s?(?:wk|week)\b|\ba\s+week\b|\bweekly\b", re.IGNORECASE
    ),
    SalaryPeriod.DAY: re.compile(
        r"\bper\s+day\b|/\s?day\b|\ba\s+day\b|\bdaily\b|\bday\s+rate\b", re.IGNORECASE
    ),
    SalaryPeriod.HOUR: re.compile(
        r"\bper\s+hour\b|/\s?(?:hr|hour)\b|\ban\s+hour\b|\bhourly\b|\bp/h\b", re.IGNORECASE
    ),
}
_SALARY_WORDS = re.compile(
    r"salary|compensation|\bpay\b|\brate\b|remuneration|\bearn|package|\bctc\b|stipend|\bbudget\b",
    re.IGNORECASE,
)
_UP_TO = re.compile(r"(?:up\s*to|max(?:imum)?)\W*$", re.IGNORECASE)
_FROM = re.compile(r"(?:from|starting(?:\s+at)?|min(?:imum)?|at\s+least)\W*$", re.IGNORECASE)

# What one period's amount can sensibly be; outside it the number is something else (a count, an
# id, a typo) and is not reported. No period stated: assume it is a yearly figure.
_PLAUSIBLE: dict[SalaryPeriod | None, tuple[int, int]] = {
    SalaryPeriod.YEAR: (1_000, 10_000_000),
    SalaryPeriod.MONTH: (100, 1_000_000),
    SalaryPeriod.WEEK: (20, 200_000),
    SalaryPeriod.DAY: (5, 50_000),
    SalaryPeriod.HOUR: (1, 5_000),
    None: (1_000, 10_000_000),
}


@dataclass(frozen=True)
class Salary:
    minimum: int | None
    maximum: int | None
    currency: str | None
    period: SalaryPeriod | None


@dataclass(frozen=True)
class _Amount:
    start: int
    end: int
    value: float
    currency: str | None
    has_k: bool


def find_salary(text: str) -> Found[Salary] | None:
    """The best salary the text states, or None. HIGH when a period or a salary word backs it."""
    amounts = _amounts(text)
    best: tuple[int, Found[Salary]] | None = None
    index = 0
    while index < len(amounts):
        first = amounts[index]
        second = amounts[index + 1] if index + 1 < len(amounts) else None
        candidate = None
        if second and _joined(text, first, second):
            candidate = _range(text, first, second)
        if candidate:
            index += 2
        else:
            candidate = _single(text, first)
            index += 1
        if candidate and (best is None or candidate[0] > best[0]):
            best = candidate
    return best[1] if best else None


def _amounts(text: str) -> list[_Amount]:
    amounts: list[_Amount] = []
    for match in _MONEY.finditer(text):
        amounts.append(
            _Amount(
                match.start(),
                match.end(),
                _to_number(match["number"]),
                _currency_of(match),
                bool(match["k"]),
            )
        )
        if len(amounts) >= MAX_AMOUNTS_SCANNED:
            break
    return amounts


def _currency_of(match: re.Match[str]) -> str | None:
    if match["symbol"]:
        letters = match["letters"] or ""
        return _LETTER_PREFIX_CURRENCY.get(letters) or _SYMBOL_CURRENCY[match["symbol"]]
    return match["code"] or match["suffix"] or None


def _to_number(text: str) -> float:
    """The pattern only lets digits, thousands separators and one decimal point through."""
    if _THOUSANDS_COMMA.fullmatch(text):
        return float(text.replace(",", ""))
    if _THOUSANDS_DOT.fullmatch(text):
        return float(text.replace(".", ""))
    return float(text)


def _joined(text: str, first: _Amount, second: _Amount) -> bool:
    gap = text[first.end : second.start]
    return len(gap) <= 8 and _RANGE_GAP.fullmatch(gap) is not None


def _range(text: str, first: _Amount, second: _Amount) -> tuple[int, Found[Salary]] | None:
    if first.currency and second.currency and first.currency != second.currency:
        return None
    low, high = first.value, second.value
    if first.has_k or second.has_k:  # "80-120k" and "80k-120k" both mean thousands
        low = low * 1000 if first.has_k or low < 1000 else low
        high = high * 1000 if second.has_k or high < 1000 else high
    currency = first.currency or second.currency
    if currency is None or low > high:
        return None
    return _candidate(text, first.start, second.end, int(low), int(high), currency)


def _single(text: str, amount: _Amount) -> tuple[int, Found[Salary]] | None:
    if amount.currency is None:
        return None
    value = int(amount.value * 1000 if amount.has_k else amount.value)
    before = text[max(0, amount.start - 14) : amount.start]
    low: int | None = value
    high: int | None = value
    if _UP_TO.search(before):
        low = None
    elif _FROM.search(before) or text[amount.end : amount.end + 1] == "+":
        high = None
    return _candidate(text, amount.start, amount.end, low, high, amount.currency)


def _candidate(
    text: str, start: int, end: int, low: int | None, high: int | None, currency: str
) -> tuple[int, Found[Salary]] | None:
    period = _period_near(text, start, end)
    floor, ceiling = _PLAUSIBLE[period]
    if any(v is not None and not floor <= v <= ceiling for v in (low, high)):
        return None
    backed_by_word = _SALARY_WORDS.search(text[max(0, start - 40) : end + 40]) is not None
    score = 2 * (period is not None) + backed_by_word
    confidence = Confidence.HIGH if score else Confidence.LOW
    return score, Found(Salary(low, high, currency, period), confidence)


def _period_near(text: str, start: int, end: int) -> SalaryPeriod | None:
    """The period word closest after the amount, else one just before it ("annual salary: ...")."""
    for window in (text[end : end + 30], text[max(0, start - 30) : start]):
        found = [
            (match.start(), period)
            for period, pattern in _PERIOD_WORDS.items()
            if (match := pattern.search(window))
        ]
        if found:
            return min(found, key=lambda item: item[0])[1]
    return None
