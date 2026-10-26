from dataclasses import dataclass

from pydantic import BaseModel

DEFAULT_PAGE_LIMIT = 20
MAX_PAGE_LIMIT = 100  # a client cannot ask for the whole table in one request
MAX_PAGE_OFFSET = 100_000  # very large offsets make the database scan and discard many rows


@dataclass(frozen=True)
class PageParams:
    """The `limit` / `offset` a list endpoint received, already validated."""

    limit: int
    offset: int


class Page[T](BaseModel):
    """Envelope for every list endpoint. `Page[T]` is a generic model (Python 3.12 syntax):
    `Page[SourceRead]` is a page whose `items` are SourceRead objects."""

    items: list[T]
    total: int  # how many rows exist in all, not just on this page
    limit: int
    offset: int
