from pydantic import BaseModel


class SyncOut(BaseModel):
    """What a manual sync did."""

    fetched: int
    inserted: int
    skipped: int
    error: str | None
