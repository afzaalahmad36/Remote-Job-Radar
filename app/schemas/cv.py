import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.cv import CvFormat


class CvRead(BaseModel):
    """The caller's own CV. `text` is what was stored (emails, phones and links masked), so the
    user can check exactly what the system holds."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source_format: CvFormat
    page_count: int | None
    char_count: int
    text: str
    created_at: datetime
    updated_at: datetime
