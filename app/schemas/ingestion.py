import uuid

from pydantic import BaseModel

from app.services.sync_service import SyncRequestStatus


class SyncAcceptedOut(BaseModel):
    """The answer to "sync this source": the work was accepted, not finished.

    The outcome shows up on the source (`last_synced_at`, `last_error`) and in `GET /jobs`.
    """

    source_id: uuid.UUID
    status: SyncRequestStatus
