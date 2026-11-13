from fastapi import APIRouter, Depends, Request, Response, status

from app.api_deps import CurrentUserDep, SessionDep, SettingsDep, rate_limit_cv
from app.core.exceptions import CvTooLargeError
from app.schemas.cv import CvRead
from app.services import cv_service
from app.services.cv_service import CvLimits

# Every endpoint takes CurrentUserDep, so all of them answer 401 without a valid access token.
router = APIRouter(prefix="/cv", tags=["cv"])

# The body is the file itself (not multipart/form-data). Starlette's multipart parser would
# write any upload over 1 MB to a temporary file on disk; reading the raw body ourselves keeps
# the CV in memory only and lets us stop at the byte limit.
_UPLOAD_BODY_DOC = {
    "requestBody": {
        "required": True,
        "content": {
            "application/pdf": {"schema": {"type": "string", "format": "binary"}},
            "text/plain": {"schema": {"type": "string"}},
        },
    }
}


async def read_limited_body(request: Request, max_bytes: int) -> bytes:
    """The request body, or CvTooLargeError as soon as it is known to exceed `max_bytes`."""
    too_large = CvTooLargeError(f"The CV is larger than {max_bytes // 1024} KB.")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > max_bytes:
        raise too_large
    body = bytearray()
    async for chunk in request.stream():  # a chunk at a time, so a huge body is never buffered
        body += chunk
        if len(body) > max_bytes:
            raise too_large
    return bytes(body)


@router.put("", dependencies=[Depends(rate_limit_cv)], openapi_extra=_UPLOAD_BODY_DOC)
async def put_cv(
    request: Request, user: CurrentUserDep, session: SessionDep, settings: SettingsDep
) -> CvRead:
    """Upload (or replace) your CV: send the PDF or plain text itself as the request body, with
    `Content-Type: application/pdf` or `text/plain`. Only the text is kept, with emails, phone
    numbers and links masked; the file is discarded."""
    data = await read_limited_body(request, settings.cv_max_file_bytes)
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    limits = CvLimits(
        max_pages=settings.cv_max_pages,
        max_chars=settings.cv_max_chars,
        pdf_timeout_seconds=settings.cv_pdf_timeout_seconds,
        pdf_memory_limit_mb=settings.cv_pdf_memory_limit_mb,
    )
    cv = await cv_service.save_cv(session, user.id, data, content_type, limits)
    return CvRead.model_validate(cv)


@router.get("")
async def get_cv(user: CurrentUserDep, session: SessionDep) -> CvRead:
    return CvRead.model_validate(await cv_service.get_cv(session, user.id))


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_cv(user: CurrentUserDep, session: SessionDep) -> Response:
    """Delete your CV (its text included) permanently."""
    await cv_service.delete_cv(session, user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
