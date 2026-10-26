import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Format each log record as one JSON object per line (easy to search in log tools)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, str] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logging(env: str) -> None:
    """Send all logs to stdout as JSON; DEBUG in development, INFO elsewhere."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]  # replace, so calling this twice doesn't duplicate lines
    root.setLevel(logging.DEBUG if env == "development" else logging.INFO)
