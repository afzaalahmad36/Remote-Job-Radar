import json
import logging
import sys

from app.core.logging import JsonFormatter


def test_json_formatter_outputs_one_json_object() -> None:
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert "time" in payload


def test_json_formatter_includes_exception_traceback() -> None:
    try:
        raise ValueError("bad value")
    except ValueError:
        record = logging.LogRecord(
            "app.test", logging.ERROR, __file__, 1, "failed", (), sys.exc_info()
        )

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: bad value" in payload["exception"]
