"""The embedding worker's place in the stack: what it can reach, and what must not reach it.

The worker that handles CV text has no route to the internet (an `internal` Docker network), only
it reads the `embed` queue, and the model's weights come from the image, never from the network.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.core.config import Settings
from app.tasks.celery_app import DEFAULT_QUEUE, EMBED_QUEUE, build_config

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(
        (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    return loaded


def test_the_embed_network_has_no_route_out(compose: dict[str, Any]) -> None:
    assert compose["networks"]["embed"] == {"internal": True}


def test_worker_embed_is_only_on_the_internal_network(compose: dict[str, Any]) -> None:
    services = compose["services"]

    assert services["worker-embed"]["networks"] == ["embed"]


def test_db_and_redis_are_on_both_networks_so_worker_embed_can_reach_them(
    compose: dict[str, Any],
) -> None:
    services = compose["services"]

    for name in ("db", "redis"):
        assert set(services[name]["networks"]) == {"default", "embed"}


def test_no_other_service_is_on_the_internal_network(compose: dict[str, Any]) -> None:
    """The api, the main worker and beat keep the default network (and its internet access)."""
    on_embed = {
        name
        for name, service in compose["services"].items()
        if "embed" in service.get("networks", [])
    }

    assert on_embed == {"db", "redis", "worker-embed"}


def test_worker_embed_runs_one_task_at_a_time_and_only_the_embed_queue(
    compose: dict[str, Any],
) -> None:
    command = compose["services"]["worker-embed"]["command"]

    assert "--concurrency=1" in command
    assert "--queues=embed" in command
    assert any(arg.startswith("--max-memory-per-child=") for arg in command)
    assert compose["services"]["worker-embed"]["mem_limit"] == "512m"


def test_the_main_worker_does_not_read_the_embed_queue(compose: dict[str, Any]) -> None:
    command = compose["services"]["worker"]["command"]

    assert "--queues=celery" in command
    assert not any("embed" in arg for arg in command)


def test_the_queue_names_in_compose_match_the_celery_config(compose: dict[str, Any]) -> None:
    assert f"--queues={EMBED_QUEUE}" in compose["services"]["worker-embed"]["command"]
    assert f"--queues={DEFAULT_QUEUE}" in compose["services"]["worker"]["command"]


def test_worker_embed_has_no_language_model_configured(compose: dict[str, Any]) -> None:
    """Whatever .env says, the process that reads CV text has no provider and no key."""
    environment = compose["services"]["worker-embed"]["environment"]

    assert environment["LLM_PROVIDER"] == "none"
    assert environment["LLM_EXTRACTION_ENABLED"] == "false"
    assert environment["GEMINI_API_KEY"] == ""


def test_only_the_two_model_tasks_are_routed_to_the_embed_queue(settings: Settings) -> None:
    routes = build_config(settings)["task_routes"]

    assert routes == {"embed_job": {"queue": EMBED_QUEUE}, "embed_cv": {"queue": EMBED_QUEUE}}
    assert build_config(settings)["task_default_queue"] == DEFAULT_QUEUE


def test_the_embedding_sweep_is_scheduled_on_the_default_queue(settings: Settings) -> None:
    schedule = build_config(settings)["beat_schedule"]

    assert schedule["sweep-embeddings"]["task"] == "sweep_embeddings"
    assert "sweep_embeddings" not in build_config(settings)["task_routes"]


# --- the image ---------------------------------------------------------------------------------


def _dockerfile() -> str:
    return (ROOT / "Dockerfile").read_text(encoding="utf-8")


def test_the_model_download_is_off_by_default_and_on_in_compose(compose: dict[str, Any]) -> None:
    default = re.search(r"^ARG EMBEDDING_MODEL_DOWNLOAD=(\d)$", _dockerfile(), re.MULTILINE)
    args = compose["services"]["worker-embed"]["build"]["args"]

    assert default is not None and default.group(1) == "0"  # the CI build downloads nothing
    assert args["EMBEDDING_MODEL_DOWNLOAD"] == "${EMBEDDING_MODEL_DOWNLOAD:-1}"  # local: on


def test_the_ci_docker_build_passes_no_model_argument() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "EMBEDDING_MODEL_DOWNLOAD" not in workflow


def test_the_dockerfile_revision_is_a_commit_hash_equal_to_the_settings_default() -> None:
    match = re.search(r"^ARG EMBEDDING_MODEL_REVISION=([0-9a-f]{40})$", _dockerfile(), re.MULTILINE)

    assert match is not None
    assert match.group(1) == Settings.model_fields["embedding_model_revision"].default


def test_the_runtime_image_is_offline_and_reads_the_weights_from_models() -> None:
    dockerfile = _dockerfile()

    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "FASTEMBED_CACHE_PATH=/models" in dockerfile
    assert "COPY --from=model /models /models" in dockerfile
