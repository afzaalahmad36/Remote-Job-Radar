"""Guards for the properties of docker-compose.yml that must not silently change."""

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(
        (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    return loaded


@pytest.fixture(scope="module")
def services(compose: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = compose["services"]
    return result


def test_the_stack_has_exactly_these_services(services: dict[str, Any]) -> None:
    assert set(services) == {"db", "redis", "api", "worker", "beat"}


def test_only_the_api_publishes_a_port_and_only_on_localhost(services: dict[str, Any]) -> None:
    published = {name: service["ports"] for name, service in services.items() if "ports" in service}

    assert published == {"api": ["127.0.0.1:8000:8000"]}


def test_nothing_sets_a_container_name(services: dict[str, Any], compose: dict[str, Any]) -> None:
    assert all("container_name" not in service for service in services.values())
    assert compose["name"] == "remote-job-radar"


def test_api_worker_and_beat_share_one_image_and_environment(services: dict[str, Any]) -> None:
    app_services = [services[name] for name in ("api", "worker", "beat")]

    assert len({service["image"] for service in app_services}) == 1
    assert all(service["build"] == "." for service in app_services)
    for service in app_services:  # inside the network, the hosts are the service names
        assert service["environment"]["DATABASE_URL"].endswith("@db:5432/jobradar")
        assert service["environment"]["REDIS_URL"] == "redis://redis:6379/0"


def test_only_the_api_runs_the_images_default_command_which_migrates(
    services: dict[str, Any],
) -> None:
    assert (
        "command" not in services["api"]
    )  # the Dockerfile CMD: alembic upgrade head, then uvicorn
    for name in ("worker", "beat"):
        command = " ".join(services[name]["command"])
        assert "alembic" not in command and "uvicorn" not in command


def test_the_dockerfile_default_command_is_the_one_that_migrates() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "alembic upgrade head" in dockerfile


@pytest.mark.parametrize(("name", "subcommand"), [("worker", "worker"), ("beat", "beat")])
def test_worker_and_beat_run_the_celery_app(
    services: dict[str, Any], name: str, subcommand: str
) -> None:
    command = services[name]["command"]

    assert command[:4] == ["celery", "-A", "app.tasks.celery_app", subcommand]


@pytest.mark.parametrize("name", ["worker", "beat"])
def test_worker_and_beat_wait_for_the_database_redis_and_the_migrating_api(
    services: dict[str, Any], name: str
) -> None:
    depends_on = services[name]["depends_on"]

    assert depends_on == {
        "db": {"condition": "service_healthy"},
        "redis": {"condition": "service_healthy"},
        "api": {"condition": "service_healthy"},  # the api has migrated before it is healthy
    }


def test_the_worker_concurrency_is_explicit_configurable_and_defaults_to_two(
    services: dict[str, Any],
) -> None:
    flags = [arg for arg in services["worker"]["command"] if arg.startswith("--concurrency")]

    assert flags == ["--concurrency=${CELERY_WORKER_CONCURRENCY:-2}"]


def test_the_concurrency_variable_is_documented_in_the_example_env_file() -> None:
    example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "CELERY_WORKER_CONCURRENCY=2" in example


def test_there_is_a_single_beat_and_it_is_not_scaled(services: dict[str, Any]) -> None:
    beat = services["beat"]

    assert "scale" not in beat
    assert "replicas" not in beat.get("deploy", {})
    assert sum("beat" in " ".join(s.get("command", [])) for s in services.values()) == 1


def test_beat_keeps_its_schedule_file_somewhere_the_non_root_user_can_write(
    services: dict[str, Any],
) -> None:
    assert "--schedule=/tmp/celerybeat-schedule" in services["beat"]["command"]


def test_containers_run_as_the_images_non_root_user(services: dict[str, Any]) -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "USER app" in dockerfile
    assert all("user" not in service for service in services.values())  # none overrides it to root


def test_the_worker_has_a_real_healthcheck_and_gets_time_to_finish_its_tasks(
    services: dict[str, Any],
) -> None:
    worker = services["worker"]

    assert "inspect ping" in " ".join(worker["healthcheck"]["test"])
    assert worker["stop_grace_period"] == "100s"  # longer than the 90 s hard task time limit


def test_the_database_and_redis_still_publish_no_ports(services: dict[str, Any]) -> None:
    assert "ports" not in services["db"] and "ports" not in services["redis"]
