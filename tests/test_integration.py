import subprocess
import time
from http import HTTPStatus
from pathlib import Path
from typing import Iterable

import pytest
import requests


COMPOSE_FILE = Path(__file__).resolve().parent.parent / "docker" / "docker-compose.yml"
BACKEND_URL = "http://localhost:8000"
FIXTURE_FASTQ = Path(__file__).resolve().parent.parent / "fastq_dataset" / "real_sample.fastq"


def _run_compose(args: Iterable[str], check: bool = True) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), *args]
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


@pytest.fixture(scope="module")
def compose_up():
    up_args = ["up", "--build", "--detach", "postgres", "backend"]
    result = _run_compose(up_args, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"docker compose up failed:\n{result.stdout}")
    try:
        yield
    finally:
        down_args = ["down", "--volumes"]
        _run_compose(down_args, check=False)


def wait_for_backend(timeout: int = 180) -> None:
    deadline = time.time() + timeout
    health_url = f"{BACKEND_URL}/health"
    while time.time() < deadline:
        try:
            res = requests.get(health_url, timeout=5)
            if res.status_code == HTTPStatus.OK:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError("Backend health check did not become available in time")


@pytest.mark.integration
def test_pipeline_integration_smoke(compose_up):
    wait_for_backend()

    payload = {"name": "integration", "description": "smoke"}
    res = requests.post(f"{BACKEND_URL}/jobs", json=payload, timeout=10)
    assert res.status_code == HTTPStatus.OK, res.text
    job = res.json()
    job_id = job["id"]

    files = {
        "files": (
            "sample.fastq",
            FIXTURE_FASTQ.read_bytes(),
            "text/plain",
        )
    }
    res = requests.post(
        f"{BACKEND_URL}/jobs/{job_id}/files",
        files=files,
        timeout=30,
    )
    assert res.status_code == HTTPStatus.OK, res.text

    res = requests.post(f"{BACKEND_URL}/jobs/{job_id}/run", timeout=10)
    assert res.status_code == HTTPStatus.OK, res.text

    status_deadline = time.time() + 600
    job_url = f"{BACKEND_URL}/jobs/{job_id}"
    while time.time() < status_deadline:
        status_res = requests.get(job_url, timeout=10)
        assert status_res.status_code == HTTPStatus.OK
        data = status_res.json()
        if data["status"] == "COMPLETED":
            break
        if data["status"] == "FAILED":
            logs = _run_compose(["logs", "backend"], check=False).stdout
            pytest.fail(f"Pipeline failed: {data.get('error')}\nBackend logs:\n{logs}")
        time.sleep(15)
    else:
        logs = _run_compose(["logs", "backend"], check=False).stdout
        pytest.fail(f"Pipeline did not finish in time. Backend logs:\n{logs}")

    meta_res = requests.get(f"{BACKEND_URL}/jobs/{job_id}/metadata", timeout=10)
    assert meta_res.status_code == HTTPStatus.OK
    metadata = meta_res.json()["metadata"]
    assert metadata["fastp"] is not None
    assert metadata["dada2"] is not None
