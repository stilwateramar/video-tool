"""Tiny on-disk job store. Each job is a JSON file under storage/jobs/."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from .config import JOBS

_lock = threading.Lock()


def new_job(kind: str, payload: dict | None = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    data = {
        "id": job_id,
        "kind": kind,
        "status": "queued",
        "progress": 0.0,
        "message": "",
        "result": None,
        "error": None,
        "payload": payload or {},
    }
    with _lock:
        _write(job_id, data)
    return job_id


def update(job_id: str, **fields: Any) -> None:
    with _lock:
        data = _read(job_id)
        data.update(fields)
        _write(job_id, data)


def set_status(job_id: str, status: str, message: str = "") -> None:
    update(job_id, status=status, message=message)


def set_progress(job_id: str, progress: float, message: str | None = None) -> None:
    fields: dict = {"progress": float(max(0.0, min(1.0, progress)))}
    if message is not None:
        fields["message"] = message
    update(job_id, **fields)


def set_result(job_id: str, result: dict) -> None:
    update(job_id, status="done", progress=1.0, result=result, message="completed")


def set_error(job_id: str, err: str) -> None:
    update(job_id, status="error", error=err, message=err)


def get(job_id: str) -> dict | None:
    p = JOBS / f"{job_id}.json"
    if not p.exists():
        return None
    with _lock:
        return _read(job_id)


def _read(job_id: str) -> dict:
    return json.loads((JOBS / f"{job_id}.json").read_text())


def _write(job_id: str, data: dict) -> None:
    """Atomically write the job file so concurrent readers never see a
    truncated/empty file.
    """
    target = JOBS / f"{job_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{job_id}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
