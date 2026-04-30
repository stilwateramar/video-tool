"""FastAPI server for the yoga reel tool."""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import jobs
from .config import (
    AESTHETIC_PRESETS,
    FRONTEND,
    MUSIC_TRACKS,
    OUTPUTS,
    POSE_SAMPLE_INTERVAL_SEC,
    MIN_POSE_HOLD_SEC,
    SAMPLES,
    UPLOADS,
)
from .pipeline import ReelRequest, apply_style, render
from .pose_detector import detect_poses
from .style_analyzer import analyze, save_profile, load_profile

app = FastAPI(title="Yoga Reel Studio")

# Persisted analysis caches keyed by upload id.
ANALYSIS = UPLOADS / "_analysis"
ANALYSIS.mkdir(exist_ok=True)
PROFILES = SAMPLES / "_profiles"
PROFILES.mkdir(exist_ok=True)


# ---------- helpers ----------

def _save_upload(file: UploadFile, dest_dir: Path) -> tuple[str, Path]:
    upload_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename or "").suffix or ".mp4"
    target = dest_dir / f"{upload_id}{suffix}"
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return upload_id, target


def _analysis_path(upload_id: str) -> Path:
    return ANALYSIS / f"{upload_id}.json"


# ---------- class video upload + pose detection ----------

@app.post("/api/upload")
async def upload_class_video(
    background: BackgroundTasks, file: UploadFile = File(...)
):
    upload_id, target = _save_upload(file, UPLOADS)
    job_id = jobs.new_job(
        "analyze",
        payload={"upload_id": upload_id, "filename": file.filename, "path": str(target)},
    )

    def run():
        try:
            jobs.set_status(job_id, "running", "Detecting poses")

            def progress(p: float):
                jobs.set_progress(job_id, p * 0.95)

            segments, meta = detect_poses(
                str(target),
                sample_interval_sec=POSE_SAMPLE_INTERVAL_SEC,
                min_hold_sec=MIN_POSE_HOLD_SEC,
                progress=progress,
            )
            payload = {
                "upload_id": upload_id,
                "filename": file.filename,
                "path": str(target),
                "meta": meta,
                "segments": [s.to_dict() for s in segments],
            }
            _analysis_path(upload_id).write_text(json.dumps(payload, indent=2))
            jobs.set_result(job_id, payload)
        except Exception as e:  # pragma: no cover
            jobs.set_error(job_id, f"{type(e).__name__}: {e}")

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id, "upload_id": upload_id}


@app.get("/api/analysis/{upload_id}")
def get_analysis(upload_id: str):
    p = _analysis_path(upload_id)
    if not p.exists():
        raise HTTPException(404, "analysis not ready")
    return JSONResponse(json.loads(p.read_text()))


# ---------- inspiration sample upload ----------

@app.post("/api/sample")
async def upload_sample(file: UploadFile = File(...)):
    sample_id, target = _save_upload(file, SAMPLES)
    profile = analyze(str(target))
    save_profile(profile, PROFILES / f"{sample_id}.json")
    return {"sample_id": sample_id, "profile": profile.to_dict()}


@app.get("/api/sample/{sample_id}")
def get_sample(sample_id: str):
    p = PROFILES / f"{sample_id}.json"
    if not p.exists():
        raise HTTPException(404, "sample not found")
    return JSONResponse(json.loads(p.read_text()))


# ---------- reel generation ----------

class GenerateReelBody(BaseModel):
    upload_id: str
    sample_id: str | None = None
    reels: list[dict]  # list of partial ReelRequest dicts


@app.post("/api/generate")
def generate(body: GenerateReelBody):
    apath = _analysis_path(body.upload_id)
    if not apath.exists():
        raise HTTPException(404, "analysis not found; upload first")
    analysis = json.loads(apath.read_text())
    source = analysis["path"]

    profile = None
    if body.sample_id:
        pp = PROFILES / f"{body.sample_id}.json"
        if pp.exists():
            profile = load_profile(pp)

    job_id = jobs.new_job(
        "generate",
        payload={
            "upload_id": body.upload_id,
            "sample_id": body.sample_id,
            "reels": body.reels,
        },
    )

    def run():
        try:
            jobs.set_status(job_id, "running", "Rendering reels")
            from .pose_detector import PoseSegment

            segments = [PoseSegment(**{k: v for k, v in s.items() if k != "duration"}) for s in analysis["segments"]]
            out_dir = OUTPUTS / body.upload_id
            out_dir.mkdir(parents=True, exist_ok=True)
            results = []
            for i, raw in enumerate(body.reels):
                # Resolve music alias to file path if known.
                music_choice = raw.pop("music", None)
                music_path = None
                if music_choice and music_choice != "none":
                    fname = MUSIC_TRACKS.get(music_choice)
                    if fname:
                        candidate = FRONTEND.parent / "assets" / "music" / fname
                        if candidate.exists():
                            music_path = str(candidate)

                req = ReelRequest(name=raw.get("name") or f"reel_{i+1}", kind=raw.get("kind", "sequence"))
                # Apply style first, then user overrides.
                if profile is not None and raw.get("apply_style", True):
                    apply_style(req, profile)
                for k, v in raw.items():
                    if hasattr(req, k) and v is not None:
                        setattr(req, k, v)
                req.music_path = music_path
                jobs.set_progress(job_id, i / max(1, len(body.reels)), f"Rendering {req.name}")
                result = render(req, source, out_dir, segments)
                # Make path web-relative.
                if "path" in result:
                    result["url"] = f"/outputs/{body.upload_id}/{Path(result['path']).name}"
                results.append(result)
            jobs.set_result(job_id, {"reels": results})
        except Exception as e:  # pragma: no cover
            jobs.set_error(job_id, f"{type(e).__name__}: {e}")

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id}


# ---------- job polling ----------

@app.get("/api/job/{job_id}")
def get_job(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return j


# ---------- options metadata ----------

@app.get("/api/options")
def options():
    return {
        "presets": list(AESTHETIC_PRESETS.keys()),
        "music": list(MUSIC_TRACKS.keys()),
        "formats": ["reel", "square", "youtube"],
        "audio_cleanup": ["off", "light", "zen"],
        "kinds": [
            {"id": "sequence", "label": "Highlight sequence (best poses)"},
            {"id": "teaser", "label": "Coming-up teaser"},
            {"id": "pose_focus", "label": "Single-pose deep dive"},
            {"id": "single", "label": "One long held pose"},
        ],
    }


# ---------- static serving ----------

app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND / "index.html"))


app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
