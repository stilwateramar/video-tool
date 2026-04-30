"""Reel generation pipeline.

Given a class video (with detected pose segments) and a configuration
choosing the reel "type" (sequence highlight, single-pose deep dive,
class teaser, etc.), produce one or more output videos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .pose_detector import PoseSegment
from .style_analyzer import StyleProfile
from .video_processor import Clip, make_reel


FORMAT_DIMS = {
    "reel": (1080, 1920),
    "square": (1080, 1080),
    "youtube": (1920, 1080),
}


@dataclass
class ReelRequest:
    name: str
    kind: str  # "sequence" | "pose_focus" | "teaser" | "single"
    target_format: str = "reel"  # "reel" | "square" | "youtube"
    target_seconds: int = 30
    color_preset: str = "natural"
    audio_cleanup: str = "light"
    add_pose_overlay: bool = True
    add_breath_cue: bool = False
    add_progress_bar: bool = False
    speed: float = 1.0
    music_path: str | None = None
    pose_filter: list[str] = field(default_factory=list)  # for pose_focus


def _pick_clips_sequence(
    segments: list[PoseSegment], target: float
) -> list[Clip]:
    """Pick a chronological highlight set whose duration sums to ~target.

    Strategy: rank by confidence, take top N until budget is met, then
    sort chronologically. Cap each held-pose clip at 4s for pacing.
    """
    if not segments:
        return []
    ranked = sorted(segments, key=lambda s: -s.confidence)
    chosen: list[PoseSegment] = []
    budget = target
    for s in ranked:
        d = min(4.0, max(2.0, s.duration))
        if d <= budget + 0.5:
            chosen.append(s)
            budget -= d
        if budget <= 0:
            break
    chosen.sort(key=lambda s: s.start)
    out: list[Clip] = []
    for s in chosen:
        d = min(4.0, max(2.0, s.duration))
        mid = (s.start + s.end) / 2
        out.append(
            Clip(
                start=max(s.start, mid - d / 2),
                end=min(s.end, mid + d / 2),
                label=s.english,
                sanskrit=s.sanskrit,
            )
        )
    return out


def _pick_clips_teaser(
    segments: list[PoseSegment], target: float
) -> list[Clip]:
    """A coming-up tease: grab 5 short snippets evenly spaced through the class."""
    if not segments:
        return []
    n = min(5, len(segments))
    step = max(1, len(segments) // n)
    picked = segments[::step][:n]
    per = target / n
    out = []
    for s in picked:
        d = min(per, max(1.5, s.duration))
        mid = (s.start + s.end) / 2
        out.append(
            Clip(
                start=max(s.start, mid - d / 2),
                end=min(s.end, mid + d / 2),
                label=s.english,
                sanskrit=s.sanskrit,
            )
        )
    return out


def _pick_clips_pose_focus(
    segments: list[PoseSegment], target: float, allowed: list[str]
) -> list[Clip]:
    matches = [s for s in segments if s.pose in allowed]
    if not matches:
        return []
    matches.sort(key=lambda s: -s.confidence)
    out: list[Clip] = []
    budget = target
    for s in matches:
        d = min(6.0, s.duration)
        out.append(Clip(s.start, s.start + d, s.english, s.sanskrit))
        budget -= d
        if budget <= 0:
            break
    out.sort(key=lambda c: c.start)
    return out


def _pick_clips_single(segments: list[PoseSegment], target: float) -> list[Clip]:
    if not segments:
        return []
    best = max(segments, key=lambda s: s.confidence * min(8.0, s.duration))
    d = min(target, max(4.0, best.duration))
    return [Clip(best.start, best.start + d, best.english, best.sanskrit)]


def select_clips(req: ReelRequest, segments: list[PoseSegment]) -> list[Clip]:
    target = float(req.target_seconds)
    if req.kind == "sequence":
        return _pick_clips_sequence(segments, target)
    if req.kind == "teaser":
        return _pick_clips_teaser(segments, target)
    if req.kind == "pose_focus":
        return _pick_clips_pose_focus(segments, target, req.pose_filter or [])
    if req.kind == "single":
        return _pick_clips_single(segments, target)
    return _pick_clips_sequence(segments, target)


def apply_style(req: ReelRequest, profile: StyleProfile) -> ReelRequest:
    """Mutate-in-place defaults using a learned style profile."""
    req.target_format = profile.target_format
    req.color_preset = profile.suggested_preset
    req.speed = profile.suggested_speed
    # Aim a Reel target near the sample's typical clip duration * 6 segments.
    if profile.avg_shot_seconds:
        approx = profile.avg_shot_seconds * 6
        req.target_seconds = int(max(15, min(60, approx)))
    return req


def render(
    req: ReelRequest,
    source: str,
    out_dir: Path,
    segments: list[PoseSegment],
) -> dict:
    clips = select_clips(req, segments)
    if not clips:
        return {"name": req.name, "error": "No matching clips found"}
    target_w, target_h = FORMAT_DIMS.get(req.target_format, FORMAT_DIMS["reel"])
    out_path = out_dir / f"{req.name}.mp4"
    make_reel(
        source=source,
        out_path=str(out_path),
        clips=clips,
        target_w=target_w,
        target_h=target_h,
        color_preset=req.color_preset,
        audio_cleanup=req.audio_cleanup,
        add_pose_overlay=req.add_pose_overlay,
        add_breath_cue=req.add_breath_cue,
        add_progress_bar=req.add_progress_bar,
        music_path=req.music_path,
        speed=req.speed,
    )
    return {
        "name": req.name,
        "kind": req.kind,
        "format": req.target_format,
        "duration": sum(c.duration for c in clips) / max(req.speed, 0.1),
        "color_preset": req.color_preset,
        "speed": req.speed,
        "clips": [
            {
                "start": c.start,
                "end": c.end,
                "label": c.label,
                "sanskrit": c.sanskrit,
            }
            for c in clips
        ],
        "path": str(out_path),
    }
