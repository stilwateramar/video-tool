"""Analyze a sample/inspiration video to extract a "style profile".

We sample frames and audio properties to derive:
- aspect ratio (-> target Reel/YouTube format)
- average shot length (-> pacing for clip duration)
- speed factor (motion magnitude vs typical -> playback speed hint)
- color palette / temperature (-> nearest aesthetic preset)
- music presence (-> whether to suggest music overlay)

The profile is JSON-serializable and consumed by `pipeline.apply_style`.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from .config import AESTHETIC_PRESETS


@dataclass
class StyleProfile:
    aspect_ratio: float
    target_format: str  # "reel" | "square" | "youtube"
    avg_shot_seconds: float
    suggested_speed: float
    avg_brightness: float
    avg_saturation: float
    color_temperature: float  # rough b/r ratio
    suggested_preset: str
    has_music: bool
    duration: float

    def to_dict(self) -> dict:
        return asdict(self)


def _detect_shot_changes(path: str, max_samples: int = 600) -> tuple[float, float]:
    """Estimate average shot length and motion magnitude.

    Compute frame-to-frame histogram distance; large jumps mark cuts.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return 4.0, 0.0
    step = max(1, total // max_samples)
    prev_hist = None
    cuts = 0
    motion = []
    sampled = 0
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, f = cap.retrieve()
            if not ok:
                break
            small = cv2.resize(f, (160, 90))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            if prev_hist is not None:
                d = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                motion.append(d)
                if d > 0.5:
                    cuts += 1
            prev_hist = hist
            sampled += 1
        idx += 1
    cap.release()
    duration = total / fps if fps else 0.0
    avg_shot = duration / max(1, cuts) if cuts else duration
    avg_motion = float(np.mean(motion)) if motion else 0.0
    return float(avg_shot), avg_motion


def _color_summary(path: str, max_samples: int = 60) -> tuple[float, float, float]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return 0.5, 0.5, 1.0
    step = max(1, total // max_samples)
    brightness = []
    saturation = []
    rb_ratio = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, f = cap.retrieve()
            if not ok:
                break
            hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
            brightness.append(float(np.mean(hsv[..., 2])) / 255.0)
            saturation.append(float(np.mean(hsv[..., 1])) / 255.0)
            b, g, r = cv2.split(f)
            rb_ratio.append(float(np.mean(r)) / max(1.0, float(np.mean(b))))
        idx += 1
    cap.release()
    return (
        float(np.mean(brightness)),
        float(np.mean(saturation)),
        float(np.mean(rb_ratio)),
    )


def _has_audible_music(path: str) -> bool:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "json",
                path,
            ]
        )
        data = json.loads(out)
        return bool(data.get("streams"))
    except Exception:
        return False


def _pick_preset(brightness: float, saturation: float, rb_ratio: float) -> str:
    if rb_ratio > 1.18 and brightness > 0.45:
        return "golden_hour"
    if saturation < 0.35:
        return "minimalist_studio"
    if rb_ratio < 0.95:
        return "moody_yin"
    if saturation > 0.55:
        return "power_vibrant"
    if 0.95 <= rb_ratio <= 1.08 and 0.35 <= saturation <= 0.55:
        return "earthy_organic"
    return "natural"


def _pick_format(aspect_ratio: float) -> str:
    if aspect_ratio < 0.85:
        return "reel"
    if aspect_ratio < 1.15:
        return "square"
    return "youtube"


def analyze(path: str) -> StyleProfile:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration = total / fps if fps else 0.0
    aspect = w / h if h else 1.0

    avg_shot, motion = _detect_shot_changes(path)
    brightness, saturation, rb = _color_summary(path)
    preset = _pick_preset(brightness, saturation, rb)
    fmt = _pick_format(aspect)

    # Motion: in our metric ~0.15 is calm, ~0.45 is fast-cut. Map to speed.
    if motion < 0.18:
        speed = 1.0
    elif motion < 0.30:
        speed = 1.25
    elif motion < 0.45:
        speed = 1.6
    else:
        speed = 2.0

    return StyleProfile(
        aspect_ratio=float(aspect),
        target_format=fmt,
        avg_shot_seconds=float(avg_shot),
        suggested_speed=float(speed),
        avg_brightness=float(brightness),
        avg_saturation=float(saturation),
        color_temperature=float(rb),
        suggested_preset=preset if preset in AESTHETIC_PRESETS else "natural",
        has_music=_has_audible_music(path),
        duration=float(duration),
    )


def save_profile(profile: StyleProfile, path: Path) -> None:
    path.write_text(json.dumps(profile.to_dict(), indent=2))


def load_profile(path: Path) -> StyleProfile:
    return StyleProfile(**json.loads(path.read_text()))
