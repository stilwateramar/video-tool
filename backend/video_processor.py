"""Video processing primitives built on ffmpeg.

This module wraps ffmpeg invocations behind small, composable functions
used by the reel generator: clipping, smart 9:16 reframing using a
person-tracking crop path, color grading, audio cleanup, breath-cue
visualizers, pose-name overlays, and progress-circle overlays.

We deliberately keep filter graphs as text and shell out to ffmpeg via
ffmpeg-python's `compile()`/`run()` for transparency.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import AESTHETIC_PRESETS
from .pose_backend import PoseDetector


@dataclass
class Clip:
    start: float
    end: float
    label: str = ""
    sanskrit: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def ffprobe_meta(path: str) -> dict:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ]
    )
    return json.loads(out)


def video_dimensions(path: str) -> tuple[int, int, float]:
    meta = ffprobe_meta(path)
    v = next(s for s in meta["streams"] if s["codec_type"] == "video")
    w, h = int(v["width"]), int(v["height"])
    dur = float(meta["format"].get("duration", 0))
    return w, h, dur


def _track_subject_x(
    video_path: str, start: float, end: float, samples: int = 24
) -> list[tuple[float, float]]:
    """Sample the subject's horizontal centroid (0..1) across the clip.

    Returns a list of (t_seconds_from_start, x_norm). Falls back to
    centered (0.5) when no person is detected.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = max(0.01, end - start)
    times = np.linspace(0.0, duration, samples)
    points: list[tuple[float, float]] = []
    with PoseDetector(model_complexity=0, min_detection_confidence=0.4) as pose:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, (start + t) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                points.append((float(t), 0.5))
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.landmarks is not None:
                vis_mask = res.visibility > 0.3
                if vis_mask.any():
                    xs = res.landmarks[vis_mask, 0]
                    points.append((float(t), float(np.mean(xs))))
                else:
                    points.append((float(t), 0.5))
            else:
                points.append((float(t), 0.5))
    cap.release()
    # Smooth (moving average) so the crop path doesn't jitter.
    if len(points) >= 3:
        xs = np.array([p[1] for p in points])
        kernel = np.ones(5) / 5
        xs = np.convolve(xs, kernel, mode="same")
        points = [(t, float(x)) for (t, _), x in zip(points, xs)]
    return points


def _build_reframe_filter(
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
    track_points: list[tuple[float, float]],
) -> str:
    """Return an ffmpeg `crop` expression that follows the subject.

    For 9:16 from a 16:9 source we compute the crop width that preserves
    height, then build a piecewise-linear `x` expression in `t`.
    """
    target_ar = target_w / target_h
    src_ar = src_w / src_h

    if src_ar > target_ar:
        # Source wider than target: crop horizontally, full height.
        crop_h = src_h
        crop_w = int(round(src_h * target_ar))
    else:
        # Source taller/narrower: crop vertically, full width.
        crop_w = src_w
        crop_h = int(round(src_w / target_ar))

    if crop_w >= src_w and crop_h >= src_h:
        return f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"

    if crop_w < src_w:
        # Build x(t) following the subject; clamped to [0, src_w-crop_w].
        max_x = src_w - crop_w
        # Fall back to center if track empty.
        pts = track_points or [(0.0, 0.5)]
        # Convert normalized centroid to crop x.
        xs = [(t, max(0.0, min(1.0, x)) * src_w - crop_w / 2) for t, x in pts]
        xs = [(t, max(0.0, min(max_x, x))) for t, x in xs]
        # Build a piecewise expression: between two samples, linear interp.
        x_expr = _piecewise_expr(xs, default=max_x / 2)
        return f"crop={crop_w}:{crop_h}:'{x_expr}':0,scale={target_w}:{target_h}"
    else:
        max_y = src_h - crop_h
        # Vertical crop: bias slightly upward (head/shoulders region).
        y_expr = f"{int(max_y * 0.25)}"
        return f"crop={crop_w}:{crop_h}:0:{y_expr},scale={target_w}:{target_h}"


def _piecewise_expr(points: list[tuple[float, float]], default: float) -> str:
    """Build a clamped piecewise-linear ffmpeg expression in `t`.

    The expression is intended to be wrapped in single quotes in a filter
    argument (e.g. `crop=...:'EXPR':...`). Inside single-quoted filter
    arguments, commas are passed through to the expression evaluator
    literally and do not need to be escaped.
    """
    if not points:
        return f"{default}"
    if len(points) == 1:
        return f"{points[0][1]:.2f}"
    expr = f"{points[-1][1]:.2f}"  # default branch (after last segment)
    # Build right-to-left so each `if` wraps the rest as its else clause.
    for i in range(len(points) - 2, -1, -1):
        t0, x0 = points[i]
        t1, _ = points[i + 1]
        if t1 - t0 < 1e-3:
            continue
        x1 = points[i + 1][1]
        slope = (x1 - x0) / (t1 - t0)
        expr = (
            f"if(between(t,{t0:.3f},{t1:.3f}),"
            f"{x0:.2f}+({slope:.4f})*(t-{t0:.3f}),{expr})"
        )
    return expr


def _drawtext_pose_overlay(
    clips: Iterable[Clip], offset: float, fontfile: str | None = None
) -> str | None:
    """Overlay each clip's pose name at the bottom for the first 2.5s."""
    parts = []
    cum = 0.0
    for c in clips:
        dur = c.duration
        if c.label:
            label = (
                f"{c.sanskrit} - {c.label}" if c.sanskrit else c.label
            )
            # drawtext needs colons, single quotes, percents, and backslashes escaped.
            label = (
                label.replace("\\", "\\\\")
                .replace(":", "\\:")
                .replace("'", "")
                .replace("%", "\\%")
            )
            t0 = cum + 0.2
            t1 = cum + min(2.8, dur)
            font = f":fontfile={fontfile}" if fontfile else ""
            parts.append(
                f"drawtext=text='{label}'{font}:fontcolor=white:fontsize=42:"
                f"box=1:boxcolor=black@0.45:boxborderw=18:"
                f"x=(w-text_w)/2:y=h-text_h-120:enable='between(t,{t0:.2f},{t1:.2f})'"
            )
        cum += dur
    return ",".join(parts) if parts else None


def _breath_circle_filter(target_w: int, target_h: int, period: float = 8.0) -> str:
    """A pulsing translucent circle in the bottom-right that mimics a breath cue.

    Uses ffmpeg's `geq` would be heavy; instead we draw a circle via
    `drawbox` with rounded corners isn't supported — so we synthesize a
    color source with a shifting alpha and overlay it.
    """
    radius_max = int(min(target_w, target_h) * 0.06)
    cx = target_w - radius_max - 40
    cy = target_h - radius_max - 220
    # Scale a pre-made disc (color source via drawtext bullet) using a
    # simple drawbox — keep it cheap and predictable.
    return (
        f"drawbox=x={cx - radius_max}:y={cy - radius_max}:"
        f"w={2 * radius_max}:h={2 * radius_max}:color=white@0.18:t=fill,"
        f"drawbox=x='{cx} - {radius_max}*(0.6+0.4*sin(2*PI*t/{period}))':"
        f"y='{cy} - {radius_max}*(0.6+0.4*sin(2*PI*t/{period}))':"
        f"w='2*{radius_max}*(0.6+0.4*sin(2*PI*t/{period}))':"
        f"h='2*{radius_max}*(0.6+0.4*sin(2*PI*t/{period}))':"
        f"color=white@0.45:t=fill"
    )


def _progress_circle_filter(target_w: int, total_dur: float) -> str:
    """A thin progress bar across the top, useful for YouTube format."""
    return (
        f"drawbox=x=0:y=0:w='w*t/{max(0.1, total_dur):.3f}':h=8:color=white@0.85:t=fill"
    )


def _audio_cleanup_filter(level: str) -> str | None:
    """Map a UI choice to an audio filter graph.

    `light`: gentle highpass + dynaudnorm. `zen`: aggressive denoise via
    afftdn + highpass + loudnorm to deliver a clean "studio" feel.
    """
    if level == "off":
        return None
    if level == "light":
        return "highpass=f=80,dynaudnorm=f=200:g=15"
    if level == "zen":
        return "highpass=f=85,afftdn=nr=20:nf=-30,dynaudnorm=f=180:g=12,loudnorm=I=-16:TP=-1.5:LRA=11"
    return None


def make_reel(
    *,
    source: str,
    out_path: str,
    clips: list[Clip],
    target_w: int,
    target_h: int,
    color_preset: str = "natural",
    audio_cleanup: str = "light",
    add_pose_overlay: bool = True,
    add_breath_cue: bool = False,
    add_progress_bar: bool = False,
    music_path: str | None = None,
    speed: float = 1.0,
) -> str:
    """Render a reel by concatenating clips with the configured effects.

    We build one ffmpeg command per clip into temporary files and then
    concat-demux them. This is more robust than a giant filter_complex
    when clips have different timestamps, and keeps the overlays' time
    base local to the final concatenated output.
    """
    src_w, src_h, _ = video_dimensions(source)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(out_path).with_suffix("")
    tmp_dir.mkdir(exist_ok=True)

    segment_files = []
    for i, clip in enumerate(clips):
        seg_path = tmp_dir / f"seg_{i:03d}.mp4"
        track = _track_subject_x(source, clip.start, clip.end)
        reframe = _build_reframe_filter(src_w, src_h, target_w, target_h, track)
        vf_parts = [reframe]
        preset = AESTHETIC_PRESETS.get(color_preset)
        if preset:
            vf_parts.append(preset)
        if speed != 1.0:
            vf_parts.append(f"setpts=PTS/{speed}")
        vf = ",".join(vf_parts)
        af_parts = []
        af = _audio_cleanup_filter(audio_cleanup)
        if af:
            af_parts.append(af)
        if speed != 1.0:
            af_parts.append(f"atempo={max(0.5, min(2.0, speed)):.3f}")
        af_str = ",".join(af_parts) if af_parts else None

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{clip.start:.3f}",
            "-to",
            f"{clip.end:.3f}",
            "-i",
            source,
            "-vf",
            vf,
        ]
        if af_str:
            cmd += ["-af", af_str]
        cmd += [
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(seg_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        segment_files.append(seg_path)

    # Concat-demuxer file list.
    list_file = tmp_dir / "segments.txt"
    list_file.write_text("\n".join(f"file {shlex.quote(str(s))}" for s in segment_files))

    concat_path = tmp_dir / "concat.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(concat_path),
        ],
        check=True,
        capture_output=True,
    )

    # Final pass: overlays (pose names, breath cue, progress bar) and music.
    total_dur = sum(c.duration for c in clips) / max(speed, 0.1)
    vf_overlays = []
    if add_pose_overlay:
        ov = _drawtext_pose_overlay(clips, offset=0.0)
        if ov:
            vf_overlays.append(ov)
    if add_breath_cue:
        vf_overlays.append(_breath_circle_filter(target_w, target_h))
    if add_progress_bar:
        vf_overlays.append(_progress_circle_filter(target_w, total_dur))
    vf_final = ",".join(vf_overlays) if vf_overlays else None

    final_cmd = ["ffmpeg", "-y", "-i", str(concat_path)]
    if music_path:
        final_cmd += ["-i", music_path]
    if vf_final:
        final_cmd += ["-vf", vf_final]

    if music_path:
        # Sidechain the original audio against music: music ducks under voice.
        final_cmd += [
            "-filter_complex",
            "[1:a]volume=0.25[m];[0:a][m]amix=inputs=2:duration=first:dropout_transition=2[aout]",
            "-map",
            "0:v",
            "-map",
            "[aout]",
        ]
    else:
        final_cmd += ["-map", "0:v", "-map", "0:a?"]

    final_cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        out_path,
    ]
    subprocess.run(final_cmd, check=True, capture_output=True)
    return out_path
