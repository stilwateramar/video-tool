"""Mediapipe pose-detection wrapper.

Recent mediapipe wheels (notably the Python 3.13 wheel) ship only the
new Tasks API and have dropped the legacy `mediapipe.solutions` module.
We therefore prefer the Tasks API and fall back to the legacy module
when it's available.

Both backends expose a common interface:

    with PoseDetector(model_complexity=1) as pose:
        result = pose.process(rgb_frame_uint8)
    if result.landmarks is not None:
        # result.landmarks: ndarray (33, 3) of (x, y, z) in [0, 1]
        # result.visibility: ndarray (33,) in [0, 1]

The Tasks backend downloads `pose_landmarker_lite.task` on first use
into `storage/models/`.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import STORAGE


_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
_MODEL_PATH = STORAGE / "models" / "pose_landmarker_lite.task"


@dataclass
class PoseResult:
    landmarks: np.ndarray | None  # (33, 3) or None
    visibility: np.ndarray | None  # (33,) or None


def _ensure_model() -> Path:
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pose] downloading {_MODEL_URL} -> {_MODEL_PATH}")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


class _LegacyBackend:
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        import mediapipe as mp  # noqa: WPS433
        self._pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
        )

    def process(self, rgb: np.ndarray) -> PoseResult:
        res = self._pose.process(rgb)
        if not res.pose_landmarks:
            return PoseResult(None, None)
        lms = res.pose_landmarks.landmark
        coords = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
        vis = np.array([p.visibility for p in lms], dtype=np.float32)
        return PoseResult(coords, vis)

    def close(self):
        self._pose.close()


class _TasksBackend:
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        import mediapipe as mp  # noqa: WPS433
        from mediapipe.tasks import python as mp_python  # noqa: WPS433
        from mediapipe.tasks.python import vision  # noqa: WPS433

        self._mp = mp
        model_path = str(_ensure_model())
        base = mp_python.BaseOptions(model_asset_path=model_path)
        opts = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opts)

    def process(self, rgb: np.ndarray) -> PoseResult:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect(mp_image)
        if not res.pose_landmarks:
            return PoseResult(None, None)
        lms = res.pose_landmarks[0]
        coords = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
        # The Tasks API renamed `visibility` to `presence`/`visibility` — both are
        # exposed on NormalizedLandmark; fall back to 1.0 if absent.
        vis = np.array(
            [getattr(p, "visibility", 1.0) or 1.0 for p in lms], dtype=np.float32
        )
        return PoseResult(coords, vis)

    def close(self):
        self._landmarker.close()


def _have_legacy() -> bool:
    try:
        import mediapipe as mp  # noqa: WPS433
        return hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
    except Exception:
        return False


class PoseDetector:
    """Picks the best available backend at construction time."""

    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5):
        if _have_legacy():
            self._impl = _LegacyBackend(model_complexity, min_detection_confidence)
            self.backend = "legacy"
        else:
            self._impl = _TasksBackend(model_complexity, min_detection_confidence)
            self.backend = "tasks"

    def process(self, rgb: np.ndarray) -> PoseResult:
        return self._impl.process(rgb)

    def close(self):
        self._impl.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
