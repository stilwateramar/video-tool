"""Pose detection and yoga-pose classification.

We sample frames at a fixed interval, run MediaPipe Pose to get 33 body
landmarks, then classify each frame using simple geometric heuristics
(joint angles + relative positions). Adjacent frames classified as the
same pose are merged into "pose segments" with start/end timestamps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# MediaPipe Pose landmark indices (subset used).
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

POSE_NAMES = {
    "downward_dog": ("Adho Mukha Svanasana", "Downward-Facing Dog"),
    "upward_dog": ("Urdhva Mukha Svanasana", "Upward-Facing Dog"),
    "plank": ("Phalakasana", "Plank"),
    "chaturanga": ("Chaturanga Dandasana", "Four-Limbed Staff"),
    "cobra": ("Bhujangasana", "Cobra"),
    "child": ("Balasana", "Child's Pose"),
    "warrior_ii": ("Virabhadrasana II", "Warrior II"),
    "tree": ("Vrksasana", "Tree"),
    "mountain": ("Tadasana", "Mountain"),
    "forward_fold": ("Uttanasana", "Forward Fold"),
    "seated": ("Sukhasana", "Easy Seat"),
    "transition": ("", "Transition"),
}


@dataclass
class PoseSegment:
    pose: str
    sanskrit: str
    english: str
    start: float
    end: float
    confidence: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        return d


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees."""
    ba = a - b
    bc = c - b
    cos = float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9))
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def _classify(lm: np.ndarray, vis: np.ndarray) -> tuple[str, float]:
    """Classify a single frame's landmarks into a yoga pose label.

    `lm` is shape (33, 3) with normalized image coords (x right, y down).
    Returns (pose_key, confidence in [0, 1]).
    """
    if vis[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]].mean() < 0.4:
        return "transition", 0.0

    sh_mid = (lm[L_SHOULDER] + lm[R_SHOULDER]) / 2
    hip_mid = (lm[L_HIP] + lm[R_HIP]) / 2
    ank_mid = (lm[L_ANKLE] + lm[R_ANKLE]) / 2
    wri_mid = (lm[L_WRIST] + lm[R_WRIST]) / 2

    # Torso vector (hip -> shoulder); positive y is down in image coords.
    torso = sh_mid - hip_mid
    torso_tilt = math.degrees(math.atan2(torso[0], -torso[1]))  # 0 = upright
    torso_len = float(np.linalg.norm(torso[:2])) + 1e-6

    knee_l = _angle(lm[L_HIP, :2], lm[L_KNEE, :2], lm[L_ANKLE, :2])
    knee_r = _angle(lm[R_HIP, :2], lm[R_KNEE, :2], lm[R_ANKLE, :2])
    elbow_l = _angle(lm[L_SHOULDER, :2], lm[L_ELBOW, :2], lm[L_WRIST, :2])
    elbow_r = _angle(lm[R_SHOULDER, :2], lm[R_ELBOW, :2], lm[R_WRIST, :2])
    hip_l = _angle(lm[L_SHOULDER, :2], lm[L_HIP, :2], lm[L_KNEE, :2])
    hip_r = _angle(lm[R_SHOULDER, :2], lm[R_HIP, :2], lm[R_KNEE, :2])

    wrists_below_shoulders = wri_mid[1] > sh_mid[1] + 0.05
    wrists_above_head = wri_mid[1] < lm[NOSE, 1] - 0.05
    hands_on_floor = wri_mid[1] > hip_mid[1]
    feet_on_floor = ank_mid[1] > hip_mid[1] + 0.05
    knees_bent = (knee_l < 140) or (knee_r < 140)
    knees_straight = knee_l > 160 and knee_r > 160
    arms_straight = elbow_l > 155 and elbow_r > 155
    arms_bent = elbow_l < 110 or elbow_r < 110

    # Down Dog: inverted V; hips highest, hands & feet on floor, body roughly diagonal.
    if (
        hands_on_floor
        and feet_on_floor
        and hip_mid[1] < sh_mid[1]
        and hip_mid[1] < ank_mid[1]
        and arms_straight
        and knees_straight
    ):
        return "downward_dog", 0.9

    # Plank: body horizontal, arms straight, shoulders over wrists.
    if (
        arms_straight
        and abs(sh_mid[1] - hip_mid[1]) < 0.15
        and abs(hip_mid[1] - ank_mid[1]) < 0.18
        and wrists_below_shoulders
    ):
        return "plank", 0.85

    # Chaturanga: like plank but elbows bent ~90.
    if (
        arms_bent
        and abs(sh_mid[1] - hip_mid[1]) < 0.18
        and wrists_below_shoulders
        and feet_on_floor
    ):
        return "chaturanga", 0.75

    # Upward Dog / Cobra: chest up, hips low, arms straightening.
    if (
        sh_mid[1] < hip_mid[1] - 0.05
        and hip_l > 150
        and hip_r > 150
        and torso_tilt < -25  # leaning back
    ):
        if arms_straight and hip_mid[1] < ank_mid[1]:
            return "upward_dog", 0.8
        return "cobra", 0.7

    # Child's pose: hips back over heels, torso folded, arms forward/down.
    if (
        knees_bent
        and hip_mid[1] > sh_mid[1]
        and abs(hip_mid[0] - ank_mid[0]) < 0.2
        and hip_l < 80
        and hip_r < 80
    ):
        return "child", 0.75

    # Forward fold (standing): legs straight, torso folded, hands near floor.
    if (
        knees_straight
        and feet_on_floor
        and sh_mid[1] > hip_mid[1]
        and hands_on_floor
    ):
        return "forward_fold", 0.8

    # Warrior II: front knee bent ~90, back leg straight, arms extended horizontally.
    arms_horizontal = (
        abs(lm[L_WRIST, 1] - lm[L_SHOULDER, 1]) < 0.08
        and abs(lm[R_WRIST, 1] - lm[R_SHOULDER, 1]) < 0.08
        and abs(lm[L_WRIST, 0] - lm[R_WRIST, 0]) > 0.4
    )
    if arms_horizontal and ((knee_l < 120 and knee_r > 150) or (knee_r < 120 and knee_l > 150)):
        return "warrior_ii", 0.8

    # Tree: standing on one leg with the other foot tucked against the standing leg.
    one_foot_lifted = abs(lm[L_ANKLE, 1] - lm[R_ANKLE, 1]) > 0.18
    if one_foot_lifted and abs(torso_tilt) < 25 and wrists_above_head is False:
        return "tree", 0.6

    # Mountain: standing tall, arms by sides or overhead, torso upright.
    if (
        knees_straight
        and abs(torso_tilt) < 15
        and ank_mid[1] > hip_mid[1] + 0.2
        and torso_len > 0.18
    ):
        if wrists_above_head:
            return "mountain", 0.75
        return "mountain", 0.65

    # Seated: hips at lowest while torso upright and ankles tucked in.
    if (
        abs(torso_tilt) < 25
        and hip_mid[1] > sh_mid[1] + 0.15
        and abs(ank_mid[0] - hip_mid[0]) < 0.2
        and ank_mid[1] - hip_mid[1] < 0.1
    ):
        return "seated", 0.55

    return "transition", 0.0


def detect_poses(
    video_path: str,
    sample_interval_sec: float = 0.5,
    min_hold_sec: float = 2.0,
    progress: callable | None = None,
) -> tuple[list[PoseSegment], dict]:
    """Run pose detection on a video and return merged pose segments.

    Returns (segments, metadata) where metadata includes fps, duration, and
    aspect ratio.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps else 0.0

    frame_step = max(1, int(round(fps * sample_interval_sec)))

    raw: list[tuple[float, str, float]] = []  # (t, pose_key, confidence)

    with mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % frame_step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                t = idx / fps
                if res.pose_landmarks:
                    lm = np.array(
                        [[p.x, p.y, p.z] for p in res.pose_landmarks.landmark],
                        dtype=np.float32,
                    )
                    vis = np.array(
                        [p.visibility for p in res.pose_landmarks.landmark],
                        dtype=np.float32,
                    )
                    key, conf = _classify(lm, vis)
                else:
                    key, conf = "transition", 0.0
                raw.append((t, key, conf))
                if progress and total_frames:
                    progress(idx / total_frames)
            idx += 1
    cap.release()

    segments = _merge(raw, sample_interval_sec, min_hold_sec)
    meta = {
        "fps": fps,
        "duration": duration,
        "width": width,
        "height": height,
        "aspect_ratio": (width / height) if height else 0.0,
    }
    return segments, meta


def _merge(
    raw: Iterable[tuple[float, str, float]],
    sample_interval_sec: float,
    min_hold_sec: float,
) -> list[PoseSegment]:
    raw = list(raw)
    if not raw:
        return []

    segments: list[PoseSegment] = []
    cur_key = raw[0][1]
    cur_start = raw[0][0]
    cur_confs = [raw[0][2]]

    def flush(end_t: float):
        if cur_key == "transition":
            return
        if end_t - cur_start < min_hold_sec:
            return
        san, eng = POSE_NAMES.get(cur_key, ("", cur_key))
        segments.append(
            PoseSegment(
                pose=cur_key,
                sanskrit=san,
                english=eng,
                start=cur_start,
                end=end_t,
                confidence=float(np.mean(cur_confs)) if cur_confs else 0.0,
            )
        )

    last_t = raw[0][0]
    for t, key, conf in raw[1:]:
        if key == cur_key:
            cur_confs.append(conf)
        else:
            flush(t)
            cur_key = key
            cur_start = t
            cur_confs = [conf]
        last_t = t
    flush(last_t + sample_interval_sec)
    return segments
