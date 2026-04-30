from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORAGE = ROOT / "storage"
UPLOADS = STORAGE / "uploads"
SAMPLES = STORAGE / "samples"
OUTPUTS = STORAGE / "outputs"
JOBS = STORAGE / "jobs"
FRONTEND = ROOT / "frontend"

for p in (UPLOADS, SAMPLES, OUTPUTS, JOBS):
    p.mkdir(parents=True, exist_ok=True)

# Pose detection: sample one frame every N seconds during analysis.
POSE_SAMPLE_INTERVAL_SEC = 0.5

# Minimum dwell time (seconds) for a pose to be reported as a "held" pose.
MIN_POSE_HOLD_SEC = 2.0

# Default reel target durations.
REEL_DURATIONS = {
    "short": 15,
    "medium": 30,
    "long": 60,
}

# Color-grade presets implemented as ffmpeg filter graphs (color-only).
AESTHETIC_PRESETS = {
    "natural": None,
    "golden_hour": "eq=contrast=1.05:saturation=1.15:gamma_r=1.08:gamma_g=1.0:gamma_b=0.92,colorbalance=rs=0.08:gs=0.02:bs=-0.08",
    "earthy_organic": "eq=contrast=1.02:saturation=0.85:gamma_g=1.05,colorbalance=rs=0.04:gs=0.06:bs=-0.04",
    "minimalist_studio": "eq=contrast=1.10:saturation=0.65:brightness=0.03",
    "moody_yin": "eq=contrast=1.12:saturation=0.80:brightness=-0.04,colorbalance=rs=-0.04:bs=0.10",
    "power_vibrant": "eq=contrast=1.15:saturation=1.30",
}

# A small library of bundled (silent) ambience options. Real deployments
# would back this with licensed tracks.
MUSIC_TRACKS = {
    "none": None,
    "lofi_calm": "lofi_calm.mp3",
    "ambient_drone": "ambient_drone.mp3",
    "uplifting_flow": "uplifting_flow.mp3",
}
