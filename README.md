# Yoga Reel Studio

A web tool that turns a long yoga class recording into multiple
social-ready reels. Yoga teachers can:

1. Upload a class video (any aspect ratio).
2. Optionally upload an "inspiration" reel they want their content to
   look like &mdash; the tool extracts color, pacing, aspect ratio,
   speed, and chooses a matching aesthetic preset.
3. Configure one or more reels (highlight sequence, single-pose deep
   dive, coming-up teaser, etc.) and render them with smart subject
   tracking, color grading, audio cleanup, pose-name overlays, breath
   cues, and ducking music.

## What's inside

```
backend/
  main.py            FastAPI app: uploads, jobs, generation, static UI
  pose_detector.py   MediaPipe Pose + heuristic yoga-pose classifier
  video_processor.py ffmpeg pipeline: clipping, smart 9:16 reframing,
                     color presets, drawtext overlays, audio cleanup,
                     breath-cue circle, progress bar, music ducking
  style_analyzer.py  Inspiration video -> style profile
                     (aspect, pacing, color temperature, preset, speed)
  pipeline.py        Reel selection strategies (sequence, teaser,
                     pose_focus, single) and render orchestration
  jobs.py            File-backed background job tracker
  config.py          Storage paths, presets, music aliases
frontend/
  index.html / app.js / style.css  Single-page UI
storage/              Created at runtime (uploads, samples, outputs, jobs)
```

## How the features map to the brief

| Feature in the brief | Where it lives |
| --- | --- |
| Pose Finder (list every pose with timestamps) | `pose_detector.detect_poses` produces `PoseSegment`s; rendered in the "Held poses found" table |
| Auto-Reel Generator (e.g. Sun Salutation B) | `pipeline.select_clips` with `kind=pose_focus` and a list of pose ids |
| Pose-name pop-ups in Sanskrit + English | `video_processor._drawtext_pose_overlay` (uses `POSE_NAMES`) |
| Breath visualizer | `video_processor._breath_circle_filter` (toggle in UI) |
| Studio-quality audio cleanup ("Zen Filter") | `video_processor._audio_cleanup_filter` with `light` / `zen` modes |
| Dynamic music sync (ducking) | Final ffmpeg pass uses `amix` with a reduced-volume music track |
| Smart 9:16 reframing centered on the teacher | `_track_subject_x` samples MediaPipe pose centroid and `_build_reframe_filter` writes a piecewise-linear `crop=x='if(between(t,...))'` expression |
| Picture-in-picture progress bar | `_progress_circle_filter` |
| Class-to-Short repurposing teaser | `pipeline._pick_clips_teaser` |
| Aesthetic presets (Golden Hour, Earthy, Minimalist Studio, ...) | `config.AESTHETIC_PRESETS` (ffmpeg `eq` + `colorbalance`) |
| Replicate features from a sample reel | `style_analyzer.analyze` -> `pipeline.apply_style` |

## Run it

```bash
pip install -r requirements.txt
# ffmpeg + ffprobe must be on PATH
uvicorn backend.main:app --reload --port 8000
```

Then open <http://localhost:8000>.

### API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/upload` | Upload class video; returns `job_id` for analysis |
| `GET` | `/api/job/{id}` | Poll job state (`queued` / `running` / `done` / `error`) |
| `GET` | `/api/analysis/{upload_id}` | Pose segments + metadata |
| `POST` | `/api/sample` | Upload inspiration reel; returns its style profile |
| `POST` | `/api/generate` | Render reels (`reels: [{kind, target_format, ...}]`) |
| `GET` | `/api/options` | Available presets, music aliases, formats, kinds |

### Reel kinds

- `sequence` &mdash; chronological best-poses highlight, ~3s per pose
- `teaser` &mdash; 5 short snippets evenly spaced through the class
- `pose_focus` &mdash; only clips matching `pose_filter` (e.g. `["downward_dog","warrior_ii"]`)
- `single` &mdash; the single longest, highest-confidence held pose

### Style replication

When a sample is uploaded, `apply_style` overrides defaults on each new
reel:

- `target_format` &leftarrow; closest of `reel` (9:16), `square`, `youtube` (16:9)
- `color_preset` &leftarrow; nearest preset by brightness / saturation / r-b ratio
- `speed` &leftarrow; mapped from frame-to-frame motion (calm 1.0&times; -> fast-cut 2.0&times;)
- `target_seconds` &leftarrow; ~6 &times; sample's average shot length

The user can still override anything in the form.

## Limitations

- The pose classifier is a small set of geometric heuristics, not a
  trained model. It works well for clearly-held poses with the camera
  facing the mat and degrades for unusual angles or partial occlusion.
- The "music" library only stores filename aliases; drop your licensed
  tracks in `assets/music/` matching `MUSIC_TRACKS` in `config.py` to
  enable them.
- Subject tracking samples 24 points per clip and interpolates &mdash; very
  fast pans across the frame can lag by a fraction of a second.
