# Kavacham: Turn 90 sec phone video into junction-risk report

Live: https://kavacham.streamlit.app/

Kavacham tracks road users in a phone video of a junction using Meta's SAM 3.1 model. It writes per-frame tracks and an annotated video for later risk analysis.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put your Meta Model API key in `.env`. The SAM endpoint lives at `https://api.meta.ai/v1`.

## Usage

```bash
python sam_track.py --test
python sam_track.py --full
```

- `--test` processes the first 100 frames. Run this first.
- `--full` processes the whole video.
- `--video PATH` picks another input file. The default is `video.mp4`.

The script prints the estimated cost before it uploads anything.

Then find potential conflicts in the tracks:

```bash
python detect.py --test
python detect.py
```

- `--test` keeps the first 20 conflicts by start frame. Run this first.
- `--tracks PATH` and `--video PATH` pick other inputs. Use the same video you tracked.

Then ask a vision model which conflicts are real near-misses:

```bash
python verify.py --test
python verify.py --full
```

- `--test` checks the first 3 clips. Run this first.
- `--full` checks every clip.

The script prints the estimated cost before it starts and the actual cost when it finishes.

Then draw the flagged pair on every clip:

```bash
python generate_marked.py
```

It writes `clips/marked/conflict_NNN.mp4`, with a red box on TARGET 1 and a green box on TARGET 2 on each frame where the tracker saw them, and other road users in faint gray. `detect.py` deletes these marked clips when it replaces the conflicts, so rerun `generate_marked.py` after it.

For the dashboard gallery, render one tracked video per conflict:

```bash
python generate_88_trackings.py
```

It writes `clips/individual/conflict_<id>_tracked.mp4` from `video.mp4`, covering 15 frames before and after each conflict. Only the flagged pair is drawn: red for TARGET 1, green for TARGET 2, with dots tracing their last 10 frames and a summary bar along the bottom. A box missing for up to 5 frames is filled in by interpolation; longer gaps leave the box off. The risk type comes from `verify.py`, so run that first. `detect.py` deletes these videos when it replaces the conflicts.

Then draw the risk-zone map:

```bash
python hotspot.py
```

It splits the camera frame into left, centre and right thirds and counts the conflicts in each. A third with a verified near-miss is high risk, one with at least a quarter of the potential conflicts is medium, and the rest are low. It writes `hotspot_frame.png`, the zones drawn on a real video frame. It also asks Muse Image (`muse-image-1.0`, $0.01 per image) for an overhead illustration, `hotspot.png`, with its search turned off. That picture is invented from the zone counts, not a view of the real junction. If the call fails, `hotspot.png` is a copy of the camera-frame map instead, and `hotspot.json` records which one it is. Rerun it after `verify.py` or `detect.py` change the results.

Then open the audit dashboard:

```bash
streamlit run app.py
```

It opens on a landing page and loads nothing until you choose:

- **See demo** shows the results already in this folder, from `video.mp4`.
- **Drop a video**, or pick one already in the Kavacham folder under **Use a video from this folder**. The folder route skips the browser upload, so use it for large phone videos. A dropped file is saved as `uploaded_video.<ext>`.
- **Ingest and reduce** happen straight away, in a new `runs/<timestamp>/` folder (`auto_pipeline.py`):
  - The page shows the original's size, resolution, frame rate, length, frame count and codec. Videos under 5 sec are rejected. Over 120 sec or over 1 GB, you're warned about cost or told to use the folder route.
  - The original is kept as `original.<ext>`, hard-linked rather than copied when it's already on this disk.
  - It's reduced to `video.mp4`: shorter side 720 px with Lanczos scaling, 10 fps, x264 at CRF 18, preset slow. The page shows the result, a warning if the bitrate falls under 1.5 Mbps, both first frames side by side, a preview, and "Ready for tracking".
  - `reduce_info.json` records both videos, the exact ffmpeg command and the SAM cost. Picking the same video again reuses this folder instead of reducing again.
  - A 700 MB 4K 60 fps clip becomes about 70 MB in under two minutes, and SAM's cost falls six-fold, since it bills per frame. The larger file takes longer to upload to SAM, so `sam_track.py` allows 15 minutes for the upload.
- **Start backend tracking** then runs the whole pipeline in that folder, from `sam_track.py --full` to `hotspot.py`, so the demo results are never overwritten. Only one run can be in progress at a time. `runs/.active.json` holds the lock, so a second browser session can't pay for the same work twice. It shows a progress bar and each step's log, saves the costs to `costs.json` and opens the results when it finishes. If a step fails, **Show partial results** opens what finished. The pipeline runs with the venv's Python, since it needs OpenCV, requests and python-dotenv. Uploads can be up to 1 GB, but restart Streamlit after changing `.streamlit/config.toml`, because the upload limit is only read at startup.

Tracking spends real API credit, so the button only appears when a `META_API_KEY` is set (in `.env` locally, or in Streamlit secrets) and the page is opened on this machine. On a public deployment, add `OWNER_PASSCODE` to the app's secrets: the owner signs in with it from the sidebar, and everyone else only sees the demo. `packages.txt` installs ffmpeg on Streamlit Cloud, and `requirements.txt` uses `opencv-python-headless` because the cloud has no display.

The results page tells the story top to bottom:

1. The tracked video with every road user boxed.
2. Three cards: potential conflicts found, clips checked by the model, and real near-misses left.
3. Each real near-miss, with its tracked video beside a danger card. The card covers what happened, the risk numbers, where it is in the frame and a suggested fix.

Fix suggestions come from a fixed table by risk type, for an engineer to assess. "Where" is described in camera terms, because the footage can't say which road or direction it is. The upload button doesn't process videos yet. The sidebar holds the tracking details and the full audit log, with downloads for every JSON file.

The dashboard needs only Streamlit, plus `ffmpeg` and `ffprobe` on your PATH for video. It imports nothing from the pipeline scripts, only `common.py`, which uses the standard library. So any Python with Streamlit 1.54 or later can run it, without the venv, OpenCV, requests or python-dotenv.

It opens on port 8901, because Windows reserves the default port 8501 on some machines. The dashboard reads the files above and needs `ffmpeg` on your PATH to play videos in the browser. It keeps converted copies in `.web_video/`, which is safe to delete. Set `PLACEHOLDER_VIDEO` to `False` in `app.py` once `video.mp4` is real footage.

## Output

`tracks.json` holds one entry per object per frame:

```json
[
  {
    "frame_id": 0,
    "object_id": 1,
    "label": "auto-rickshaw",
    "bbox": [412, 233, 590, 401],
    "confidence": 0.96
  }
]
```

`bbox` is `[x1, y1, x2, y2]` in pixels of the video.

`masked.mp4` is the video with a coloured box and label on every tracked object. Each run overwrites both files.

`potential_conflicts.json` holds one entry per conflict, in order of start frame:

```json
[
  {
    "conflict_id": 81,
    "object_id_1": 79,
    "label_1": "motorcycle",
    "object_id_2": 91,
    "label_2": "motorcycle",
    "frame_start": 88,
    "frame_end": 99,
    "min_distance": 21.78,
    "min_ttc": 0.657,
    "closing_speed": 79.89,
    "clip_start": 77
  }
]
```

- `min_distance` is the smallest distance between box centers during the conflict, in pixels.
- `min_ttc` is the lowest time-to-collision during the conflict, in seconds.
- `closing_speed` is the highest closing speed during the conflict, in pixels per second.
- `clip_start` is the video frame where the conflict's clip begins. `verify.py` uses it to find boxes in `tracks.json`.

`clips/conflict_NNN.mp4` holds 3 seconds of `video.mp4` around the lowest-TTC frame of each conflict. Each `detect.py` run replaces the old clips and JSON, so a `--test` run leaves only 20.

`verified_conflicts.json` holds the conflicts the model called real, highest severity first. Each entry has every field from `potential_conflicts.json`, plus `clip`, `is_real`, `reason`, `severity` (1 to 10) and `risk_type` (`pedestrian_crossing`, `wrong_side`, `signal_jump`, `blind_turn` or `close_pass`).

`llama_raw_responses.json` is the audit log. It holds the full API response for every clip, including rejected conflicts and replies that couldn't be parsed. Clips with `status` `parse_error`, `missing_clip` or `frame_error` have no verdict and are left out of `verified_conflicts.json`. Each `verify.py` run replaces both files, so a `--test` run overwrites the results of a `--full` run.

## How it works

1. **Prompts:** SAM 3.1 accepts one concept per request, so the video is sent once for each label: pedestrian, auto-rickshaw, motorcycle, car, bus, bicycle and scooter.
2. **De-duplication:** if two labels claim the same object on a frame, the higher-confidence detection is kept.
3. **Tracking:** greedy IoU matching gives each object one ID across all labels and frames.
4. **Filtering:** `detect.py` drops tracks shorter than 5 frames and keeps every label, including any with no tracks. It also drops a pedestrian track when at least 80% of its box sits inside a vehicle box on at least half its frames, since those are usually drivers, riders or passengers.
5. **Conflicts:** for each pair of objects seen together, a frame is flagged when center distance is under 80 pixels, closing speed is above 0, and TTC is under 2.5 seconds. Flagged frames within 0.5 seconds merge into one conflict.

TTC is distance divided by closing speed, assuming constant velocity. Closing speed uses centers averaged over 0.5 seconds, because box jitter on parked vehicles otherwise reads as motion.
6. **Verification:** `verify.py` sends the first, middle and last frame of each clip to `muse-spark-1.3`, Meta's current vision model, and keeps the conflicts it calls real. It draws a magenta box around the first object and a cyan box around the second, and tells the model to judge only those two. The model's reasons can mention those colors, while marked clips in the dashboard use red and green. Llama 4 Maverick isn't available on Meta Model API, so change `MODEL` in `verify.py` if that changes. A reply that isn't valid is requested once more before the clip is marked `parse_error`.

## Cost

SAM 3.1 video costs $0.20 per 1,000 frames, with a 50-frame minimum per request. Each frame is billed once per label, so seven labels multiply the frame count by seven.

| Run | Frames | Billed frames | Cost |
| --- | --- | --- | --- |
| `--test` | 100 | 700 | $0.14 |
| `--full`, 90 sec at 30 fps | 2,700 | 18,900 | $3.78 |

`verify.py` costs roughly one cent per clip at $1.25 per million input tokens and $4.25 per million output tokens, so 88 clips cost about $1. The real figure depends on how much the model reasons.

## Limits

- SAM 3.1 tracks up to 16 objects per frame for each label. On a crowded junction the cap may miss objects, and `detect.py` can't flag a conflict involving an object that was never tracked.
- Conflicts are potential, not confirmed. Review the clips before reporting them. The occupant filter targets riders and drivers labelled as pedestrians, but it can miss some. From a high camera angle it can also remove a real person standing beside or in front of a vehicle for most of their track.
- `verify.py` sees 3 frames about 1.5 seconds apart, so braking or a brief swerve can happen between frames. A box is left out when its object wasn't tracked within 0.5 seconds of that frame, so some frames show one box or none. Review verdicts against the clips.
- Distances and speeds are in pixels, so 80 pixels covers far more road at the back of the scene than the front, and box centers sit far apart on large vehicles even when their edges nearly touch.
- Videos must be MP4 or MOV in H.264, H.265 or MPEG-4, up to 15,000 frames and 1 GiB.
- Results can vary slightly between runs of the same video.
- `masked.mp4` is MPEG-4, not H.264. It plays in VLC and desktop players, but most browsers won't play it.
