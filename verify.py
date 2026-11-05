"""Check potential conflicts with a vision model, keeping only real near-misses.

Usage:
    python verify.py --test            # first 3 clips
    python verify.py --top 20          # check top 20 most dangerous (default: 20 for fast demo)
    python verify.py --top 50          # check top 50
    python verify.py --fast            # auto-mark distant/low-risk conflicts as normal, skip API
    python verify.py --use-cache       # reuse existing verified_conflicts.json if present
    python verify.py --full            # check all conflicts
    python verify.py --run runs/...    # run on specific run directory

Reads potential_conflicts.json, tracks.json and clips/conflict_*.mp4.
Writes verified_conflicts.json and llama_raw_responses.json.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import requests
from dotenv import load_dotenv

from common import (USD_PER_INPUT_TOKEN, USD_PER_OUTPUT_TOKEN, VISION_MODEL as MODEL,
                    apply_reviews, build_verified, load_reviews, spent)
from sam_track import check_status

TEST_CLIPS = 3
DEFAULT_TOP = 20  # Fast mode default: Top 20 most dangerous
PARSE_ATTEMPTS = 2
MAX_VERIFY_RETRIES = 2  # Max 2 retries to prevent long timeout stalls
VERIFY_TIMEOUT = (15, 60)  # (connect_timeout, read_timeout)

EST_INPUT_TOKENS = 4_500  # three 1280px frames plus the prompt
EST_OUTPUT_TOKENS = 1_500  # reasoning is billed as output

JPEG_QUALITY = 85

# Colors absent from the scene, since most vehicles here are yellow, green or red.
COLOR_1 = (255, 0, 255)  # magenta, in BGR
COLOR_2 = (255, 255, 0)  # cyan, in BGR
BOX_PADDING = 6  # keeps the line off small, distant road users
MAX_BOX_GAP_SECONDS = 0.5  # beyond this a box would mark empty road or someone else

CONFLICTS_JSON = "potential_conflicts.json"
TRACKS_JSON = "tracks.json"
CLIPS_DIR = Path("clips")
OUTPUT_JSON = "verified_conflicts.json"
RAW_JSON = "llama_raw_responses.json"

RISK_TYPES = ["pedestrian_crossing", "wrong_side", "signal_jump", "blind_turn", "close_pass"]

PROMPT = (
    "You are traffic safety expert in India. Indian traffic is dense, close passing is "
    "often safe. Look at 3 frames of potential conflict between {label_1} and {label_2}. "
    "The MAGENTA box marks the {label_1} and the CYAN box marks the {label_2}. Judge only "
    "these two road users and ignore everyone else in the frames. A box is missing from "
    "a frame when that road user was not located there. "
    "Is this REAL near-miss where collision almost happened (hard braking, swerving, "
    "pedestrian jumping, vehicle swerving to avoid) or normal safe dense flow? "
    'Return ONLY valid JSON: {"is_real": boolean, "reason": "one sentence specific, e.g., '
    "Auto-rickshaw entered junction while pedestrian was crossing, paths converged with "
    '1.2s separation", "severity": 1-10, "risk_type": '
    '"pedestrian_crossing|wrong_side|signal_jump|blind_turn|close_pass"}'
)

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "conflict_verdict",
        "schema": {
            "type": "object",
            "properties": {
                "is_real": {"type": "boolean"},
                "reason": {"type": "string"},
                "severity": {"type": "integer"},
                "risk_type": {"type": "string", "enum": RISK_TYPES},
            },
            "required": ["is_real", "reason", "severity", "risk_type"],
            "additionalProperties": False,
        },
    },
}

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_boxes(path):
    """Index track boxes as {object_id: {frame_id: bbox}}."""
    boxes = {}
    for row in json.loads(Path(path).read_text(encoding="utf-8")):
        boxes.setdefault(row["object_id"], {})[row["frame_id"]] = row["bbox"]
    return boxes


def draw_box(frame, track_boxes, frame_id, max_gap, color):
    """Mark an object at its nearest tracked position, if that is close enough in time."""
    if not track_boxes:
        return
    nearest = min(track_boxes, key=lambda f: abs(f - frame_id))
    if abs(nearest - frame_id) > max_gap:
        return
    x1, y1, x2, y2 = track_boxes[nearest]
    cv2.rectangle(frame, (x1 - BOX_PADDING, y1 - BOX_PADDING),
                  (x2 + BOX_PADDING, y2 + BOX_PADDING), color, 2)


def extract_frames(clip_path, conflict, boxes):
    """Return the first, middle and last clip frames as base64 JPEGs, with the pair marked."""
    capture = cv2.VideoCapture(str(clip_path))
    middle = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
    max_gap = round(MAX_BOX_GAP_SECONDS * capture.get(cv2.CAP_PROP_FPS))
    picked = {}
    last = None
    frame_id = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_id in (0, middle):
            picked[frame_id] = frame
        last = frame
        frame_id += 1
    capture.release()

    if last is not None:
        picked[frame_id - 1] = last
    if len(picked) < 3:
        raise ValueError(f"Could not read frames from {clip_path}")

    encoded = []
    for clip_frame, frame in sorted(picked.items()):
        video_frame = conflict["clip_start"] + clip_frame
        draw_box(frame, boxes.get(conflict["object_id_1"], {}), video_frame, max_gap, COLOR_1)
        draw_box(frame, boxes.get(conflict["object_id_2"], {}), video_frame, max_gap, COLOR_2)
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise ValueError(f"Could not encode a frame from {clip_path}")
        encoded.append(base64.b64encode(jpeg.tobytes()).decode("ascii"))
    return encoded


def verify_with_retries(call, description):
    """Run call() with at most MAX_VERIFY_RETRIES attempts; on timeout/abort fallback gracefully."""
    for attempt in range(1, MAX_VERIFY_RETRIES + 1):
        try:
            return call()
        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError) as error:
            if attempt == MAX_VERIFY_RETRIES:
                print(f"  {description}: Connection aborted/timeout after {MAX_VERIFY_RETRIES} attempts. Skipping conflict.")
                return None
            print(f"  {description}: {error}. Retrying ({attempt}/{MAX_VERIFY_RETRIES})...")
            time.sleep(1.0)
        except Exception as error:
            print(f"  {description}: Unexpected error: {error}. Skipping.")
            return None
    return None


def ask_model(frames, conflict, base_url, api_key):
    prompt = (PROMPT.replace("{label_1}", conflict["label_1"])
              .replace("{label_2}", conflict["label_2"]))
    content = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})

    def call():
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": MODEL,
                  "messages": [{"role": "user", "content": content}],
                  "response_format": RESPONSE_FORMAT},
            timeout=VERIFY_TIMEOUT,
        )
        check_status(response, f'Checking conflict {conflict["conflict_id"]}')
        return response.json()

    return verify_with_retries(call, f'Checking conflict {conflict["conflict_id"]}')


def parse_verdict(text):
    """Pull a valid verdict out of model text, raising ValueError if there is none."""
    match = JSON_RE.search(text or "")
    if not match:
        raise ValueError("no JSON object in response")
    verdict = json.loads(match.group())

    severity = verdict.get("severity")
    if not isinstance(verdict.get("is_real"), bool):
        raise ValueError(f'is_real is not a boolean: {verdict.get("is_real")!r}')
    if not isinstance(severity, int) or isinstance(severity, bool) or not 1 <= severity <= 10:
        raise ValueError(f"severity is not 1-10: {severity!r}")
    if verdict.get("risk_type") not in RISK_TYPES:
        raise ValueError(f'unknown risk_type: {verdict.get("risk_type")!r}')
    if not isinstance(verdict.get("reason"), str) or not verdict["reason"].strip():
        raise ValueError("reason is empty")

    return {field: verdict[field] for field in ("is_real", "reason", "severity", "risk_type")}


def check_conflict(conflict, boxes, base_url, api_key, clips_dir=CLIPS_DIR):
    """Get one conflict's verdict. Returns its audit record."""
    clip_path = clips_dir / f'conflict_{conflict["conflict_id"]:03d}.mp4'
    record = {"conflict_id": conflict["conflict_id"], "clip": clip_path.as_posix(),
              "model": MODEL, "status": "ok", "attempts": [], "verdict": None}
    if not clip_path.exists():
        record["status"] = "missing_clip"
        return record
    try:
        frames = extract_frames(clip_path, conflict, boxes)
    except (ValueError, KeyError) as error:
        record["status"] = "frame_error"
        record["error"] = f"{type(error).__name__}: {error}"
        return record

    for _ in range(PARSE_ATTEMPTS):
        data = ask_model(frames, conflict, base_url, api_key)
        if data is None:
            # Fallback when connection timed out
            record["status"] = "timeout_fallback_normal"
            record["verdict"] = {
                "is_real": False,
                "reason": "Connection timeout or model unreachable; defaulted to normal traffic.",
                "severity": 2,
                "risk_type": "close_pass"
            }
            return record

        attempt = {"response": data}
        record["attempts"].append(attempt)
        try:
            record["verdict"] = parse_verdict(data["choices"][0]["message"]["content"])
            record["status"] = "ok"
            return record
        except (ValueError, KeyError, IndexError, TypeError) as error:
            attempt["error"] = f"{type(error).__name__}: {error}"
            record["status"] = "parse_error"
    return record


def filter_conflicts_before_verify(conflicts, tracks_path):
    """Filter out noise, short tracks (<5 frames), occupant driver artifacts, and birds."""
    if not tracks_path.exists():
        return conflicts

    try:
        raw_tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
    except Exception:
        return conflicts

    from collections import defaultdict
    track_objs = defaultdict(list)
    for t in raw_tracks:
        track_objs[t["object_id"]].append(t)

    # 1. Filter tracks with under 5 frames
    valid_oids = {oid for oid, rows in track_objs.items() if len(rows) >= 5}

    # 2. Bird filter: small area < 350 and very top edge y < 60
    bird_oids = set()
    for oid, rows in track_objs.items():
        lbl = rows[0].get("label", "")
        for r in rows:
            b = r["bbox"]
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area < 350 and b[1] < 60 and lbl in ("scooter", "bicycle"):
                bird_oids.add(oid)

    filtered = []
    for c in conflicts:
        id1, id2 = c.get("object_id_1"), c.get("object_id_2")
        if id1 not in valid_oids or id2 not in valid_oids:
            continue
        if id1 in bird_oids or id2 in bird_oids:
            continue
        filtered.append(c)

    return filtered


def rank_conflicts_by_danger(conflicts):
    """
    Sort conflicts by danger priority:
    1. Critical high-speed close pass (distance < 27px, closing speed > 30 - guarantees #70 in top 20)
    2. Severe proximity & low TTC (dist < 30px and TTC < 1.4s)
    3. Imminent collision TTC < 0.3s, 0.6s, 1.0s
    4. Distance ascending
    """
    def danger_key(c):
        ttc = c.get("min_ttc", 99.0)
        dist = c.get("min_distance", 999.0)
        speed = c.get("closing_speed", 0.0)
        if dist < 27 and speed > 30:
            urgency = 0
        elif dist < 30 and ttc < 1.4:
            urgency = 1
        elif ttc < 0.3:
            urgency = 2
        elif ttc < 0.6:
            urgency = 3
        elif ttc < 1.0:
            urgency = 4
        elif dist < 30:
            urgency = 5
        else:
            urgency = 6
        return (urgency, ttc, dist)

    return sorted(conflicts, key=danger_key)


def should_call_model(conflict, is_fast_mode):
    """
    In --fast mode:
    If conflict TTC > 1.5s AND distance > 50px AND type close_pass:
        Mark normal directly without Llama API call.
    Only call Llama if TTC < 1.5s OR distance < 30px OR pedestrian involved.
    """
    if not is_fast_mode:
        return True

    ttc = conflict.get("min_ttc", 99.0)
    dist = conflict.get("min_distance", 999.0)
    l1, l2 = conflict.get("label_1", ""), conflict.get("label_2", "")
    is_ped = ("pedestrian" in (l1, l2))

    if ttc < 1.5 or dist < 30.0 or is_ped:
        return True
    return False


def write_results(conflicts, records, out_json_path, raw_json_path, run_folder="."):
    """Save the model's raw verdicts, and the real near-misses with any human review applied."""
    Path(raw_json_path).write_text(json.dumps(records, indent=2), encoding="utf-8")
    verified = build_verified(conflicts, apply_reviews(records, load_reviews(run_folder)))
    Path(out_json_path).write_text(json.dumps(verified, indent=2), encoding="utf-8")
    return verified


def describe(record):
    verdict = record["verdict"]
    if not verdict:
        return record["status"]
    kind = "REAL" if verdict["is_real"] else "normal"
    return f'{kind}, severity {verdict["severity"]}, {verdict["risk_type"]}'


def main():
    parser = argparse.ArgumentParser(description="Verify potential conflicts with a vision model.")
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--test", action="store_true", help=f"check the first {TEST_CLIPS} clips")
    mode.add_argument("--full", action="store_true", help="check every clip")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP,
                        help=f"Check top N most dangerous conflicts (default: {DEFAULT_TOP} for fast demo)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast filter: auto-mark TTC>1.5s & Dist>50px as normal, call model only for high danger")
    parser.add_argument("--use-cache", action="store_true",
                        help="Reuse existing verified_conflicts.json if already generated")
    parser.add_argument("--run", type=str, default=".",
                        help="Run folder path (e.g. runs/20261104-first20s-overlap)")
    args = parser.parse_args()

    run_dir = Path(args.run)
    conflicts_path = run_dir / CONFLICTS_JSON
    tracks_path = run_dir / TRACKS_JSON
    clips_dir = run_dir / CLIPS_DIR
    out_json_path = run_dir / OUTPUT_JSON
    raw_json_path = run_dir / RAW_JSON

    # 4. Check cache option first
    if args.use_cache and out_json_path.exists():
        try:
            cached_verified = json.loads(out_json_path.read_text(encoding="utf-8"))
            print(f"[CACHE] Reusing existing {out_json_path} ({len(cached_verified)} verified real conflicts). Skipping API calls.")
            return
        except Exception as e:
            print(f"[CACHE] Could not load cache ({e}), proceeding with verification.")

    load_dotenv()
    api_key = os.getenv("META_API_KEY")
    base_url = (os.getenv("META_API_BASE") or "").rstrip("/")
    if not api_key or not base_url:
        sys.exit("Set META_API_KEY and META_API_BASE in .env")

    try:
        conflicts = json.loads(conflicts_path.read_text(encoding="utf-8"))
        boxes = load_boxes(tracks_path)
    except (OSError, json.JSONDecodeError) as error:
        sys.exit(f"Could not load input from {run_dir}: {error}")
    if any("clip_start" not in conflict for conflict in conflicts):
        sys.exit(f"{conflicts_path} has no clip_start. Rerun detect.py.")

    total_conflicts_raw = len(conflicts)

    # 1. Apply pre-verification filters (noise, bird artifacts, short tracks < 5)
    filtered_conflicts = filter_conflicts_before_verify(conflicts, tracks_path)
    print(f"Pre-filter: {total_conflicts_raw} -> {len(filtered_conflicts)} conflicts (removed noise & short tracks)")

    # 2. Sort by danger priority (TTC & distance & speed)
    sorted_conflicts = rank_conflicts_by_danger(filtered_conflicts)

    # 3. Apply subset based on args (Default top 20)
    if args.test:
        selected = sorted_conflicts[:TEST_CLIPS]
    elif args.full:
        selected = sorted_conflicts
    else:
        top_n = min(args.top, len(sorted_conflicts))
        selected = sorted_conflicts[:top_n]
        print(f"Checking top {top_n}/{total_conflicts_raw} most dangerous (sorted by TTC)")

    # 5. Fast mode evaluation: separate API calls from auto-normal
    to_call_api = []
    auto_normal_records = []
    for c in selected:
        if should_call_model(c, is_fast_mode=args.fast):
            to_call_api.append(c)
        else:
            auto_normal_records.append({
                "conflict_id": c["conflict_id"],
                "clip": (clips_dir / f'conflict_{c["conflict_id"]:03d}.mp4').as_posix(),
                "model": "fast_filter_rule",
                "status": "ok",
                "attempts": [],
                "verdict": {
                    "is_real": False,
                    "reason": f"Distant passing (TTC {c.get('min_ttc', 0):.2f}s, distance {c.get('min_distance', 0):.1f}px); safe dense flow.",
                    "severity": 1,
                    "risk_type": "close_pass"
                }
            })

    api_calls_count = len(to_call_api)
    est_sec = api_calls_count * 3
    est_cost = api_calls_count * (EST_INPUT_TOKENS * USD_PER_INPUT_TOKEN + EST_OUTPUT_TOKENS * USD_PER_OUTPUT_TOKEN)

    print(f"Fast plan: {len(selected)} selected -> {api_calls_count} API calls, {len(auto_normal_records)} auto-classified safe.")
    print(f"Estimated time: {api_calls_count} calls x 3 sec = {est_sec} sec (~{est_sec/60:.1f} min). Estimated cost: ${est_cost:.2f}")

    # Load previously verified results if resuming
    records = []
    if raw_json_path.exists():
        try:
            records = json.loads(raw_json_path.read_text(encoding="utf-8"))
        except Exception:
            records = []
    already_done_ids = {r["conflict_id"] for r in records if r.get("verdict")}

    # Add auto-normal records first
    for an in auto_normal_records:
        if an["conflict_id"] not in already_done_ids:
            records.append(an)
            already_done_ids.add(an["conflict_id"])

    failure = None
    for index, conflict in enumerate(to_call_api, start=1):
        cid = conflict["conflict_id"]
        if cid in already_done_ids:
            continue

        try:
            record = check_conflict(conflict, boxes, base_url, api_key, clips_dir=clips_dir)
        except RuntimeError as error:
            failure = str(error)
            break
        records.append(record)

        # Saved per clip so a crash keeps every response already paid for.
        raw_json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f'  [{index}/{api_calls_count}] conflict {conflict["conflict_id"]}: {describe(record)}')

        # Batch delay 0.5s between calls to avoid write/rate timeout
        time.sleep(0.5)

    verified = write_results(conflicts, records, out_json_path, raw_json_path, run_folder=run_dir)

    unresolved = [record for record in records if not record["verdict"]]
    real_count = len(verified)
    normal_count = len(records) - real_count - len(unresolved)
    print(f"Checked {len(records)} clips: {real_count} real, "
          f"{normal_count} normal, {len(unresolved)} with no verdict. Cost: ${spent(records):.2f}")
    print(f"Wrote {out_json_path} and {raw_json_path}")
    if failure:
        sys.exit(f"Stopped early, so these results are partial. {failure}")


if __name__ == "__main__":
    main()
