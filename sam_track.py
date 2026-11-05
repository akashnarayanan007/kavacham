"""Track road users in a video with Meta Model API SAM 3.1.

Usage:
    python sam_track.py --test    # first 100 frames
    python sam_track.py --full    # whole video

Writes tracks.json and masked.mp4.
"""

import argparse
import json
import math
import os
import random
import re
import sys
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import requests
from dotenv import load_dotenv

# In common.py so the dashboard can quote the cost before a run.
from common import PROMPTS, SAM_MIN_BILLED_FRAMES, sam_cost as estimate_cost

MODEL = "sam-3.1"

TEST_FRAMES = 100

MAX_RETRIES = 5
RETRYABLE_STATUS = {429, 500, 502, 503}
TIMEOUT = (30, 300)  # connect, read between stream chunks
# requests applies the connect timeout to sending the whole body, so a 70 MB CRF 18
# video on an ordinary uplink needs minutes, not 30 seconds.
UPLOAD_TIMEOUT = (900, 300)

MIN_CONFIDENCE = 0.3
DUPLICATE_IOU = 0.7  # same object claimed by two labels on one frame
MATCH_IOU = 0.3
MAX_MISSED_FRAMES = 20  # keep-alive bridged gaps increased from 10 to 20 frames
SAME_OBJECT_IOU = 0.5  # same-frame boxes overlapping this much are one road user
SAME_OBJECT_CENTRE = 0.3  # ...or whose centres sit within this share of the smaller box
MERGE_SHARE = 0.5  # two SAM tracks are one road user if they match on half their shared frames
# Box sizes a road user may move per missed frame and still be rejoined. Measured on
# backup.mp4, 95% of road users move under 0.25 box sizes per frame; anything looser
# rejoins different road users at chunk boundaries and fakes sudden approaches.
REJOIN_DISTANCE = 0.5
# SAM picks at most 16 objects per prompt at the start of a request and follows only those,
# so later arrivals are missed. Short chunks make it pick again every 10 seconds.
CHUNK_FRAMES = 100
# Overlapping chunks lets both chunks see the same road users on the same frames, so they
# are merged directly instead of guessed across a gap.
CHUNK_OVERLAP = 5
RAW_DIR = Path("sam_raw")  # SAM's own output, so tracking can be rebuilt without paying again

TILE_OVERLAP = 0.1  # share of the long side neighbouring bands share, so a road user at a seam is whole in one
TILE_ZOOM = 2  # tiles are cut from the original at up to twice video.mp4's resolution
TILE_CRF = 18
EDGE_MARGIN = 3  # tile pixels: a box this close to an inner seam is cut off there

# SAM can flip between these on the same vehicle, so they share a track.
TRACK_GROUP = {"scooter": "motorcycle"}

OUTPUT_JSON = "tracks.json"
OUTPUT_VIDEO = "masked.mp4"

COLORS = {
    "pedestrian": (0, 0, 255),
    "auto-rickshaw": (0, 215, 255),
    "motorcycle": (255, 0, 255),
    "car": (0, 255, 0),
    "bus": (255, 128, 0),
    "bicycle": (255, 255, 0),
    "scooter": (203, 102, 255),
}

FRAME_RE = re.compile(r"<(\d+)f>")
RECORD_RE = re.compile(r"(\d+)<\|box;([^|>]*)\|>")


class RetryableError(Exception):
    """A transient failure worth another attempt."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def with_retries(call, description):
    """Run call(), backing off on RetryableError."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call()
        except (RetryableError, requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError) as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"{description} failed after {MAX_RETRIES} attempts: {error}"
                ) from error
            delay = getattr(error, "retry_after", None)
            if delay is None:
                delay = 2 ** (attempt - 1) + random.uniform(0, 1)
            print(f"  {description}: {error}. Retrying in {delay:.1f}s "
                  f"({attempt}/{MAX_RETRIES - 1})")
            time.sleep(delay)


def check_status(response, description):
    """Raise RetryableError for transient statuses, RuntimeError for permanent ones."""
    if response.ok:
        return
    message = f"HTTP {response.status_code}: {response.text[:300]}"
    if response.status_code in RETRYABLE_STATUS:
        retry_after = response.headers.get("Retry-After", "")
        raise RetryableError(message, float(retry_after) if retry_after.isdigit() else None)
    raise RuntimeError(f"{description} failed. {message}")


def upload_video(path, base_url, api_key):
    def call():
        with open(path, "rb") as video:
            response = requests.post(
                f"{base_url}/files",
                headers={"Authorization": f"Bearer {api_key}"},
                data={"purpose": "user_data"},
                files={"file": video},
                timeout=UPLOAD_TIMEOUT,
            )
        check_status(response, "Upload")
        return response.json()["id"]

    return with_retries(call, "Upload")


def delete_file(file_id, base_url, api_key):
    """Uploaded files never expire and count toward team storage."""
    try:
        requests.delete(
            f"{base_url}/files/{file_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT,
        )
    except requests.RequestException as error:
        print(f"Could not delete uploaded file {file_id}: {error}")


def read_stream(response, label):
    """Collect one label's frame lines from the server-sent event stream."""
    lines = []
    for raw_line in response.iter_lines(decode_unicode=False):
        line = raw_line.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            break
        event = json.loads(data)
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            lines.append(event.get("delta") or "")
            print(f"\r  {label}: {len(lines)} frames", end="", flush=True)
        elif event_type == "response.completed":
            print()
            return lines
        elif event_type in ("response.failed", "error"):
            print()
            raise RetryableError(f"stream failed: {data[:300]}")
    print()
    # A dropped stream looks like a normal end, so never accept it as a result.
    raise RetryableError("stream ended before response.completed")


def segment(file_id, label, base_url, api_key):
    def call():
        with requests.post(
            f"{base_url}/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            json={
                "model": MODEL,
                "input": [{
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": label},
                        {"type": "input_video", "file_id": file_id},
                    ],
                }],
                "stream": True,
                "metadata": {"include_confidence": "true", "mask_encoding": "one_bit"},
            },
            stream=True,
            timeout=TIMEOUT,
        ) as response:
            check_status(response, f"Segmenting {label}")
            return read_stream(response, label)

    return with_retries(call, f"Segmenting {label}")


def parse_lines(lines, label, frame_size, source="full"):
    """Turn SAM output lines into detections, scaled to our frame size.

    Each keeps SAM's own object ID, which is stable while the object stays in view, and
    the pass or tile it came from, since IDs are only unique within one response.
    """
    frame_w, frame_h = frame_size
    detections = []
    warned_shape = False
    for line in lines:
        for text in line.splitlines():
            if not text.strip():
                continue
            marker = FRAME_RE.match(text)
            if not marker:
                print(f"  Skipped unreadable {label} line: {text[:80]}")
                continue
            frame_id = int(marker.group(1))
            for sam_id, body in RECORD_RE.findall(text):
                fields = dict(field.split("=", 1) for field in body.split(";"))
                scale_x = frame_w / float(fields["w"])
                scale_y = frame_h / float(fields["h"])
                if not warned_shape and not math.isclose(scale_x, scale_y, rel_tol=0.02):
                    print(f"  Warning: SAM saw {fields['w']}x{fields['h']} but this video "
                          f"reads as {frame_w}x{frame_h}. Check its rotation.")
                    warned_shape = True
                confidence = float(fields["c"]) if "c" in fields else None
                if confidence is not None and confidence < MIN_CONFIDENCE:
                    continue
                bbox = [
                    round(float(fields["x1"]) * scale_x),
                    round(float(fields["y1"]) * scale_y),
                    round(float(fields["x2"]) * scale_x),
                    round(float(fields["y2"]) * scale_y),
                ]
                # Filter false positives: Bird labelled scooter/motorcycle in top sky area
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if label in ("scooter", "motorcycle") and area < 500 and bbox[1] < 100:
                    continue
                detections.append({
                    "frame_id": frame_id,
                    "label": label,
                    "bbox": bbox,
                    "confidence": confidence,
                    "sam_key": (source, label, int(sam_id)),
                })
    return detections


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def remove_duplicates(detections):
    """Keep the most confident detection when labels overlap on one frame."""
    kept = []
    for detection in sorted(detections, key=lambda d: d["confidence"] or 0, reverse=True):
        if all(iou(detection["bbox"], other["bbox"]) < DUPLICATE_IOU for other in kept):
            kept.append(detection)
    return kept


def same_object(a, b):
    """Two boxes on one frame that are the same road user: they overlap heavily, or
    their centres almost coincide (copies from two prompts or tiles can differ in size)."""
    if iou(a, b) >= SAME_OBJECT_IOU:
        return True
    size = min(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1])
    centre_a = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    centre_b = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return math.dist(centre_a, centre_b) <= SAME_OBJECT_CENTRE * max(size, 1)


def assign_track_ids(detections):
    """One ID per road user, built from SAM's own object IDs.

    SAM keeps an object's ID stable while it stays in view, so each (pass, prompt, SAM ID)
    is a track segment. Then:
    1. segments that are the same road user, found by two prompts (motorcycle and scooter)
       or two tiles, are merged when their boxes match on at least MERGE_SHARE of the
       frames they share;
    2. a road user whose SAM ID broke, because it was hidden or fell outside the 16-object
       cap, is rejoined when a new segment of the same kind starts nearby within
       MAX_MISSED_FRAMES;
    3. on each frame only the most confident box of each road user is kept.
    Detections without a SAM ID fall back to IoU matching.
    """
    if not detections or any("sam_key" not in d for d in detections):
        return assign_track_ids_by_iou(detections)

    segments = {}
    for detection in detections:
        segments.setdefault(detection["sam_key"], {})
        frame_best = segments[detection["sam_key"]].get(detection["frame_id"])
        if frame_best is None or (detection["confidence"] or 0) > (frame_best["confidence"] or 0):
            segments[detection["sam_key"]][detection["frame_id"]] = detection

    # 1. merge segments that are the same road user
    parent = {key: key for key in segments}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    by_frame = {}
    for key, frames in segments.items():
        for frame_id, detection in frames.items():
            by_frame.setdefault(frame_id, []).append((key, detection["bbox"]))
    matches = {}
    for boxes in by_frame.values():
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i][0] != boxes[j][0] and same_object(boxes[i][1], boxes[j][1]):
                    pair = tuple(sorted((boxes[i][0], boxes[j][0])))
                    matches[pair] = matches.get(pair, 0) + 1
    for (a, b), count in matches.items():
        shared = len(segments[a].keys() & segments[b].keys())
        if count >= MERGE_SHARE * shared:
            parent[root(a)] = root(b)

    objects = {}
    for key, frames in segments.items():
        merged = objects.setdefault(root(key), {})
        for frame_id, detection in frames.items():
            best = merged.get(frame_id)
            if best is None or (detection["confidence"] or 0) > (best["confidence"] or 0):
                merged[frame_id] = detection

    # 2. rejoin a road user whose SAM ID broke for a few frames
    def group(frames):
        labels = [d["label"] for d in frames.values()]
        label = max(set(labels), key=labels.count)
        return TRACK_GROUP.get(label, label)

    def gap_distance(earlier, later):
        last, first = earlier[max(earlier)], later[min(later)]
        a, b = last["bbox"], first["bbox"]
        size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 1)
        distance = math.dist(((a[0] + a[2]) / 2, (a[1] + a[3]) / 2),
                             ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
        return distance / size

    chains = sorted(objects.values(), key=lambda frames: min(frames))
    joined = []  # each is a dict frame_id -> detection
    for frames in chains:
        start, kind = min(frames), group(frames)
        # Match candidates: normal rejoin or chunk boundary bridge (<20px)
        candidates = []
        for index, chain in enumerate(joined):
            last_f = max(chain)
            gap = start - last_f
            if 0 < gap <= MAX_MISSED_FRAMES and group(chain) == kind:
                dist_ratio = gap_distance(chain, frames)
                last_box = chain[last_f]["bbox"]
                first_box = frames[start]["bbox"]
                px_dist = math.dist(((last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2),
                                    ((first_box[0] + first_box[2]) / 2, (first_box[1] + first_box[3]) / 2))
                # If within 20px (especially across boundary 99->100) or under REJOIN_DISTANCE
                if px_dist <= 20 or dist_ratio <= REJOIN_DISTANCE * gap:
                    candidates.append((dist_ratio, index))

        if candidates:
            joined[min(candidates)[1]].update(frames)
        else:
            joined.append(dict(frames))

    # 3. one row per road user per frame, numbered by first appearance
    results = []
    for object_id, frames in enumerate(sorted(joined, key=lambda frames: min(frames)), start=1):
        for frame_id in sorted(frames):
            detection = frames[frame_id]
            results.append({
                "frame_id": frame_id,
                "object_id": object_id,
                "label": detection["label"],
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
            })
    # Across road users, the same box can still appear twice on a frame; keep the stronger.
    by_frame = {}
    for row in results:
        by_frame.setdefault(row["frame_id"], []).append(row)
    kept = []
    for rows in by_frame.values():
        frame_kept = []
        for row in sorted(rows, key=lambda r: r["confidence"] or 0, reverse=True):
            if not any(same_object(row["bbox"], other["bbox"]) for other in frame_kept):
                frame_kept.append(row)
        kept.extend(frame_kept)
    return sorted(kept, key=lambda r: (r["frame_id"], r["object_id"]))


def assign_track_ids_by_iou(detections):
    """Greedy IoU matching of each frame's detections against live tracks."""
    frames = {}
    for detection in detections:
        frames.setdefault(detection["frame_id"], []).append(detection)

    tracks = []  # {"id", "group", "bbox", "last_frame"}
    next_id = 1
    results = []

    for frame_id in sorted(frames):
        current = remove_duplicates(frames[frame_id])
        live = [t for t in tracks if frame_id - t["last_frame"] <= MAX_MISSED_FRAMES]

        pairs = []
        for d_index, detection in enumerate(current):
            group = TRACK_GROUP.get(detection["label"], detection["label"])
            for t_index, track in enumerate(live):
                overlap = iou(detection["bbox"], track["bbox"])
                if track["group"] == group and overlap >= MATCH_IOU:
                    pairs.append((overlap, d_index, t_index))

        matches = {}
        used_tracks = set()
        for _, d_index, t_index in sorted(pairs, reverse=True):
            if d_index not in matches and t_index not in used_tracks:
                matches[d_index] = t_index
                used_tracks.add(t_index)

        for d_index, detection in enumerate(current):
            if d_index in matches:
                track = live[matches[d_index]]
            else:
                track = {
                    "id": next_id,
                    "group": TRACK_GROUP.get(detection["label"], detection["label"]),
                }
                next_id += 1
                tracks.append(track)
            track["bbox"] = detection["bbox"]
            track["last_frame"] = frame_id
            results.append({
                "frame_id": frame_id,
                "object_id": track["id"],
                "label": detection["label"],
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
            })

    return results


def open_writer(path, fps, frame_size):
    # Not avc1: without OpenH264 installed, OpenCV crashes at process exit on Windows.
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer for {path}")
    return writer


def write_clip(video_path, clip_path, max_frames, fps, frame_size):
    """Cut the first frames into a clip, since concept prompts have no frame range."""
    capture = cv2.VideoCapture(str(video_path))
    writer = open_writer(clip_path, fps, frame_size)
    written = 0
    while written < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    capture.release()
    writer.release()
    return written


def render_video(video_path, output_path, tracks, max_frames, fps, frame_size, use_clean_gray=False):
    by_frame = {}
    for track in tracks:
        by_frame.setdefault(track["frame_id"], []).append(track)

    capture = cv2.VideoCapture(str(video_path))
    writer = open_writer(output_path, fps, frame_size)
    frame_id = 0
    while frame_id < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        for track in by_frame.get(frame_id, []):
            x1, y1, x2, y2 = track["bbox"]
            color = (220, 220, 220) if use_clean_gray else COLORS[track["label"]]
            text = f'{track["label"]} #{track["object_id"]}'
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(y1, text_h + baseline + 4)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, top - text_h - baseline - 4), (x1 + text_w + 4, top),
                          color, cv2.FILLED)
            cv2.putText(frame, text, (x1 + 2, top - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        writer.write(frame)
        frame_id += 1
    capture.release()
    writer.release()


def video_info(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return fps, frame_size, frame_count


def segment_video(path, frame_size, base_url, api_key, tag="", source="full"):
    """Upload a video, run every prompt on it, and return its detections.

    SAM's raw output is saved to RAW_DIR, so tracking can be rebuilt with --from-raw.
    """
    RAW_DIR.mkdir(exist_ok=True)
    file_id = upload_video(path, base_url, api_key)
    detections = []
    try:
        for label in PROMPTS:
            lines = segment(file_id, label, base_url, api_key)
            (RAW_DIR / f"{source}_{label}.txt").write_text("\n".join(lines), encoding="utf-8")
            found = parse_lines(lines, label, frame_size, source)
            print(f"  {tag}{label}: {len(found)} detections")
            detections.extend(found)
    finally:
        delete_file(file_id, base_url, api_key)
    return detections


def tile_layout(frame_size, tiles):
    """Overlapping bands along the long side, in video.mp4 pixels, as (x0, y0, x1, y1)."""
    width, height = frame_size
    portrait = height >= width
    length = height if portrait else width
    overlap = length * TILE_OVERLAP
    bands = []
    for index in range(tiles):
        start = max(0, int(index * length / tiles - overlap / 2)) // 2 * 2
        end = min(length, -(-int((index + 1) * length / tiles + overlap / 2) // 2) * 2)
        bands.append((0, start, width, end) if portrait else (start, 0, end, height))
    return bands


def to_frame_coordinates(detections, band, zoom, tile_size, inner_edges):
    """Map tile boxes into video.mp4 pixels, dropping boxes cut off at an inner seam.

    The bands overlap, so a road user cut off at one band's seam is whole in the other.
    """
    x0, y0 = band[0], band[1]
    kept = []
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        if (("top" in inner_edges and y1 <= EDGE_MARGIN)
                or ("bottom" in inner_edges and y2 >= tile_size[1] - EDGE_MARGIN)
                or ("left" in inner_edges and x1 <= EDGE_MARGIN)
                or ("right" in inner_edges and x2 >= tile_size[0] - EDGE_MARGIN)):
            continue
        kept.append({**detection, "bbox": [round(x1 / zoom + x0), round(y1 / zoom + y0),
                                           round(x2 / zoom + x0), round(y2 / zoom + y0)]})
    return kept


def chunk_ranges(frames, chunk):
    """Frame ranges [start, end) of at most chunk frames. A last chunk shorter than SAM's
    50-frame minimum joins the one before, rather than being billed as 50."""
    if not chunk or chunk >= frames:
        return [(0, frames)]
    ranges = [(start, min(start + chunk, frames)) for start in range(0, frames, chunk)]
    if len(ranges) > 1 and ranges[-1][1] - ranges[-1][0] < SAM_MIN_BILLED_FRAMES:
        ranges[-2:] = [(ranges[-2][0], ranges[-1][1])]
    return ranges


def plan_parts(frame_size, frames, tiles, chunk):
    """Every (time chunk, tile) SAM has to see, each with its own 16-object cap."""
    portrait = frame_size[1] >= frame_size[0]
    bands = tile_layout(frame_size, tiles) if tiles > 1 else [(0, 0, *frame_size)]
    parts = []
    ranges = chunk_ranges(frames, chunk)
    for chunk_index, (start, end) in enumerate(ranges):
        if chunk_index < len(ranges) - 1:
            end = min(end + CHUNK_OVERLAP, frames)
        for tile_index, band in enumerate(bands):
            inner = set()
            if tile_index > 0:
                inner.add("top" if portrait else "left")
            if tile_index < len(bands) - 1:
                inner.add("bottom" if portrait else "right")
            parts.append({"source": f"c{chunk_index + 1}t{tile_index + 1}", "band": list(band),
                          "frames": [start, end], "inner": sorted(inner), "tiled": tiles > 1})
    return parts


def cut_part(source, part, frame_size, fps, output):
    """Cut one part from the source: its band, at up to TILE_ZOOM times video.mp4's
    resolution when tiled, and its frames.

    Uses the same fps filter as the reduced video before trimming, so frame n of the part
    is frame start + n of video.mp4. Returns the zoom and the part's size.
    """
    capture = cv2.VideoCapture(str(source))
    source_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    scale = source_size[0] / frame_size[0]  # source pixels per video.mp4 pixel
    zoom = min(TILE_ZOOM if part["tiled"] else 1, scale)  # never enlarge past the source
    x0, y0, x1, y1 = part["band"]
    start, end = part["frames"]
    crop = (f"crop={round((x1 - x0) * scale) // 2 * 2}:{round((y1 - y0) * scale) // 2 * 2}:"
            f"{round(x0 * scale)}:{round(y0 * scale)}")
    size = (round((x1 - x0) * zoom) // 2 * 2, round((y1 - y0) * zoom) // 2 * 2)
    filters = (f"{crop},scale={size[0]}:{size[1]}:flags=lanczos,fps={fps:g},"
               f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS")
    # A constant output rate: without it the trimmed file's last frame doesn't decode, so
    # every chunk reached SAM one frame short.
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-vf", filters,
         "-fps_mode", "cfr", "-r", f"{fps:g}",
         "-c:v", "libx264", "-crf", str(TILE_CRF), "-preset", "slow", "-pix_fmt", "yuv420p",
         "-an", str(output)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f'Could not cut part {part["source"]}: {result.stderr[-300:]}')
    return zoom, size


def to_video_frames(detections, part):
    """Map a part's detections into video.mp4: its pixels, and its frame numbers."""
    mapped = to_frame_coordinates(detections, part["band"], part["zoom"], tuple(part["size"]),
                                  set(part["inner"]))
    offset = part["frames"][0]
    return [{**detection, "frame_id": detection["frame_id"] + offset} for detection in mapped]


def track_parts(source, frame_size, fps, parts, base_url, api_key):
    """Run SAM on every part separately. Returns the detections and the parts, now with
    each part's zoom and size, for meta.json."""
    detections = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for index, part in enumerate(parts, start=1):
            path = Path(temp_dir) / f'{part["source"]}.mp4'
            part["zoom"], part["size"] = cut_part(source, part, frame_size, fps, path)
            start, end = part["frames"]
            print(f'Part {index} of {len(parts)} ({part["source"]}): frames {start}-{end - 1}, '
                  f'video.mp4 pixels {tuple(part["band"])}, {part["size"][0]}x{part["size"][1]}, '
                  f"{path.stat().st_size / 1_000_000:.1f} MB. Cost: ${estimate_cost(end - start):.2f}")
            found = segment_video(path, part["size"], base_url, api_key,
                                  f'{part["source"]} ', part["source"])
            mapped = to_video_frames(found, part)
            if part["inner"]:
                print(f'  {part["source"]}: {len(mapped)} of {len(found)} detections kept after '
                      "dropping boxes cut off at the seam")
            detections.extend(mapped)
    return detections, parts


def detections_from_raw():
    """Rebuild detections from saved SAM output, with no API call."""
    meta = json.loads((RAW_DIR / "meta.json").read_text(encoding="utf-8"))
    detections = []
    for part in meta["parts"]:
        part.setdefault("frames", [0, meta["frames"]])
        for label in PROMPTS:
            path = RAW_DIR / f'{part["source"]}_{label}.txt'
            if not path.exists():
                continue
            found = parse_lines(path.read_text(encoding="utf-8").splitlines(), label,
                                tuple(part["size"]), part["source"])
            detections.extend(to_video_frames(found, part))
    return detections, meta["frames"]


def finish(video_path, detections, frames, fps, frame_size):
    tracks = assign_track_ids(detections)
    Path(OUTPUT_JSON).write_text(json.dumps(tracks, indent=2), encoding="utf-8")
    render_video(video_path, OUTPUT_VIDEO, tracks, frames, fps, frame_size)
    object_count = len({track["object_id"] for track in tracks})
    print(f"Wrote {OUTPUT_JSON} ({len(tracks)} rows, {object_count} objects) and {OUTPUT_VIDEO}")


def run(video_path, max_frames, base_url, api_key, tiles=1, source=None, chunk=None):
    fps, frame_size, frame_count = video_info(video_path)
    frames = min(frame_count, max_frames) if max_frames else frame_count
    parts = plan_parts(frame_size, frames, tiles, chunk)
    cost = sum(estimate_cost(end - start) for start, end in {tuple(p["frames"]) for p in parts}) * tiles
    print(f"{video_path}: {frame_count} frames, {frame_size[0]}x{frame_size[1]}, {fps:g} fps")
    print(f"Processing {frames} frames with {len(PROMPTS)} prompts in {len(parts)} "
          f"part{'s' if len(parts) > 1 else ''}. Estimated cost: ${cost:.2f}")

    if len(parts) > 1:
        detections, parts = track_parts(source or video_path, frame_size, fps, parts,
                                        base_url, api_key)
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_path = video_path
            if max_frames:
                upload_path = Path(temp_dir) / "clip.mp4"
                frames = write_clip(video_path, upload_path, max_frames, fps, frame_size)
            detections = segment_video(upload_path, frame_size, base_url, api_key)
        parts = [{"source": "full", "size": list(frame_size), "band": [0, 0, *frame_size],
                  "frames": [0, frames], "zoom": 1, "inner": [], "tiled": False}]
    (RAW_DIR / "meta.json").write_text(json.dumps({"frames": frames, "parts": parts}, indent=2),
                                       encoding="utf-8")
    finish(video_path, detections, frames, fps, frame_size)


def main():
    parser = argparse.ArgumentParser(description="Track road users with SAM 3.1.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", action="store_true",
                      help=f"process the first {TEST_FRAMES} frames")
    mode.add_argument("--full", action="store_true", help="process the full video")
    mode.add_argument("--from-raw", action="store_true",
                      help=f"rebuild tracks from SAM output saved in {RAW_DIR}/, with no API call")
    parser.add_argument("--video", type=Path, default=Path("video.mp4"), help="input video")
    parser.add_argument("--chunk", type=int, default=CHUNK_FRAMES,
                        help="send the video to SAM in chunks of this many frames, so it picks "
                             "up to 16 new objects per prompt in each (0 = one piece)")
    parser.add_argument("--tiles", type=int, default=1,
                        help="track the frame in this many overlapping bands, each with its "
                             "own 16-object cap (costs this many times more)")
    parser.add_argument("--source", type=Path,
                        help="higher-resolution video to cut tiles from "
                             "(default: original.* here, else --video)")
    args = parser.parse_args()
    if args.tiles < 1:
        parser.error("--tiles must be at least 1")
    source = args.source or next(iter(sorted(Path(".").glob("original.*"))), args.video)

    if args.from_raw:
        if not (RAW_DIR / "meta.json").exists():
            sys.exit(f"No saved SAM output in {RAW_DIR}/. Run with --full first.")
        fps, frame_size, _ = video_info(args.video)
        detections, frames = detections_from_raw()
        finish(args.video, detections, frames, fps, frame_size)
        return

    load_dotenv()
    api_key = os.getenv("META_API_KEY")
    base_url = (os.getenv("META_API_BASE") or "").rstrip("/")
    if not api_key or not base_url:
        sys.exit("Set META_API_KEY and META_API_BASE in .env")

    try:
        run(args.video, TEST_FRAMES if args.test else None, base_url, api_key, args.tiles, source,
            args.chunk)
    except RuntimeError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
