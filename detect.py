"""Find potential conflicts between road users tracked by sam_track.py.

Usage:
    python detect.py           # every conflict
    python detect.py --test    # first 20 conflicts

Reads tracks.json and video.mp4.
Writes potential_conflicts.json and clips/conflict_*.mp4.
"""

import argparse
import json
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import cv2

# Filtering lives in common.py so the dashboard can show it without OpenCV.
from common import MAX_DISTANCE, MAX_TTC, MIN_TRACK_FRAMES, drop_occupants, load_tracks
from sam_track import PROMPTS, open_writer, video_info

SMOOTHING_SECONDS = 0.5  # box jitter on parked vehicles otherwise reads as closing speed
MERGE_GAP_SECONDS = 0.5  # flagged frames this close count as one conflict
CLIP_SECONDS = 3
TEST_CONFLICTS = 20

OUTPUT_JSON = "potential_conflicts.json"
CLIPS_DIR = Path("clips")


def smooth_centers(centers, window):
    """Moving average over the frames within window of each frame."""
    smoothed = {}
    for frame_id in centers:
        nearby = [centers[f] for f in range(frame_id - window, frame_id + window + 1)
                  if f in centers]
        smoothed[frame_id] = (sum(c[0] for c in nearby) / len(nearby),
                              sum(c[1] for c in nearby) / len(nearby))
    return smoothed


def pair_metrics(track_a, track_b, fps):
    """Yield distance, closing speed and TTC for each frame two tracks share."""
    shared = sorted(track_a["centers"].keys() & track_b["centers"].keys())
    for previous, frame_id in zip(shared, shared[1:]):
        before = math.dist(track_a["smoothed"][previous], track_b["smoothed"][previous])
        after = math.dist(track_a["smoothed"][frame_id], track_b["smoothed"][frame_id])
        closing_speed = (before - after) * fps / (frame_id - previous)
        distance = math.dist(track_a["centers"][frame_id], track_b["centers"][frame_id])
        ttc = distance / closing_speed if closing_speed > 0 else math.inf
        yield {"frame_id": frame_id, "distance": distance,
               "closing_speed": closing_speed, "ttc": ttc}


def is_conflict(metric):
    return (metric["distance"] < MAX_DISTANCE
            and metric["closing_speed"] > 0
            and metric["ttc"] < MAX_TTC)


def group_episodes(flagged, max_gap):
    """Split flagged frames into runs, bridging gaps of up to max_gap frames."""
    episodes = []
    for metric in flagged:
        if episodes and metric["frame_id"] - episodes[-1][-1]["frame_id"] <= max_gap:
            episodes[-1].append(metric)
        else:
            episodes.append([metric])
    return episodes


def find_conflicts(tracks, fps):
    window = round(SMOOTHING_SECONDS * fps / 2)
    max_gap = round(MERGE_GAP_SECONDS * fps)
    for track in tracks.values():
        track["smoothed"] = smooth_centers(track["centers"], window)

    conflicts = []
    for (id_a, track_a), (id_b, track_b) in combinations(sorted(tracks.items()), 2):
        flagged = [m for m in pair_metrics(track_a, track_b, fps) if is_conflict(m)]
        for episode in group_episodes(flagged, max_gap):
            peak = min(episode, key=lambda m: m["ttc"])
            conflicts.append({
                "object_id_1": id_a,
                "label_1": track_a["label"],
                "object_id_2": id_b,
                "label_2": track_b["label"],
                "frame_start": episode[0]["frame_id"],
                "frame_end": episode[-1]["frame_id"],
                "min_distance": round(min(m["distance"] for m in episode), 2),
                "min_ttc": round(peak["ttc"], 3),
                "closing_speed": round(max(m["closing_speed"] for m in episode), 2),
                "peak_frame": peak["frame_id"],
            })

    conflicts.sort(key=lambda c: (c["frame_start"], c["frame_end"],
                                  c["object_id_1"], c["object_id_2"]))
    for conflict_id, conflict in enumerate(conflicts, start=1):
        conflict["conflict_id"] = conflict_id
    return conflicts


def clip_window(peak_frame, fps, frame_count):
    """A fixed-length window around peak_frame, shifted to stay inside the video."""
    length = round(CLIP_SECONDS * fps)
    start = max(0, min(peak_frame - length // 2, frame_count - length))
    return start, min(start + length, frame_count)


def cut_clips(video_path, conflicts, fps, frame_size, frame_count):
    """Write every clip in one pass, since seeking per clip is slow on long videos."""
    CLIPS_DIR.mkdir(exist_ok=True)
    # Marked and tracked videos belong to the old conflict IDs, so they go too.
    for stale in [*CLIPS_DIR.glob("conflict_*.mp4"), *CLIPS_DIR.glob("marked/conflict_*.mp4"),
                  *CLIPS_DIR.glob("individual/conflict_*_tracked.mp4")]:
        stale.unlink()

    starting = {}
    for conflict in conflicts:
        start, end = clip_window(conflict["peak_frame"], fps, frame_count)
        path = CLIPS_DIR / f'conflict_{conflict["conflict_id"]:03d}.mp4'
        starting.setdefault(start, []).append((path, end))
    last_frame = max((end for clips in starting.values() for _, end in clips), default=0)

    capture = cv2.VideoCapture(str(video_path))
    active = []  # (writer, end)
    try:
        for frame_id in range(last_frame):
            ok, frame = capture.read()
            if not ok:
                break
            for path, end in starting.get(frame_id, []):
                active.append((open_writer(path, fps, frame_size), end))
            for writer, _ in active:
                writer.write(frame)
            for writer, end in active:
                if frame_id + 1 >= end:
                    writer.release()
            active = [(writer, end) for writer, end in active if frame_id + 1 < end]
    finally:
        capture.release()
        for writer, _ in active:
            writer.release()


def print_summary(tracks, dropped, occupants, conflicts, total):
    print(f"Kept {len(tracks)} tracks, dropped {dropped} under {MIN_TRACK_FRAMES} frames "
          f"and {occupants} pedestrians inside vehicles")
    labels = Counter(track["label"] for track in tracks.values())
    for label in PROMPTS:
        print(f"  {label}: {labels[label]} tracks")
    print(f"Found {total} potential conflicts, wrote {len(conflicts)}")
    pairs = Counter(" / ".join(sorted((c["label_1"], c["label_2"]))) for c in conflicts)
    for pair, count in pairs.most_common():
        print(f"  {pair}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Find potential conflicts in tracks.json.")
    parser.add_argument("--test", action="store_true",
                        help=f"process the first {TEST_CONFLICTS} conflicts only")
    parser.add_argument("--tracks", type=Path, default=Path("tracks.json"), help="input tracks")
    parser.add_argument("--video", type=Path, default=Path("video.mp4"), help="source video")
    args = parser.parse_args()

    try:
        fps, frame_size, frame_count = video_info(args.video)
        tracks, dropped = load_tracks(args.tracks)
    except (RuntimeError, OSError, json.JSONDecodeError) as error:
        sys.exit(f"Could not load input: {error}")

    last_frame = max((f for t in tracks.values() for f in t["centers"]), default=0)
    if last_frame >= frame_count:
        sys.exit(f"{args.tracks} reaches frame {last_frame} but {args.video} has "
                 f"{frame_count} frames. Rerun sam_track.py on this video.")

    tracks, occupants = drop_occupants(tracks)
    conflicts = find_conflicts(tracks, fps)
    total = len(conflicts)
    if args.test:
        conflicts = conflicts[:TEST_CONFLICTS]

    cut_clips(args.video, conflicts, fps, frame_size, frame_count)
    for conflict in conflicts:
        conflict["clip_start"] = clip_window(conflict["peak_frame"], fps, frame_count)[0]

    fields = ["conflict_id", "object_id_1", "label_1", "object_id_2", "label_2", "frame_start",
              "frame_end", "min_distance", "min_ttc", "closing_speed", "clip_start"]
    output = [{field: conflict[field] for field in fields} for conflict in conflicts]
    Path(OUTPUT_JSON).write_text(json.dumps(output, indent=2), encoding="utf-8")

    print_summary(tracks, dropped, occupants, conflicts, total)
    print(f"Wrote {OUTPUT_JSON} and {len(conflicts)} clips to {CLIPS_DIR}/")


if __name__ == "__main__":
    main()
