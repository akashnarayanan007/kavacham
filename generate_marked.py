"""Draw each conflict's flagged pair on its clip, for review in the dashboard.

Usage:
    python generate_marked.py

Reads potential_conflicts.json, tracks.json and clips/conflict_*.mp4.
Writes clips/marked/conflict_*.mp4.
"""

import json
import sys
from pathlib import Path

import cv2

from sam_track import open_writer

CONFLICTS_JSON = Path("potential_conflicts.json")
TRACKS_JSON = Path("tracks.json")
CLIPS_DIR = Path("clips")
MARKED_DIR = CLIPS_DIR / "marked"

COLOR_1 = (0, 0, 220)  # red, in BGR
COLOR_2 = (0, 140, 0)  # green, dark enough for white text
OTHER_COLOR = (170, 170, 170)
OTHER_ALPHA = 0.35  # other road users stay visible as context without competing
THICKNESS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
DEFAULT_FPS = 10


def load_boxes(path):
    """Index track boxes as {frame_id: [(object_id, bbox)]}."""
    by_frame = {}
    for row in json.loads(Path(path).read_text(encoding="utf-8")):
        by_frame.setdefault(row["frame_id"], []).append((row["object_id"], row["bbox"]))
    return by_frame


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def draw_label(frame, text, box, color, taken):
    """White text on a solid tab above the box, moved down if it would hide another tab.

    Returns the tab's rectangle so later labels can avoid it.
    """
    x1, y1, _, y2 = box
    (text_w, text_h), baseline = cv2.getTextSize(text, FONT, FONT_SCALE, 1)
    tab_w, tab_h = text_w + 8, text_h + baseline + 8
    left = min(max(0, x1), frame.shape[1] - tab_w)
    top = y1 - tab_h if y1 - tab_h >= 0 else y2
    while any(overlaps((left, top, left + tab_w, top + tab_h), other) for other in taken):
        top += tab_h
    cv2.rectangle(frame, (left, top), (left + tab_w, top + tab_h), color, cv2.FILLED)
    cv2.putText(frame, text, (left + 4, top + text_h + 4), FONT, FONT_SCALE,
                (255, 255, 255), 1, cv2.LINE_AA)
    return (left, top, left + tab_w, top + tab_h)


def mark_frame(frame, objects, conflict):
    targets = {
        conflict["object_id_1"]: (COLOR_1, f'TARGET 1: {conflict["label_1"]} #{conflict["object_id_1"]}'),
        conflict["object_id_2"]: (COLOR_2, f'TARGET 2: {conflict["label_2"]} #{conflict["object_id_2"]}'),
    }

    overlay = frame.copy()
    for object_id, (x1, y1, x2, y2) in objects:
        if object_id not in targets:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), OTHER_COLOR, 1)
    cv2.addWeighted(overlay, OTHER_ALPHA, frame, 1 - OTHER_ALPHA, 0, dst=frame)

    # Only frames where the tracker saw a target get its box: no guessed positions.
    seen = {object_id: box for object_id, box in objects if object_id in targets}
    for object_id, box in seen.items():
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), targets[object_id][0], THICKNESS)
    taken = []
    for object_id in (conflict["object_id_1"], conflict["object_id_2"]):
        if object_id in seen:
            color, label = targets[object_id]
            taken.append(draw_label(frame, label, seen[object_id], color, taken))


def mark_clip(conflict, by_frame):
    """Write the marked clip. Returns False if the source clip is missing."""
    name = f'conflict_{conflict["conflict_id"]:03d}.mp4'
    source = CLIPS_DIR / name
    if not source.exists():
        return False

    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = open_writer(MARKED_DIR / name, fps, frame_size)
    clip_frame = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        objects = by_frame.get(conflict["clip_start"] + clip_frame, [])
        mark_frame(frame, objects, conflict)
        writer.write(frame)
        clip_frame += 1
    capture.release()
    writer.release()
    return True


def main():
    try:
        conflicts = json.loads(CONFLICTS_JSON.read_text(encoding="utf-8"))
        by_frame = load_boxes(TRACKS_JSON)
    except (OSError, json.JSONDecodeError) as error:
        sys.exit(f"Could not load input: {error}")
    if any("clip_start" not in conflict for conflict in conflicts):
        sys.exit(f"{CONFLICTS_JSON} has no clip_start. Rerun detect.py.")

    MARKED_DIR.mkdir(parents=True, exist_ok=True)
    for stale in MARKED_DIR.glob("conflict_*.mp4"):
        stale.unlink()

    missing = []
    for index, conflict in enumerate(conflicts, start=1):
        if mark_clip(conflict, by_frame):
            print(f"Marked {index}/{len(conflicts)}")
        else:
            missing.append(conflict["conflict_id"])
            print(f"Skipped {index}/{len(conflicts)}: no clip for conflict {conflict['conflict_id']}")

    print(f"Wrote {len(conflicts) - len(missing)} marked clips to {MARKED_DIR}/")
    if missing:
        print(f"Missing clips for conflicts {missing}. Rerun detect.py.")


if __name__ == "__main__":
    main()
