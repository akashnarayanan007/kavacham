"""Render one tracked video per conflict, showing all tracked road users with
the conflicting pair highlighted.

Usage:
    python generate_88_trackings.py [folder]

Reads potential_conflicts.json, tracks.json, video.mp4 and, if present,
llama_raw_responses.json for each conflict's risk type.
Writes clips/individual/conflict_<id>_tracked.mp4.
"""

import json
import math
import sys
from pathlib import Path

import cv2

from common import TRAIL_FRAMES, apply_reviews, load_reviews
from generate_marked import COLOR_1, COLOR_2, FONT, THICKNESS, draw_label
from sam_track import open_writer, video_info

CONFLICTS_JSON = Path("potential_conflicts.json")
TRACKS_JSON = Path("tracks.json")
RAW_JSON = Path("llama_raw_responses.json")
VIDEO = Path("video.mp4")
INDIVIDUAL_DIR = Path("clips") / "individual"

PAD_FRAMES = 15  # 1.5 s either side at 10 fps
MAX_INTERPOLATION_GAP = 5  # frames to the nearest detection on each side
TRAIL_RADIUS = 3
OTHER_COLOR = (210, 210, 210)  # light gray/white for background road users
OTHER_TRAIL_COLOR = (180, 180, 180)
BAR_HEIGHT = 34
BAR_FONT_SCALE = 0.6


def load_tracks_data(path):
    """Load both by-object and by-frame tracks from tracks.json."""
    by_object = {}
    by_frame = {}
    for row in json.loads(Path(path).read_text(encoding="utf-8")):
        oid = row["object_id"]
        fid = row["frame_id"]
        bbox = row["bbox"]
        label = row.get("label", "user")
        by_object.setdefault(oid, {})[fid] = bbox
        by_frame.setdefault(fid, []).append((oid, bbox, label))
    return by_object, by_frame


def load_verdicts(path, folder="."):
    """Each conflict's final verdict, with any human review applied, if verify.py has run."""
    try:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    records = apply_reviews(records, load_reviews(folder))
    return {record["conflict_id"]: {**record["verdict"], "reviewed": "review" in record}
            for record in records if record.get("verdict")}


def draw_gap(frame, box_a, box_b):
    """A line between the pair's centres, labelled with the gap between their boxes, so a
    reviewer can see how close they came."""
    centre_a = ((box_a[0] + box_a[2]) // 2, (box_a[1] + box_a[3]) // 2)
    centre_b = ((box_b[0] + box_b[2]) // 2, (box_b[1] + box_b[3]) // 2)
    dx = max(0, box_a[0] - box_b[2], box_b[0] - box_a[2])
    dy = max(0, box_a[1] - box_b[3], box_b[1] - box_a[3])
    edge = math.hypot(dx, dy)
    centres = math.dist(centre_a, centre_b)
    text = f"gap {edge:.0f}px" if edge else "boxes overlap"
    cv2.line(frame, centre_a, centre_b, (255, 255, 255), 1, cv2.LINE_AA)
    middle = ((centre_a[0] + centre_b[0]) // 2 + 6, (centre_a[1] + centre_b[1]) // 2)
    label = f"{text}, centres {centres:.0f}px"
    (width, height), _ = cv2.getTextSize(label, FONT, 0.45, 1)
    cv2.rectangle(frame, (middle[0] - 2, middle[1] - height - 4), (middle[0] + width + 2, middle[1] + 3),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, label, middle, FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def box_at(track, frame_id):
    """The tracked box, or one interpolated across a short detection drop. None if neither."""
    if frame_id in track:
        return track[frame_id]
    before = [f for f in track if frame_id - MAX_INTERPOLATION_GAP <= f < frame_id]
    after = [f for f in track if frame_id < f <= frame_id + MAX_INTERPOLATION_GAP]
    if not before or not after:
        return None
    start, end = max(before), min(after)
    share = (frame_id - start) / (end - start)
    return [round(a + (b - a) * share) for a, b in zip(track[start], track[end])]


def draw_bar(frame, text):
    """White summary text on a black bar along the bottom, shrunk to fit if needed."""
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, height - BAR_HEIGHT), (width, height), (0, 0, 0), cv2.FILLED)
    scale = BAR_FONT_SCALE
    while scale > 0.3 and cv2.getTextSize(text, FONT, scale, 1)[0][0] > width - 20:
        scale -= 0.05
    text_h = cv2.getTextSize(text, FONT, scale, 1)[0][1]
    cv2.putText(frame, text, (10, height - (BAR_HEIGHT - text_h) // 2), FONT, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def render_conflict(conflict, tracks_by_object, tracks_by_frame, verdict, video_path, out_dir, frame_count, fps, frame_size):
    """Write one conflict's tracked video showing all road users with the conflicting pair highlighted.

    Returns its file name.
    """
    start = max(0, conflict["frame_start"] - PAD_FRAMES)
    end = min(frame_count - 1, conflict["frame_end"] + PAD_FRAMES)
    if verdict:
        colors = (COLOR_1, COLOR_1) if verdict["is_real"] else (COLOR_2, COLOR_2)
        outcome = "REAL near-miss" if verdict["is_real"] else "normal"
        outcome += ", human reviewed" if verdict["reviewed"] else f', {verdict["risk_type"]}'
    else:
        colors, outcome = (COLOR_1, COLOR_2), "not verified"

    target_ids = {conflict["object_id_1"], conflict["object_id_2"]}
    targets = [
        (tracks_by_object.get(conflict["object_id_1"], {}), colors[0],
         f'TARGET 1: {conflict["label_1"]} #{conflict["object_id_1"]}'),
        (tracks_by_object.get(conflict["object_id_2"], {}), colors[1],
         f'TARGET 2: {conflict["label_2"]} #{conflict["object_id_2"]}'),
    ]
    summary = (f'Conflict {conflict["conflict_id"]}: {conflict["label_1"]} #{conflict["object_id_1"]} / '
               f'{conflict["label_2"]} #{conflict["object_id_2"]} | TTC {conflict["min_ttc"]}s | '
               f'Dist {conflict["min_distance"]}px | {outcome}')

    name = f'conflict_{conflict["conflict_id"]}_tracked.mp4'
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = open_writer(out_dir / name, fps, frame_size)

    # Trails dictionary mapping object_id -> list of (frame_id, (center_x, center_y))
    trails = {}

    for frame_id in range(start, end + 1):
        ok, frame = capture.read()
        if not ok:
            break

        # 1. Update and draw all other detected road users (light gray/white)
        frame_objects = tracks_by_frame.get(frame_id, [])
        for oid, box, _ in frame_objects:
            if oid not in target_ids:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), OTHER_COLOR, 1)
                cx = (box[0] + box[2]) // 2
                cy = (box[1] + box[3]) // 2
                trails.setdefault(oid, []).append((frame_id, (cx, cy)))

        # 2. Update pair centers
        pair_boxes = [box_at(t, frame_id) for t, _, _ in targets]
        for tid, box in zip([conflict["object_id_1"], conflict["object_id_2"]], pair_boxes):
            if box:
                cx = (box[0] + box[2]) // 2
                cy = (box[1] + box[3]) // 2
                trails.setdefault(tid, []).append((frame_id, (cx, cy)))

        # 3. Prune and draw trails for all objects (last TRAIL_FRAMES frames)
        for oid, pts in list(trails.items()):
            valid_pts = [(f, pt) for f, pt in pts if frame_id - f < TRAIL_FRAMES]
            trails[oid] = valid_pts
            if oid == conflict["object_id_1"]:
                trail_color = colors[0]
                radius = TRAIL_RADIUS
            elif oid == conflict["object_id_2"]:
                trail_color = colors[1]
                radius = TRAIL_RADIUS
            else:
                trail_color = OTHER_TRAIL_COLOR
                radius = 1

            for _, center in valid_pts:
                cv2.circle(frame, center, radius, trail_color, cv2.FILLED)

        # 4. Draw pair highlighted boxes thick
        for box, (_, color, _) in zip(pair_boxes, targets):
            if box:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, THICKNESS)

        # 5. Gap readout line between conflicting pair
        if all(pair_boxes):
            draw_gap(frame, *pair_boxes)

        # 6. Pair labels
        taken = []
        for box, (_, color, label) in zip(pair_boxes, targets):
            if box:
                taken.append(draw_label(frame, label, box, color, taken))

        draw_bar(frame, summary)
        writer.write(frame)

    capture.release()
    writer.release()
    return name


def run_generation(base_dir=Path(".")):
    base = Path(base_dir)
    conflicts_p = base / CONFLICTS_JSON
    tracks_p = base / TRACKS_JSON
    raw_p = base / RAW_JSON
    video_p = base / VIDEO
    out_dir = base / INDIVIDUAL_DIR

    try:
        conflicts = json.loads(conflicts_p.read_text(encoding="utf-8"))
        tracks_by_object, tracks_by_frame = load_tracks_data(tracks_p)
        fps, frame_size, frame_count = video_info(video_p)
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        sys.exit(f"Could not load input in {base}: {error}")
    verdicts = load_verdicts(raw_p, base)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Tells the dashboard what the colours mean in these videos.
    legend = {"mode": "verdict" if verdicts else "target"}
    (out_dir / "legend.json").write_text(json.dumps(legend), encoding="utf-8")

    for index, conflict in enumerate(conflicts, start=1):
        verdict = verdicts.get(conflict["conflict_id"])
        name = render_conflict(conflict, tracks_by_object, tracks_by_frame, verdict,
                               video_p, out_dir, frame_count, fps, frame_size)
        if index % 20 == 0 or index == len(conflicts):
            print(f'[{base.name or "demo"}] Generated {index}/{len(conflicts)}: {name}')

    print(f"Wrote {len(conflicts)} tracked videos to {out_dir}/")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    run_generation(folder)


if __name__ == "__main__":
    main()

