"""Pieces shared by the pipeline scripts and the dashboard.

Standard library only, so app.py can import this without the pipeline's
dependencies (OpenCV, requests, python-dotenv).
"""

import json
from collections import Counter
from pathlib import Path

# sam_track.py: what SAM 3.1 looks for, and its price
PROMPTS = ["pedestrian", "auto-rickshaw", "motorcycle", "car", "bus", "bicycle", "scooter"]
SAM_USD_PER_FRAME = 0.20 / 1000
SAM_MIN_BILLED_FRAMES = 50  # per request, and each prompt is its own request

# detect.py: track filtering and conflict thresholds
MIN_TRACK_FRAMES = 5
CONTAINED_OVERLAP = 0.8  # share of a pedestrian box that must lie inside a vehicle box
CONTAINED_FRACTION = 0.5  # share of its frames spent that way before a track is dropped
MAX_DISTANCE = 80  # pixels between bbox centers
MAX_TTC = 2.5  # seconds

# verify.py: the vision model and its prices
VISION_MODEL = "muse-spark-1.3"
USD_PER_INPUT_TOKEN = 1.25 / 1_000_000
USD_PER_OUTPUT_TOKEN = 4.25 / 1_000_000

# generate_88_trackings.py: how far back each target's trail reaches
TRAIL_FRAMES = 10

# hotspot.py: risk zones
ZONE_NAMES = ["Left third", "Centre third", "Right third"]
MEDIUM_SHARE = 0.25  # a zone with this share of all potential conflicts is worth monitoring


# A person's corrections to model verdicts, kept apart from the model's own log.
REVIEW_JSON = "human_review.json"
TWO_WHEELERS = {"motorcycle", "scooter", "bicycle"}


def load_reviews(folder):
    """{conflict_id: review} from human_review.json in folder, or {} if there is none."""
    try:
        reviews = json.loads((Path(folder) / REVIEW_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {int(conflict_id): review for conflict_id, review in reviews.items()}


def apply_reviews(records, reviews):
    """Model verdicts with a person's corrections laid over them, marked as reviewed.
    The records themselves, and so the model's own verdicts, are left unchanged."""
    reviewed = []
    for record in records:
        review = reviews.get(record["conflict_id"])
        if review and record.get("verdict"):
            record = {**record, "verdict": {**record["verdict"], **review["verdict"]},
                      "review": review}
        reviewed.append(record)
    return reviewed


def build_verified(conflicts, records):
    """The real near-misses, worst first, with each conflict's details and its verdict."""
    by_id = {conflict["conflict_id"]: conflict for conflict in conflicts}
    verified = [
        {**by_id[record["conflict_id"]], "clip": record["clip"], **record["verdict"],
         **({"reviewed_by": record["review"].get("reviewer", "person")} if "review" in record else {})}
        for record in records
        if record.get("verdict") and record["verdict"]["is_real"] and record["conflict_id"] in by_id
    ]
    verified.sort(key=lambda row: (-row["severity"], row["conflict_id"]))
    return verified


def road_user_class(label):
    return "two-wheeler" if label in TWO_WHEELERS else label


def conflict_type(conflict):
    """'pedestrian / two-wheeler', for counting conflicts by who was involved."""
    return " / ".join(sorted((road_user_class(conflict["label_1"]),
                              road_user_class(conflict["label_2"]))))


def sam_cost(frames):
    """Dollars to track a clip of this many frames with every prompt."""
    return max(frames, SAM_MIN_BILLED_FRAMES) * len(PROMPTS) * SAM_USD_PER_FRAME


def load_tracks(path):
    """Group rows by object_id, dropping tracks too short to be real."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["object_id"], []).append(row)

    tracks = {}
    for object_id, track_rows in grouped.items():
        if len(track_rows) < MIN_TRACK_FRAMES:
            continue
        labels = Counter(row["label"] for row in track_rows)
        tracks[object_id] = {
            "label": labels.most_common(1)[0][0],
            "boxes": {row["frame_id"]: row["bbox"] for row in track_rows},
            "centers": {
                row["frame_id"]: ((row["bbox"][0] + row["bbox"][2]) / 2,
                                  (row["bbox"][1] + row["bbox"][3]) / 2)
                for row in track_rows
            },
        }
    return tracks, len(grouped) - len(tracks)


def contained_share(inner, outer):
    """How much of the inner box lies inside the outer box, from 0 to 1."""
    width = min(inner[2], outer[2]) - max(inner[0], outer[0])
    height = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return max(0, width) * max(0, height) / area if area else 0.0


def drop_occupants(tracks):
    """Drop pedestrians who stay inside a vehicle box: drivers, riders and passengers.

    Judged over the whole track, because a person walking past a vehicle also
    sits inside its box briefly when filmed from above.
    """
    vehicle_boxes = {}
    for track in tracks.values():
        if track["label"] != "pedestrian":
            for frame_id, box in track["boxes"].items():
                vehicle_boxes.setdefault(frame_id, []).append(box)

    kept = {}
    for object_id, track in tracks.items():
        if track["label"] == "pedestrian":
            inside = sum(
                any(contained_share(box, vehicle) >= CONTAINED_OVERLAP
                    for vehicle in vehicle_boxes.get(frame_id, []))
                for frame_id, box in track["boxes"].items()
            )
            if inside / len(track["boxes"]) >= CONTAINED_FRACTION:
                continue
        kept[object_id] = track
    return kept, len(tracks) - len(kept)


def spent(records):
    """Dollars used by verify.py so far, from the tokens the API reported."""
    total = 0.0
    for record in records:
        for attempt in record["attempts"]:
            usage = attempt["response"].get("usage") or {}
            total += usage.get("prompt_tokens", 0) * USD_PER_INPUT_TOKEN
            total += usage.get("completion_tokens", 0) * USD_PER_OUTPUT_TOKEN
    return total


def conflict_points(conflicts, centers):
    """Midpoint of each conflict's two objects, as {conflict_id: (x, y)}."""
    points = {}
    for conflict in conflicts:
        frame_id = (conflict["frame_start"] + conflict["frame_end"]) // 2
        pair = []
        for key in ("object_id_1", "object_id_2"):
            track = centers.get(conflict[key])
            if track:
                pair.append(track[min(track, key=lambda f: abs(f - frame_id))])
        if len(pair) == 2:
            points[conflict["conflict_id"]] = ((pair[0][0] + pair[1][0]) / 2,
                                               (pair[0][1] + pair[1][1]) / 2)
    return points


def zone_summary(conflicts, verified, centers, frame_size):
    """Split the frame into thirds and count potential and verified conflicts in each.

    A zone is high risk if it holds a verified near-miss, medium if it holds at least
    MEDIUM_SHARE of the potential conflicts, and low otherwise.
    """
    points = conflict_points(conflicts, centers)
    real = {row["conflict_id"]: row for row in verified}
    zones = [{"name": name, "potential": 0, "real": []} for name in ZONE_NAMES]
    for conflict in conflicts:
        point = points.get(conflict["conflict_id"])
        if point is None:
            continue
        zone = zones[min(2, int(3 * point[0] / frame_size[0]))]
        zone["potential"] += 1
        if conflict["conflict_id"] in real:
            zone["real"].append({**real[conflict["conflict_id"]], "point": point})

    total = sum(zone["potential"] for zone in zones)
    for zone in zones:
        if zone["real"]:
            zone["level"] = "high"
        elif total and zone["potential"] >= MEDIUM_SHARE * total:
            zone["level"] = "medium"
        else:
            zone["level"] = "low"
    return zones
