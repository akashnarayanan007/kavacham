"""Build the hotspot map: risk zones from where conflicts happened, as an overhead illustration.

Usage:
    python hotspot.py

Reads potential_conflicts.json, verified_conflicts.json, tracks.json and video.mp4.
Writes hotspot.png (a Muse Image illustration, or the camera-frame map if that fails),
hotspot_frame.png (the zones drawn on a real video frame) and hotspot.json (how
hotspot.png was made).
"""

import base64
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import requests
from dotenv import load_dotenv

from common import drop_occupants, load_tracks, zone_summary
from sam_track import TIMEOUT, check_status, video_info, with_retries

IMAGE_MODEL = "muse-image-1.0"
USD_PER_IMAGE = 0.01

VIDEO = Path("video.mp4")
TRACKS_JSON = Path("tracks.json")
CONFLICTS_JSON = Path("potential_conflicts.json")
VERIFIED_JSON = Path("verified_conflicts.json")
HOTSPOT_PNG = Path("hotspot.png")
FRAME_PNG = Path("hotspot_frame.png")
META_JSON = Path("hotspot.json")

LEVEL_LABELS = {"high": "HIGH RISK", "medium": "MEDIUM", "low": "LOW"}
LEVEL_COLORS = {"high": (40, 40, 198), "medium": (0, 170, 235), "low": (60, 150, 60)}  # BGR
OVERLAY_ALPHA = 0.25
FONT = cv2.FONT_HERSHEY_SIMPLEX


def build_prompt(zones):
    """Describe the zones for Muse Image. It only sees words, not our coordinates."""
    marks = []
    for zone in zones:
        place = zone["name"].replace(" third", "").lower()
        if zone["level"] == "high":
            worst = min(zone["real"], key=lambda row: row["min_ttc"])
            marks.append(
                f"mark the {place} of the junction with a red circle labelled 'HIGH RISK: "
                f'{len(zone["real"])} near-miss {worst["label_2"]}+{worst["label_1"]} TTC '
                f'{worst["min_ttc"]:.2f}s\'')
        elif zone["level"] == "medium":
            marks.append(f"shade the {place} of the junction yellow")
        else:
            marks.append(f"shade the {place} of the junction green")
    return ("Overhead satellite-style illustration of a busy Indian urban junction where roads "
            "meet, with autos, motorcycles, cars and pedestrians seen from directly above. The "
            "left, centre and right of the image match the left, centre and right of a camera "
            f'view of the junction. {"; ".join(marks)}. Clear daylight, realistic style, no '
            "text except the risk labels.")


def generate_image(prompt, base_url, api_key):
    """Ask Muse Image for the illustration. Returns PNG bytes and the response without them."""
    def call():
        response = requests.post(
            f"{base_url}/images/generations",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": IMAGE_MODEL,
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024",
                "output_format": "png",
                # No search: it must not pull in imagery of some real junction.
                "tool_enablement": {"enable_image_search": False, "enable_web_search": False,
                                    "enable_shell": False},
            },
            timeout=TIMEOUT,
        )
        check_status(response, "Image generation")
        return response.json()

    data = with_retries(call, "Image generation")
    image = base64.b64decode(data["data"][0]["b64_json"])
    log = {key: value for key, value in data.items() if key != "data"}
    return image, log


def draw_frame_map(video_path, zones, frame_size):
    """Shade each third of a real video frame by risk level and ring the near-misses."""
    capture = cv2.VideoCapture(str(video_path))
    real = [row for zone in zones for row in zone["real"]]
    frame_id = real[0]["frame_start"] if real else int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_id} from {video_path}")

    width, height = frame_size
    overlay = frame.copy()
    for index, zone in enumerate(zones):
        left, right = index * width // 3, (index + 1) * width // 3
        cv2.rectangle(overlay, (left, 0), (right, height), LEVEL_COLORS[zone["level"]], cv2.FILLED)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, dst=frame)

    for index, zone in enumerate(zones):
        left = index * width // 3
        cv2.line(frame, (left, 0), (left, height), (255, 255, 255), 1)
        text = f'{LEVEL_LABELS[zone["level"]]}: {zone["potential"]} potential, {len(zone["real"])} real'
        (text_w, text_h), _ = cv2.getTextSize(text, FONT, 0.6, 1)
        cv2.rectangle(frame, (left + 8, 8), (left + text_w + 20, text_h + 24),
                      LEVEL_COLORS[zone["level"]], cv2.FILLED)
        cv2.putText(frame, text, (left + 14, text_h + 16), FONT, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
    for row in real:
        x, y = (round(v) for v in row["point"])
        cv2.circle(frame, (x, y), 34, (255, 255, 255), 5)
        cv2.circle(frame, (x, y), 34, LEVEL_COLORS["high"], 3)
        cv2.putText(frame, f'#{row["conflict_id"]}', (x + 40, y + 6), FONT, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    load_dotenv()
    try:
        conflicts = json.loads(CONFLICTS_JSON.read_text(encoding="utf-8"))
        verified = json.loads(VERIFIED_JSON.read_text(encoding="utf-8"))
        tracks, _ = load_tracks(TRACKS_JSON)
        tracks, _ = drop_occupants(tracks)
        fps, frame_size, _ = video_info(VIDEO)
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        sys.exit(f"Could not load input: {error}")

    centers = {object_id: track["centers"] for object_id, track in tracks.items()}
    zones = zone_summary(conflicts, verified, centers, frame_size)
    for zone in zones:
        print(f'{zone["name"]}: {LEVEL_LABELS[zone["level"]]}, {zone["potential"]} potential, '
              f'{len(zone["real"])} real')

    cv2.imwrite(str(FRAME_PNG), draw_frame_map(VIDEO, zones, frame_size))
    print(f"Wrote {FRAME_PNG}")

    prompt = build_prompt(zones)
    meta = {"model": IMAGE_MODEL, "prompt": prompt, "cost_usd": 0.0,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    api_key = os.getenv("META_API_KEY")
    base_url = (os.getenv("META_API_BASE") or "").rstrip("/")
    try:
        if not api_key or not base_url:
            raise RuntimeError("META_API_KEY or META_API_BASE is not set")
        image, log = generate_image(prompt, base_url, api_key)
        HOTSPOT_PNG.write_bytes(image)
        meta.update(source="muse-image", cost_usd=USD_PER_IMAGE, response=log)
        print(f"Wrote {HOTSPOT_PNG} with {IMAGE_MODEL}. Cost: ${USD_PER_IMAGE:.2f}")
    except (RuntimeError, KeyError, IndexError, ValueError) as error:
        # The dashboard still needs a map, and the camera-frame one is accurate.
        shutil.copyfile(FRAME_PNG, HOTSPOT_PNG)
        meta.update(source="camera-frame", error=str(error))
        print(f"{IMAGE_MODEL} failed, so {HOTSPOT_PNG} is the camera-frame map. {error}")

    META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
