"""
generate_c70_crops.py
Generates the isolated single-track videos for motorcycle #73 and scooter #83
during their conflict frames (122-152), cropped with padding and trajectory trail.
"""

import os
import json
import cv2
import numpy as np

RUN_DIR = "runs/20261104-first20s-overlap"
VIDEO_PATH = os.path.join(RUN_DIR, "video.mp4")
TRACKS_PATH = os.path.join(RUN_DIR, "tracks.json")
OUT_DIR = os.path.join(RUN_DIR, "clips", "individual")
os.makedirs(OUT_DIR, exist_ok=True)


def generate_single_object_clip(obj_id, label, color_bgr, out_name):
    with open(TRACKS_PATH, "r") as f:
        tracks = json.load(f)

    # Frame 122 to 152 (clip_start is 122)
    obj_data = {t["frame_id"]: t["bbox"] for t in tracks if t["object_id"] == obj_id}
    all_frames = sorted(obj_data.keys())
    if not all_frames:
        print(f"No frames for object {obj_id}")
        return

    # Calculate bounding crop center
    all_boxes = [obj_data[f] for f in all_frames]
    min_x = min(b[0] for b in all_boxes)
    max_x = max(b[2] for b in all_boxes)
    min_y = min(b[1] for b in all_boxes)
    max_y = max(b[3] for b in all_boxes)

    center_x = int((min_x + max_x) / 2)
    center_y = int((min_y + max_y) / 2)

    # Crop box 240x240 around object center
    crop_size = 240
    half = crop_size // 2

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Clamp crop coordinates
    c_x1 = max(0, min(center_x - half, video_w - crop_size))
    c_y1 = max(0, min(center_y - half, video_h - crop_size))
    c_x2 = c_x1 + crop_size
    c_y2 = c_y1 + crop_size

    out_path = os.path.join(OUT_DIR, out_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (crop_size, crop_size))

    trail = []
    frame_id = 0
    start_frame = 122
    end_frame = 152

    while True:
        ret, frame = cap.read()
        if not ret or frame_id > end_frame:
            break

        if start_frame <= frame_id <= end_frame:
            crop = frame[c_y1:c_y2, c_x1:c_x2].copy()

            if frame_id in obj_data:
                bx = obj_data[frame_id]
                # Map to crop coords
                bx1 = bx[0] - c_x1
                by1 = bx[1] - c_y1
                bx2 = bx[2] - c_x1
                by2 = bx[3] - c_y1

                # Center point
                cx = int((bx1 + bx2) / 2)
                cy = int((by1 + by2) / 2)
                trail.append((cx, cy))
                if len(trail) > 12:
                    trail.pop(0)

                # Draw trail
                for i in range(1, len(trail)):
                    cv2.line(crop, trail[i-1], trail[i], color_bgr, 2)
                    cv2.circle(crop, trail[i], 3, color_bgr, -1)

                # Draw bbox
                cv2.rectangle(crop, (bx1, by1), (bx2, by2), color_bgr, 2)
                cv2.putText(crop, f"{label} #{obj_id}", (bx1, max(15, by1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)

            # Frame label
            cv2.putText(crop, f"Frame {frame_id}", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            writer.write(crop)

        frame_id += 1

    cap.release()
    writer.release()
    print(f"Generated {out_path} ({crop_size}x{crop_size})")


if __name__ == "__main__":
    generate_single_object_clip(73, "Motorcycle", (0, 0, 255), "track_073_alone.mp4")
    generate_single_object_clip(83, "Scooter", (0, 255, 0), "track_083_alone.mp4")

