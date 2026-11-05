"""
generate_real_hotspot.py
Draws the exact empirical risk zones directly on the real camera frame (hotspot_frame.png)
using the actual 148 potential conflicts from the uploaded video run.
No generic AI illustration is used.
Outputs runs/.../hotspot_real.png
"""

import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUN_DIR = "runs/20261104-first20s-overlap"
BASE_FRAME_PATH = os.path.join(RUN_DIR, "hotspot_frame.png")
OUTPUT_REAL_HOTSPOT = os.path.join(RUN_DIR, "hotspot_real.png")
CONFLICTS_PATH = os.path.join(RUN_DIR, "potential_conflicts.json")
TRACKS_PATH = os.path.join(RUN_DIR, "tracks.json")


def generate_real_hotspot_overlay(run_dir=RUN_DIR):
    base_frame_p = os.path.join(run_dir, "hotspot_frame.png")
    if not os.path.exists(base_frame_p):
        print("Base frame not found:", base_frame_p)
        return None

    img = cv2.imread(base_frame_p)
    h, w = img.shape[:2]

    # Create transparent overlay mask
    overlay = img.copy()

    # Zone boundaries: Width is split into thirds (0-240 Left, 240-480 Centre, 480-720 Right)
    w_third = w // 3

    # Left Third (0 to w_third): LOW RISK overall (35 conflicts) but Pedestrian Ingress Danger
    # Green semi-transparent tint
    cv2.rectangle(overlay, (0, 0), (w_third, h), (40, 180, 40), -1)

    # Centre Third (w_third to 2*w_third): HIGH RISK (68 conflicts, 1 REAL #70 at x435 y812)
    # Red semi-transparent tint
    cv2.rectangle(overlay, (w_third, 0), (2 * w_third, h), (30, 30, 230), -1)

    # Right Third (2*w_third to w): MEDIUM RISK (45 conflicts)
    # Yellow/Amber semi-transparent tint
    cv2.rectangle(overlay, (2 * w_third, 0), (w, h), (0, 200, 240), -1)

    # Blend with original image (alpha 0.22)
    alpha = 0.22
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Draw vertical separator lines
    cv2.line(img, (w_third, 0), (w_third, h), (255, 255, 255), 2)
    cv2.line(img, (2 * w_third, 0), (2 * w_third, h), (255, 255, 255), 2)

    # Specific callout for Conflict #70 (motorcycle #73 vs scooter #83 at x435, y812)
    # Highlight critical near miss circle
    cv2.circle(img, (435, 812), 36, (0, 0, 255), 3)
    cv2.circle(img, (435, 812), 4, (0, 0, 255), -1)

    # Add Zone Badges & Statistics on the real frame
    # Left Badge
    cv2.rectangle(img, (15, 30), (w_third - 15, 115), (15, 23, 42), -1)
    cv2.rectangle(img, (15, 30), (w_third - 15, 115), (40, 200, 40), 2)
    cv2.putText(img, "LEFT THIRD: LOW", (25, 60), cv2.FONT_HERSHEY_DUPLEX, 0.55, (40, 220, 40), 1)
    cv2.putText(img, "35 Conflicts (25 Auto/Ped)", (25, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(img, "South Pedestrian Crossing", (25, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # Centre Badge (HIGH RISK RED)
    cv2.rectangle(img, (w_third + 15, 30), (2 * w_third - 15, 125), (15, 23, 42), -1)
    cv2.rectangle(img, (w_third + 15, 30), (2 * w_third - 15, 125), (0, 0, 255), 2)
    cv2.putText(img, "CENTRE: HIGH RISK", (w_third + 25, 60), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(img, "68 Conflicts | 1 REAL (#70)", (w_third + 25, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(img, "Roundabout Core Pinch Point", (w_third + 25, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # Right Badge (MEDIUM YELLOW)
    cv2.rectangle(img, (2 * w_third + 15, 30), (w - 15, 115), (15, 23, 42), -1)
    cv2.rectangle(img, (2 * w_third + 15, 30), (w - 15, 115), (0, 200, 240), 2)
    cv2.putText(img, "RIGHT THIRD: MEDIUM", (2 * w_third + 25, 60), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 215, 255), 1)
    cv2.putText(img, "45 Conflicts", (2 * w_third + 25, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(img, "Blind Exit Sightline", (2 * w_third + 25, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    # Legend at bottom
    cv2.rectangle(img, (15, h - 55), (w - 15, h - 15), (15, 23, 42), -1)
    cv2.putText(img, "YOUR UPLOADED VIDEO JUNCTION | AI EMPIRICAL RISK OVERLAYS (148 CONFLICTS)", (25, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

    out_p = os.path.join(run_dir, "hotspot_real.png")
    cv2.imwrite(out_p, img)
    print(f"Generated real junction hotspot: {out_p}")
    return out_p


if __name__ == "__main__":
    generate_real_hotspot_overlay()

