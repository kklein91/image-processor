from src.processor import BowlingProcessor
from src.config import Config
import cv2
import os
import numpy as np

cfg = Config()
proc = BowlingProcessor(cfg)

images = [
    "bowling_frames/full_set.jpg",
    "bowling_frames/ball_1.jpg",
]

out_root = "debug_output"
if not os.path.exists(out_root):
    os.makedirs(out_root, exist_ok=True)

for img_path in images:
    print("---", img_path)
    if not os.path.exists(img_path):
        print("MISSING:", img_path)
        continue
    detected = proc.detect_pins_in_image(img_path)
    print("detected:", detected)

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    for i, (nx, ny) in enumerate(cfg.pin_centers, start=1):
        cx = int(nx * w)
        cy = int(ny * h)
        r = cfg.pin_radius
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(w, cx + r)
        y2 = min(h, cy + r)
        roi = blur[y1:y2, x1:x2]
        if roi.size == 0:
            print(f"ROI {i} is empty: {x1,y1,x2,y2}")
            continue
        _, roi_th = cv2.threshold(roi, cfg.roi_white_pixel_threshold, 255, cv2.THRESH_BINARY)
        white = int(np.count_nonzero(roi_th > 0))
        area = roi.shape[0] * roi.shape[1]
        present = white >= max(cfg.roi_area_threshold, area * 0.02)
        print(f"ROI {i}: coords=({x1},{y1},{x2},{y2}) size={roi.shape} area={area} white={white} present={present}")

        # save debug images
        base = os.path.join(out_root, os.path.basename(img_path).replace('.', '_'))
        os.makedirs(base, exist_ok=True)
        cv2.imwrite(os.path.join(base, f"roi_{i}_orig.png"), roi)
        cv2.imwrite(os.path.join(base, f"roi_{i}_th.png"), roi_th)

print("Done")
