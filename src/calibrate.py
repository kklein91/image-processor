import argparse
import cv2
import os
from pathlib import Path
from typing import List, Tuple


def ensure_dir(folder: str) -> None:
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


STATIC_OVERLAY_RADIUS = 10  # pixels, fixed size for calibration overlay circles


def draw_calibration_overlay(image_path: str, pin_centers: List[Tuple[float, float]], output_path: str) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = image.shape[:2]
    radius = STATIC_OVERLAY_RADIUS

    for index, (nx, ny) in enumerate(pin_centers, start=1):
        cx = int(nx * w)
        cy = int(ny * h)
        cv2.circle(image, (cx, cy), radius, (0, 0, 255), 2)
        cv2.putText(
            image,
            str(index),
            (cx - 10, cy - radius - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    ensure_dir(os.path.dirname(output_path) or ".")
    cv2.imwrite(output_path, image)


def save_calibration_snippet(output_path: Path, pin_centers: List[Tuple[float, float]]) -> None:
    # Write only the tuple lines, one per line, no surrounding list or variable.
    lines = []
    for x, y in pin_centers:
        lines.append(f"({x:.16f}, {y:.16f}),")

    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def interactive_select_pin_centers(image_path: str, pin_count: int = 5) -> List[Tuple[float, float]]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    clone = image.copy()
    points: List[Tuple[int, int]] = []
    window_name = "Calibration - click pin centers"

    def draw_points(frame):
        for idx, (x, y) in enumerate(points, start=1):
            cv2.circle(frame, (x, y), 12, (0, 0, 255), 2)
            cv2.putText(frame, str(idx), (x - 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Left-click to add, right-click to remove, s=save, r=reset, q=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Selected {len(points)} / {pin_count}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < pin_count:
                points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display = clone.copy()
        draw_points(display)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q") or key == 27:
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt("Calibration cancelled by user")
        if key == ord("r"):
            points.clear()
        if key == ord("s") and len(points) == pin_count:
            break

    cv2.destroyWindow(window_name)
    points_sorted = sorted(points, key=lambda p: p[0])
    h, w = image.shape[:2]
    return [(x / w, y / h) for x, y in points_sorted]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a calibration overlay from interactive pin selection and save values to a Python config snippet.")
    parser.add_argument("--image", required=True, help="Path to a sample image with all pins visible.")
    parser.add_argument(
        "--output",
        default="calibration_overlay.png",
        help="Output path for the calibration overlay image.",
    )
    parser.add_argument(
        "--save-config",
        default="copy_calibration_values_to_config.py",
        help="Save pin centers as a Python config snippet for config.py.",
    )
    args = parser.parse_args()

    pin_centers = interactive_select_pin_centers(args.image)
    draw_calibration_overlay(args.image, pin_centers, args.output)

    print(f"Calibration overlay saved to: {args.output}")
    print("Selected pin centers (normalized x,y):")
    for idx, center in enumerate(pin_centers, start=1):
        print(f"  Pin {idx}: ({center[0]:.6f}, {center[1]:.6f})")
    print(f"Overlay circle radius: {STATIC_OVERLAY_RADIUS} pixels")

    save_path = Path(args.save_config)
    save_calibration_snippet(save_path, pin_centers)
    print(f"Saved Python config snippet to: {save_path}")


if __name__ == "__main__":
    main()
