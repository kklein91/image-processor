from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # Input folder where images are dropped
    input_folder: str = "bowling_frames"

    # Root folder where processed/frame folders are created
    processed_root: str = "processed_frames"

    # Number of frames in a game
    total_frames: int = 10

    # Max balls per frame
    max_balls_per_frame: int = 3

    # Pin values left -> right
    pin_values: List[int] = field(default_factory=lambda: [2, 3, 5, 3, 2])

    # Order: left -> right (5 pins)
    pin_centers: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.3322916666666667, 0.4768518518518519),
            (0.3921875000000000, 0.5305555555555556),
            (0.4786458333333333, 0.6129629629629629),
            (0.5651041666666666, 0.5407407407407407),
            (0.6229166666666667, 0.4879629629629630),
        ]
    )

    # ROI radius in pixels around each pin center used for pin presence detection.
    pin_radius: int = 10
    # Brightness/area thresholds for deciding a pin is present.
    roi_white_pixel_threshold: int = 30
    roi_area_threshold: int = 50
    # Mean absolute-difference threshold (between baseline ROI and current ROI) used
    # to decide whether a pin is still present. Lower = more sensitive to change.
    roi_mean_diff_threshold: float = 15.0
    # Optional path to a baseline image (full set of pins). If provided, the
    # processor will load this image at startup and use it as the baseline for
    # per-ROI comparisons instead of waiting for the first all-pins image.
    calibration_baseline_image: str = "bowling_frames/calibration_baseline.jpg"

    # Processing wait time for newly created files (seconds)
    file_stability_wait: float = 0.5

    # Whether to run continuously (watch) or batch process once
    continuous: bool = True
