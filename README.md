# image-processor

This project implements a simple 5-pin bowling image processor.

Usage (watch a folder for incoming images):

```
python -m src.main --input bowling_frames --processed processed_frames
```

Drop images into the `bowling_frames` folder. The script will process images in arrival order, detect knocked-down pins and move each processed image into `processed_frames/frame_X`.

Calibration utility

Interactive selection (creates overlay image + Python config snippet):

Note: Place an image that contains a full pin set named `calibration_baseline` in `./bowling_frames`

```
python -m src.calibrate --image ./bowling_frames/calibration_baseline.jpg
```

What this does:
- Opens an interactive window to click the five pin centers (left-click add, right-click remove, `s` to save).
- Writes an overlay image showing the selected centers to `calibration_overlay.png`.
- Saves a ready-to-paste Python snippet with `pin_centers` to the file specified by `--save-config` (default `copy_calibration_values_to_config.py`).
- Replace lines 25-29 in `config.py` with the output in `copy_calibration_values_to_config.py`

Notes and configuration:
- The calibration overlay circle radius is fixed at 20 pixels and is not user-configurable.
- To have the processor use a known "full set" image as the baseline (instead of learning the baseline at runtime), set `calibration_baseline_image` in `src/config.py` to the baseline image path (for example `"bowling_frames/calibration_baseline.jpg"`).
- Use `roi_mean_diff_threshold` in `src/config.py` to tune sensitivity of per-ROI comparisons (default `15.0`). Lower values make the detector more sensitive to small changes.

Configuration is located in `src/config.py` and contains pin positions and thresholds that you can tune for your camera.

Run unit tests:

```
python -m unittest discover tests
```
