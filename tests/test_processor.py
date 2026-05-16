import os
import tempfile
import unittest
import cv2
import numpy as np

from src.config import Config
from src.processor import BowlingProcessor


class TestBowlingProcessor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config = Config(
            input_folder=os.path.join(self.temp_dir.name, "in"),
            processed_root=os.path.join(self.temp_dir.name, "out"),
            pin_centers=[
                (0.20, 0.40),
                (0.38, 0.38),
                (0.50, 0.36),
                (0.62, 0.38),
                (0.80, 0.40),
            ],
            pin_radius=20,
        )
        os.makedirs(self.config.input_folder, exist_ok=True)
        os.makedirs(self.config.processed_root, exist_ok=True)
        self.processor = BowlingProcessor(self.config)

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_image(self, standing_pins):
        width, height = 640, 480
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for present, (nx, ny) in zip(standing_pins, self.config.pin_centers):
            if present:
                cx = int(nx * width)
                cy = int(ny * height)
                cv2.circle(image, (cx, cy), 18, (255, 255, 255), -1)
        return image

    def save_image(self, image, name):
        path = os.path.join(self.config.input_folder, name)
        cv2.imwrite(path, image)
        return path

    def test_detect_pins_in_image(self):
        image = self.make_image([True, False, True, False, True])
        path = self.save_image(image, "pins.png")
        standing = self.processor.detect_pins_in_image(path)
        self.assertEqual(standing, [True, False, True, False, True])

    def test_process_frame_sequence(self):
        # First ball: all pins standing
        path1 = self.save_image(self.make_image([True, True, True, True, True]), "ball1.png")
        result1 = self.processor.process_image(path1)
        self.assertFalse(result1["frame_done"])
        self.assertEqual(result1["ball"], 1)
        self.assertEqual(result1["standing"], [True, True, True, True, True])

        # Second ball: one pin knocked down
        path2 = self.save_image(self.make_image([True, False, True, True, True]), "ball2.png")
        result2 = self.processor.process_image(path2)
        self.assertFalse(result2["frame_done"])
        self.assertEqual(result2["ball"], 2)
        self.assertEqual(result2["knocked"], [False, True, False, False, False])

        # Third ball: all pins down
        path3 = self.save_image(self.make_image([False, False, False, False, False]), "ball3.png")
        result3 = self.processor.process_image(path3)
        self.assertTrue(result3["frame_done"])
        self.assertEqual(result3["ball"], 3)
        self.assertEqual(len(self.processor.frames_results), 1)
        self.assertEqual(self.processor.frames_results[0]["score"], 15)
        self.assertEqual(self.processor.current_frame, 2)

    def test_strike_bonus_scoring(self):
        path1 = self.save_image(self.make_image([False, False, False, False, False]), "strike.png")
        result1 = self.processor.process_image(path1)
        self.assertTrue(result1["frame_done"])
        self.assertEqual(result1["frame_score"], 15)
        self.assertEqual(len(self.processor.frames_results), 1)
        self.assertIsNone(self.processor.frames_results[0]["score"])

        path2 = self.save_image(self.make_image([True, True, False, True, True]), "ball2.png")
        result2 = self.processor.process_image(path2)
        self.assertFalse(result2["frame_done"])
        self.assertEqual(result2["ball"], 1)

        path3 = self.save_image(self.make_image([False, False, False, True, True]), "ball3.png")
        result3 = self.processor.process_image(path3)
        self.assertFalse(result3["frame_done"])
        self.assertEqual(self.processor.frames_results[0]["score"], 25)

    def test_tenth_frame_allows_three_balls(self):
        self.processor.current_frame = 10

        path1 = self.save_image(self.make_image([False, True, True, True, True]), "tenth1.png")
        result1 = self.processor.process_image(path1)
        self.assertFalse(result1["frame_done"])
        self.assertEqual(result1["ball"], 1)

        path2 = self.save_image(self.make_image([False, False, True, True, True]), "tenth2.png")
        result2 = self.processor.process_image(path2)
        self.assertFalse(result2["frame_done"])
        self.assertEqual(result2["ball"], 2)

        path3 = self.save_image(self.make_image([False, False, False, True, True]), "tenth3.png")
        result3 = self.processor.process_image(path3)
        self.assertTrue(result3["frame_done"])
        self.assertEqual(result3["ball"], 3)
        self.assertEqual(self.processor.frames_results[0]["score"], 10)

    def test_frame_folder_move(self):
        path = self.save_image(self.make_image([True, True, True, True, True]), "frame_move.png")
        result = self.processor.process_image(path)
        expected_folder = os.path.join(self.config.processed_root, "frame_1")
        self.assertTrue(os.path.exists(expected_folder))
        self.assertTrue(os.path.isfile(result["image"]))
        self.assertTrue(result["image"].startswith(expected_folder))


if __name__ == "__main__":
    unittest.main()
