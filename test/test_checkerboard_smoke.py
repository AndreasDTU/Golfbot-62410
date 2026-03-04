import unittest
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    cv2 = None

from camera.calibration import (
    calibrate_from_single_checkerboard,
    draw_checkerboard_overlay,
    find_checkerboard_corners,
    make_side_by_side,
    undistort_image,
)


INPUT_IMAGE = Path("/Users/alex/PycharmProjects/Golfbot-62410/test/images/checkerboard_test.png")
PATTERN = (29, 17)
EXPECTED_CORNER_COUNT = 493


@unittest.skipIf(cv2 is None, "cv2 not installed")
class CheckerboardSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_COLOR)
        if cls.image is None:
            raise unittest.SkipTest(f"Input image missing or unreadable: {INPUT_IMAGE}")

    def test_checkerboard_found_with_expected_corner_count(self) -> None:
        found, corners = find_checkerboard_corners(self.image, pattern=PATTERN)
        self.assertTrue(found)
        self.assertIsNotNone(corners)
        self.assertEqual(int(corners.shape[0]), EXPECTED_CORNER_COUNT)

    def test_overlay_matches_input_dimensions(self) -> None:
        found, corners = find_checkerboard_corners(self.image, pattern=PATTERN)
        overlay = draw_checkerboard_overlay(
            image_bgr=self.image,
            pattern=PATTERN,
            corners=corners,
            found=found,
        )
        self.assertEqual(overlay.shape, self.image.shape)

    def test_detection_is_deterministic_for_repeated_runs(self) -> None:
        found_1, corners_1 = find_checkerboard_corners(self.image, pattern=PATTERN)
        found_2, corners_2 = find_checkerboard_corners(self.image, pattern=PATTERN)

        self.assertEqual(found_1, found_2)
        count_1 = 0 if corners_1 is None else int(corners_1.shape[0])
        count_2 = 0 if corners_2 is None else int(corners_2.shape[0])
        self.assertEqual(count_1, count_2)

        if corners_1 is not None and corners_2 is not None:
            self.assertEqual(corners_1.tolist(), corners_2.tolist())

    def test_provisional_undistort_and_side_by_side_shapes(self) -> None:
        found, corners = find_checkerboard_corners(self.image, pattern=PATTERN)
        self.assertTrue(found)
        self.assertIsNotNone(corners)

        _, camera_matrix, dist_coeffs = calibrate_from_single_checkerboard(
            image_bgr=self.image,
            corners=corners,
            pattern=PATTERN,
            square_size=1.0,
        )
        undistorted = undistort_image(self.image, camera_matrix, dist_coeffs)
        self.assertEqual(undistorted.shape, self.image.shape)

        side_by_side = make_side_by_side(self.image, undistorted)
        self.assertEqual(side_by_side.shape[0], self.image.shape[0])
        self.assertEqual(side_by_side.shape[1], self.image.shape[1] * 2)
        self.assertEqual(side_by_side.shape[2], self.image.shape[2])


if __name__ == "__main__":
    unittest.main()
