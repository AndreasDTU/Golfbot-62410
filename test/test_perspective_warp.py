import unittest
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    cv2 = None

from camera.calibration import (
    compute_inner_corner_quad,
    find_checkerboard_corners,
    warp_from_checkerboard_inner_corners,
)

INPUT_IMAGE = Path("/Users/alex/PycharmProjects/Golfbot-62410/test/images/checkerboard_test.png")
PATTERN = (29, 17)
SQUARE_PX = 40


@unittest.skipIf(cv2 is None, "cv2 not installed")
class PerspectiveWarpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_COLOR)
        if cls.image is None:
            raise unittest.SkipTest(f"Input image missing or unreadable: {INPUT_IMAGE}")

        found, corners = find_checkerboard_corners(cls.image, pattern=PATTERN)
        if not found or corners is None:
            raise unittest.SkipTest("Checkerboard corners not found in test image")
        cls.corners = corners

    def test_compute_inner_corner_quad_shape(self) -> None:
        quad = compute_inner_corner_quad(self.corners, pattern=PATTERN)
        self.assertEqual(quad.shape, (4, 2))

    def test_warp_dimensions_match_pattern_and_square_px(self) -> None:
        warped, homography, src_quad = warp_from_checkerboard_inner_corners(
            image_bgr=self.image,
            corners=self.corners,
            pattern=PATTERN,
            square_px=SQUARE_PX,
        )
        expected_width = (PATTERN[0] - 1) * SQUARE_PX
        expected_height = (PATTERN[1] - 1) * SQUARE_PX
        self.assertEqual(warped.shape[1], expected_width)
        self.assertEqual(warped.shape[0], expected_height)
        self.assertEqual(src_quad.shape, (4, 2))
        self.assertEqual(homography.shape, (3, 3))

    def test_warp_is_deterministic(self) -> None:
        warped_1, homography_1, _ = warp_from_checkerboard_inner_corners(
            image_bgr=self.image,
            corners=self.corners,
            pattern=PATTERN,
            square_px=SQUARE_PX,
        )
        warped_2, homography_2, _ = warp_from_checkerboard_inner_corners(
            image_bgr=self.image,
            corners=self.corners,
            pattern=PATTERN,
            square_px=SQUARE_PX,
        )
        self.assertEqual(warped_1.tolist(), warped_2.tolist())
        self.assertEqual(homography_1.tolist(), homography_2.tolist())


if __name__ == "__main__":
    unittest.main()
