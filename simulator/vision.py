import pygame
import cv2
import numpy as np


class VisionSensor:

    def __init__(self, width=640, height=360):

        self.width = width
        self.height = height

        # ----------------------------------------------------
        # IMAGES
        # ----------------------------------------------------

        self.clean_image = None
        self.image = None
        self.processed_image = None

        # ----------------------------------------------------
        # ROBUSTNESS TEST MODE
        # ----------------------------------------------------

        self.robustness_mode = "CLEAN"

        # Noise strength
        self.noise_std = 20.0

        # Blur strength
        self.blur_kernel = 11

        # Reproducible random generator
        self.rng = np.random.default_rng(42)

    # ========================================================
    # ROBUSTNESS MODE
    # ========================================================

    def set_robustness_mode(self, mode):

        valid_modes = [
            "CLEAN",
            "NOISE",
            "BLUR"
        ]

        if mode not in valid_modes:
            return

        self.robustness_mode = mode

        print(
            f"\n[ROBUSTNESS] Mode changed to: {mode}"
        )

    # ========================================================
    # APPLY CORRUPTION
    # ========================================================

    def apply_corruption(self, image):

        # ----------------------------------------------------
        # CLEAN
        # ----------------------------------------------------

        if self.robustness_mode == "CLEAN":

            return image.copy()

        # ----------------------------------------------------
        # GAUSSIAN NOISE
        # ----------------------------------------------------

        if self.robustness_mode == "NOISE":

            noise = self.rng.normal(
                0,
                self.noise_std,
                image.shape
            )

            noisy_image = (
                image.astype(np.float32)
                + noise
            )

            noisy_image = np.clip(
                noisy_image,
                0,
                255
            )

            return noisy_image.astype(
                np.uint8
            )

        # ----------------------------------------------------
        # GAUSSIAN BLUR
        # ----------------------------------------------------

        if self.robustness_mode == "BLUR":

            return cv2.GaussianBlur(
                image,
                (
                    self.blur_kernel,
                    self.blur_kernel
                ),
                0
            )

        return image.copy()

    # ========================================================
    # CAPTURE
    # ========================================================

    def capture(self, screen):

        # ----------------------------------------------------
        # CAPTURE PYGAME SCREEN
        # ----------------------------------------------------

        raw_image = pygame.surfarray.array3d(
            screen
        )

        # Pygame:
        # width x height x RGB
        #
        # OpenCV:
        # height x width x RGB

        raw_image = np.transpose(
            raw_image,
            (1, 0, 2)
        )

        # ----------------------------------------------------
        # RGB -> BGR
        # ----------------------------------------------------

        image = cv2.cvtColor(
            raw_image,
            cv2.COLOR_RGB2BGR
        )

        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        image = cv2.resize(
            image,
            (
                self.width,
                self.height
            )
        )

        # ----------------------------------------------------
        # SAVE CLEAN IMAGE
        # ----------------------------------------------------

        self.clean_image = image.copy()

        # ----------------------------------------------------
        # APPLY ROBUSTNESS CONDITION
        # ----------------------------------------------------

        image = self.apply_corruption(
            image
        )

        self.image = image

        return image

    # ========================================================
    # CLASSICAL LANE DETECTION
    # ========================================================

    def detect_lanes(self):

        if self.image is None:
            return None

        # ----------------------------------------------------
        # 1. GRAYSCALE
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            self.image,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # 2. BLUR
        # ----------------------------------------------------

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # ----------------------------------------------------
        # 3. EDGE DETECTION
        # ----------------------------------------------------

        edges = cv2.Canny(
            blur,
            50,
            150
        )

        # ----------------------------------------------------
        # 4. REGION OF INTEREST
        # ----------------------------------------------------

        mask = np.zeros_like(edges)

        height, width = edges.shape

        polygon = np.array([
            [
                (
                    int(width * 0.1),
                    height
                ),
                (
                    int(width * 0.4),
                    int(height * 0.55)
                ),
                (
                    int(width * 0.6),
                    int(height * 0.55)
                ),
                (
                    int(width * 0.9),
                    height
                )
            ]
        ], np.int32)

        cv2.fillPoly(
            mask,
            polygon,
            255
        )

        roi = cv2.bitwise_and(
            edges,
            mask
        )

        # ----------------------------------------------------
        # 5. HOUGH TRANSFORM
        # ----------------------------------------------------

        lines = cv2.HoughLinesP(
            roi,
            1,
            np.pi / 180,
            threshold=40,
            minLineLength=30,
            maxLineGap=100
        )

        # ----------------------------------------------------
        # 6. DRAW DETECTED LINES
        # ----------------------------------------------------

        output = self.image.copy()

        if lines is not None:

            for line in lines:

                x1, y1, x2, y2 = line[0]

                cv2.line(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

        self.processed_image = output

        return output

    # ========================================================
    # GETTERS
    # ========================================================

    def get_image(self):

        return self.image

    def get_clean_image(self):

        return self.clean_image

    def get_processed_image(self):

        return self.processed_image

    def get_robustness_mode(self):

        return self.robustness_mode