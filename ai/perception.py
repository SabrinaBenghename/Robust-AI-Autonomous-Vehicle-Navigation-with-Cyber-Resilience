import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ============================================================
# FIND ROBUST-LANENET MODEL
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "research",
    "robust_model"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "models",
    "robust_lanenet_v3_best.pth"
)

# Allow import of research.robust_model.model
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from model import RobustLaneNet


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_WIDTH = 1280
MODEL_HEIGHT = 720

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# ROBUST-LANENET PERCEPTION
# ============================================================

class RobustLanePerception:

    def __init__(
        self,
        model_path=MODEL_PATH
    ):

        print()
        print("=" * 70)
        print("ROBUST-LANENET V3 PERCEPTION")
        print("=" * 70)

        print()
        print("Device:", DEVICE)
        print("Model:", model_path)

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"RobustLaneNet model not found:\n"
                f"{model_path}"
            )

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        self.model = RobustLaneNet().to(
            DEVICE
        )

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE
        )

        self.model.load_state_dict(
            checkpoint
        )

        self.model.eval()

        # ----------------------------------------------------
        # IMAGE TRANSFORMATION
        # ----------------------------------------------------

        self.transform = transforms.Compose([
            transforms.Resize(
                (MODEL_HEIGHT, MODEL_WIDTH)
            ),
            transforms.ToTensor()
        ])

        print()
        print("RobustLaneNet V3 loaded successfully.")

    # ========================================================
    # PREDICT LANES
    # ========================================================

    def predict(self, image):

        """
        Predict five lane x-coordinates.

        Input:
            image:
                OpenCV BGR image or RGB numpy image.

        Output:
            numpy array containing five x-coordinates
            in the model input coordinate system.
        """

        if image is None:

            return None

        # ----------------------------------------------------
        # CONVERT OPENCV BGR → RGB
        # ----------------------------------------------------

        if isinstance(image, np.ndarray):

            if image.ndim != 3:

                raise ValueError(
                    "Expected a color image."
                )

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            pil_image = Image.fromarray(
                image_rgb
            )

        else:

            pil_image = image.convert(
                "RGB"
            )

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        tensor = self.transform(
            pil_image
        )

        tensor = tensor.unsqueeze(
            0
        ).to(DEVICE)

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        with torch.no_grad():

            prediction = self.model(
                tensor
            )

        # ----------------------------------------------------
        # NORMALIZED → PIXELS
        # ----------------------------------------------------

        prediction = (
            prediction.squeeze(0)
            .cpu()
            .numpy()
        )

        lane_x = (
            prediction
            * MODEL_WIDTH
        )

        return lane_x

    # ========================================================
    # DRAW LANES
    # ========================================================

    def draw_lanes(
        self,
        image,
        lane_x
    ):

        """
        Draw the five predicted lane positions
        on an image.
        """

        if image is None:
            return None

        output = image.copy()

        if lane_x is None:
            return output

        height, width = output.shape[:2]

        # Scale model coordinates to current image size.

        scale_x = (
            width /
            MODEL_WIDTH
        )

        # ----------------------------------------------------
        # DRAW EACH LANE
        # ----------------------------------------------------

        for x in lane_x:

            x_draw = int(
                x * scale_x
            )

            cv2.line(
                output,
                (x_draw, 0),
                (x_draw, height),
                (0, 255, 0),
                2
            )

        return output