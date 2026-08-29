import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from model import RobustLaneNet


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "results/robustness"

# IMPORTANT:
# These are the original labels for ALL 257 images.
LABEL_DIR = "datasets/labels"

MODEL_PATH = "results/models/robust_lanenet_v3_best.pth"

IMAGE_WIDTH = 1280

BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class RobustnessDataset(Dataset):

    def __init__(self, condition):

        self.condition = condition

        # Example:
        # results/robustness/noise
        # results/robustness/blur
        # results/robustness/low_contrast
        # results/robustness/occlusion

        self.image_dir = os.path.join(
            DATASET_DIR,
            condition
        )

        # ALL 257 original labels
        self.label_dir = LABEL_DIR

        if not os.path.exists(self.image_dir):

            raise FileNotFoundError(
                f"Robustness image directory not found: "
                f"{self.image_dir}"
            )

        if not os.path.exists(self.label_dir):

            raise FileNotFoundError(
                f"Label directory not found: "
                f"{self.label_dir}"
            )

        self.images = sorted(
            [
                f
                for f in os.listdir(self.image_dir)
                if f.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                )
            ]
        )

        if len(self.images) == 0:

            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

        self.to_tensor = transforms.ToTensor()

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.to_tensor(
            image
        )

        # ----------------------------------------------------
        # FIND ORIGINAL LABEL
        # ----------------------------------------------------

        base_name = os.path.splitext(
            image_name
        )[0]

        label_path = os.path.join(
            self.label_dir,
            base_name + ".json"
        )

        if not os.path.exists(label_path):

            raise FileNotFoundError(
                f"Label not found for {image_name}: "
                f"{label_path}"
            )

        # ----------------------------------------------------
        # LOAD JSON
        # ----------------------------------------------------

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        lanes = data["lanes"]

        # ----------------------------------------------------
        # JSON FORMAT
        # ----------------------------------------------------

        if isinstance(lanes, dict):

            target = torch.tensor(
                [
                    lanes["road_left"],
                    lanes["lane_line_1"],
                    lanes["center_line"],
                    lanes["lane_line_2"],
                    lanes["road_right"]
                ],
                dtype=torch.float32
            )

        elif isinstance(lanes, list):

            target = torch.tensor(
                lanes,
                dtype=torch.float32
            )

        else:

            raise ValueError(
                f"Unknown lane format in: "
                f"{label_path}"
            )

        return image, target


# ============================================================
# EVALUATE ONE CONDITION
# ============================================================

def evaluate_condition(
    model,
    condition
):

    dataset = RobustnessDataset(
        condition
    )

    print(
        f"  Images: {len(dataset)}"
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    errors = []

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(
                DEVICE
            )

            targets = targets.to(
                DEVICE
            )

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            predictions = model(
                images
            )

            # Model outputs normalized coordinates
            # between 0 and 1.
            #
            # Convert them back to pixel coordinates.

            predictions = (
                predictions * IMAGE_WIDTH
            )

            # ------------------------------------------------
            # PIXEL ERROR
            # ------------------------------------------------

            batch_errors = torch.abs(
                predictions - targets
            )

            errors.extend(
                batch_errors
                .cpu()
                .flatten()
                .tolist()
            )

    errors = torch.tensor(
        errors,
        dtype=torch.float32
    )

    mean_error = errors.mean().item()

    median_error = errors.median().item()

    success_rate = (
        (errors <= 5.0)
        .float()
        .mean()
        .item()
        * 100
    )

    return (
        mean_error,
        median_error,
        success_rate
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)
    print("ROBUST-LANENET FAIR ROBUSTNESS EVALUATION")
    print("=" * 70)

    print()

    print(
        "Device:",
        DEVICE
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Images:",
        DATASET_DIR
    )

    print(
        "Labels:",
        LABEL_DIR
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = RobustLaneNet().to(
        DEVICE
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    print()

    print(
        "Model loaded:",
        MODEL_PATH
    )

    # --------------------------------------------------------
    # CONDITIONS
    # --------------------------------------------------------

    conditions = [
        "noise",
        "blur",
        "low_contrast",
        "occlusion"
    ]

    results = {}

    print()

    print("-" * 70)

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    for condition in conditions:

        print(
            f"Testing: {condition}..."
        )

        mean_error, median_error, success_rate = (
            evaluate_condition(
                model,
                condition
            )
        )

        results[condition] = {
            "mean_error": mean_error,
            "median_error": median_error,
            "success_rate": success_rate
        }

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()

    print("=" * 70)
    print("ROBUST-LANENET FAIR ROBUSTNESS RESULTS")
    print("=" * 70)

    print()

    print(
        f"{'Condition':<18}"
        f"{'Mean Error':>15}"
        f"{'Median Error':>17}"
        f"{'Success':>14}"
    )

    print("-" * 70)

    for condition in conditions:

        metrics = results[condition]

        print(
            f"{condition:<18}"
            f"{metrics['mean_error']:>15.2f}"
            f"{metrics['median_error']:>17.2f}"
            f"{metrics['success_rate']:>13.2f}%"
        )

    print()

    print("=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()