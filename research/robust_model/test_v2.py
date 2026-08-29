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

DATASET_DIR = "datasets/robust_lane"
MODEL_PATH = "results/models/robust_lanenet_v3_best.pth"

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TEST DATASET
# ============================================================

class LaneTestDataset(Dataset):

    def __init__(self):

        self.image_dir = os.path.join(
            DATASET_DIR,
            "test",
            "images"
        )

        self.label_dir = os.path.join(
            DATASET_DIR,
            "test",
            "labels"
        )

        self.images = sorted([
            f
            for f in os.listdir(self.image_dir)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ])

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        base_name = os.path.splitext(
            image_name
        )[0]

        label_path = os.path.join(
            self.label_dir,
            base_name + ".json"
        )

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        lanes = data["lanes"]

        # ----------------------------------------------------
        # SUPPORT BOTH JSON FORMATS
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
                f"Unknown lane format in: {label_path}"
            )

        # ----------------------------------------------------
        # CHECK TARGET
        # ----------------------------------------------------

        if len(target) != 5:

            raise ValueError(
                f"Expected 5 lane coordinates in "
                f"{label_path}, got {len(target)}"
            )

        return image, target, image_name


# ============================================================
# TEST
# ============================================================

def test():

    print()
    print("=" * 70)
    print("ROBUST-LANENET TEST")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = LaneTestDataset()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        "Test samples:",
        len(dataset)
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
    # EVALUATION
    # --------------------------------------------------------

    all_errors = []

    print()
    print("-" * 70)

    with torch.no_grad():

        for images, targets, names in loader:

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

            # ------------------------------------------------
            # CONVERT NORMALIZED OUTPUT
            # BACK TO PIXEL COORDINATES
            # ------------------------------------------------

            predictions = (
                predictions * IMAGE_WIDTH
            )

            # ------------------------------------------------
            # CALCULATE ABSOLUTE PIXEL ERROR
            # ------------------------------------------------

            errors = torch.abs(
                predictions - targets
            )

            all_errors.extend(
                errors.cpu().flatten().tolist()
            )

    # --------------------------------------------------------
    # CONVERT ERRORS TO TENSOR
    # --------------------------------------------------------

    all_errors = torch.tensor(
        all_errors,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    mean_error = (
        all_errors.mean().item()
    )

    median_error = (
        all_errors.median().item()
    )

    max_error = (
        all_errors.max().item()
    )

    # --------------------------------------------------------
    # SUCCESS RATE
    # --------------------------------------------------------

    success_threshold = 5.0

    success_rate = (
        (all_errors <= success_threshold)
        .float()
        .mean()
        .item()
        * 100
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ROBUST-LANENET TEST RESULTS")
    print("=" * 70)

    print()

    print(
        f"Mean lane error:   "
        f"{mean_error:.2f} pixels"
    )

    print(
        f"Median lane error: "
        f"{median_error:.2f} pixels"
    )

    print(
        f"Maximum error:     "
        f"{max_error:.2f} pixels"
    )

    print(
        f"Success rate (<=5px): "
        f"{success_rate:.2f}%"
    )

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test()