import os
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms

from model import RobustLaneNet
from loss import LaneLoss


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "datasets/robust_lane"

BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.001

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# V3 TARGETED ROBUSTNESS AUGMENTATION
# ============================================================

def add_noise(image, strength=None):

    if strength is None:
        strength = random.uniform(0.06, 0.14)

    image = image.convert("RGB")

    pixels = torch.from_numpy(
        __import__("numpy").array(image)
    ).float()

    noise = torch.randn_like(pixels) * (
        strength * 255.0
    )

    pixels = pixels + noise

    pixels = torch.clamp(
        pixels,
        0,
        255
    ).byte()

    return Image.fromarray(
        pixels.numpy()
    )


def apply_blur(image):

    radius = random.uniform(
        1.5,
        2.5
    )

    return image.filter(
        ImageFilter.GaussianBlur(
            radius=radius
        )
    )


def reduce_contrast(image):

    factor = random.uniform(
        0.40,
        0.65
    )

    enhancer = ImageEnhance.Contrast(
        image
    )

    return enhancer.enhance(
        factor
    )


def apply_occlusion(image):

    image = image.copy()

    width, height = image.size

    max_width = int(
        width * 0.18
    )

    max_height = int(
        height * 0.18
    )

    block_width = random.randint(
        int(width * 0.08),
        max_width
    )

    block_height = random.randint(
        int(height * 0.08),
        max_height
    )

    left = random.randint(
        0,
        width - block_width
    )

    top = random.randint(
        int(height * 0.35),
        height - block_height
    )

    pixels = __import__(
        "numpy"
    ).array(image)

    pixels[
        top:top + block_height,
        left:left + block_width
    ] = 0

    return Image.fromarray(
        pixels
    )


def apply_v3_augmentation(image):

    r = random.random()

    # --------------------------------------------------------
    # Clean: 35%
    # --------------------------------------------------------

    if r < 0.35:

        return image, "clean"

    # --------------------------------------------------------
    # Stronger / varied noise: 30%
    # --------------------------------------------------------

    elif r < 0.65:

        return add_noise(
            image
        ), "noise"

    # --------------------------------------------------------
    # Blur: 15%
    # --------------------------------------------------------

    elif r < 0.80:

        return apply_blur(
            image
        ), "blur"

    # --------------------------------------------------------
    # Stronger / varied low contrast: 15%
    # --------------------------------------------------------

    elif r < 0.95:

        return reduce_contrast(
            image
        ), "low_contrast"

    # --------------------------------------------------------
    # Occlusion: 5%
    # --------------------------------------------------------

    else:

        return apply_occlusion(
            image
        ), "occlusion"


# ============================================================
# LANE DATASET
# ============================================================

class LaneDataset(Dataset):

    def __init__(
        self,
        split,
        augment=False
    ):

        self.image_dir = os.path.join(
            DATASET_DIR,
            split,
            "images"
        )

        self.label_dir = os.path.join(
            DATASET_DIR,
            split,
            "labels"
        )

        self.augment = augment

        self.images = sorted([
            f
            for f in os.listdir(
                self.image_dir
            )
            if f.lower().endswith(
                (
                    ".png",
                    ".jpg",
                    ".jpeg"
                )
            )
        ])

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):

        return len(
            self.images
        )

    def __getitem__(
        self,
        index
    ):

        image_name = self.images[index]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # V3 augmentation
        # ----------------------------------------------------

        if self.augment:

            image, _ = (
                apply_v3_augmentation(
                    image
                )
            )

        image = self.transform(
            image
        )

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

            data = json.load(
                file
            )

        lanes = data["lanes"]

        # ----------------------------------------------------
        # Convert pixel coordinates
        # to normalized coordinates
        # ----------------------------------------------------

        if isinstance(
            lanes,
            dict
        ):

            lanes = [
                lanes["road_left"],
                lanes["lane_line_1"],
                lanes["center_line"],
                lanes["lane_line_2"],
                lanes["road_right"]
            ]

        lanes = [
            float(x) / IMAGE_WIDTH
            for x in lanes
        ]

        target = torch.tensor(
            lanes,
            dtype=torch.float32
        )

        return image, target


# ============================================================
# CREATE DATA LOADERS
# ============================================================

def create_loaders():

    train_dataset = LaneDataset(
        "train",
        augment=True
    )

    val_dataset = LaneDataset(
        "val",
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return (
        train_loader,
        val_loader
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(
                DEVICE
            )

            targets = targets.to(
                DEVICE
            )

            predictions = model(
                images
            )

            loss = criterion(
                predictions,
                targets
            )

            total_loss += loss.item()

    return (
        total_loss /
        len(loader)
    )


# ============================================================
# TRAINING
# ============================================================

def train():

    print()
    print("=" * 70)
    print("ROBUST-LANENET V3 GEOMETRY-AWARE TRAINING")
    print("=" * 70)

    print()
    print(
        "Device:",
        DEVICE
    )

    print()
    print("V3 Training augmentation:")
    print("  Clean:        35%")
    print("  Noise:        30%")
    print("  Blur:         15%")
    print("  Low contrast: 15%")
    print("  Occlusion:     5%")

    train_loader, val_loader = (
        create_loaders()
    )

    print()
    print(
        "Training samples:",
        len(train_loader.dataset)
    )

    print(
        "Validation samples:",
        len(val_loader.dataset)
    )

    model = RobustLaneNet().to(
        DEVICE
    )

    # V3 geometry-aware loss
    criterion = LaneLoss(
        geometry_weight=0.10
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    os.makedirs(
        "results/models",
        exist_ok=True
    )

    best_val_loss = float(
        "inf"
    )

    print()
    print("-" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        for images, targets in train_loader:

            images = images.to(
                DEVICE
            )

            targets = targets.to(
                DEVICE
            )

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(
                images
            )

            # V3 geometry-aware loss
            loss = criterion(
                predictions,
                targets
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += (
                loss.item()
            )

        train_loss = (
            running_loss /
            len(train_loader)
        )

        val_loss = validate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # ----------------------------------------------------
        # Save best V3 model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "results/models/"
                "robust_lanenet_v3_best.pth"
            )

            print(
                "  -> Best V3 model saved."
            )

    print()
    print("=" * 70)
    print("ROBUST-LANENET V3 TRAINING COMPLETE")
    print("=" * 70)

    print()
    print(
        "Best validation loss:",
        best_val_loss
    )

    print()
    print(
        "Model saved to:"
    )

    print(
        "results/models/"
        "robust_lanenet_v3_best.pth"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()