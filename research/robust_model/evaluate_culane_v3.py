
import os
import json

import torch
import numpy as np

from PIL import Image
from torchvision import transforms

from model import RobustLaneNet


# ============================================================
# CONFIGURATION
# ============================================================

CULANE_DIR = "datasets/external/culane"

MODEL_PATH = "results/models/robust_lanenet_v3_best.pth"

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Original CULane resolution
CULANE_WIDTH = 1640
CULANE_HEIGHT = 590

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SUCCESS_THRESHOLD = 5.0

# Vertical positions in MODEL coordinates
Y_LEVELS = [
    300,
    380,
    460,
    540,
    620
]


# ============================================================
# ASPECT-RATIO-PRESERVING TRANSFORMATION
# ============================================================

def get_resize_parameters():

    scale = min(
        IMAGE_WIDTH / CULANE_WIDTH,
        IMAGE_HEIGHT / CULANE_HEIGHT
    )

    resized_width = int(
        round(CULANE_WIDTH * scale)
    )

    resized_height = int(
        round(CULANE_HEIGHT * scale)
    )

    pad_x = (
        IMAGE_WIDTH - resized_width
    ) / 2.0

    pad_y = (
        IMAGE_HEIGHT - resized_height
    ) / 2.0

    return (
        scale,
        resized_width,
        resized_height,
        pad_x,
        pad_y
    )


(
    SCALE,
    RESIZED_WIDTH,
    RESIZED_HEIGHT,
    PAD_X,
    PAD_Y
) = get_resize_parameters()


print()
print("CULane transformation:")
print(
    f"  Original: "
    f"{CULANE_WIDTH} x {CULANE_HEIGHT}"
)

print(
    f"  Resized:  "
    f"{RESIZED_WIDTH} x {RESIZED_HEIGHT}"
)

print(
    f"  Padding:  "
    f"x={PAD_X:.2f}, y={PAD_Y:.2f}"
)

print(
    f"  Scale:    "
    f"{SCALE:.6f}"
)


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

def preprocess_image(image):

    # Preserve aspect ratio
    image = image.resize(
        (
            RESIZED_WIDTH,
            RESIZED_HEIGHT
        ),
        Image.Resampling.BILINEAR
    )

    # Create 1280 x 720 canvas
    canvas = Image.new(
        "RGB",
        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        ),
        (0, 0, 0)
    )

    canvas.paste(
        image,
        (
            int(round(PAD_X)),
            int(round(PAD_Y))
        )
    )

    tensor = transforms.ToTensor()(
        canvas
    )

    return tensor


# ============================================================
# FIND CULANE SAMPLES
# ============================================================

def find_samples():

    samples = []

    print()
    print("Searching CULane dataset...")

    for root, dirs, files in os.walk(
        CULANE_DIR
    ):

        for filename in files:

            if not filename.endswith(
                ".lines.txt"
            ):
                continue

            label_path = os.path.join(
                root,
                filename
            )

            image_filename = filename.replace(
                ".lines.txt",
                ".jpg"
            )

            image_path = os.path.join(
                root,
                image_filename
            )

            if not os.path.exists(
                image_path
            ):
                continue

            samples.append(
                (
                    image_path,
                    label_path
                )
            )

    samples.sort()

    return samples


# ============================================================
# READ CULANE POLYLINES
# ============================================================

def read_lanes(label_path):

    lanes = []

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            values = line.strip().split()

            if len(values) < 4:
                continue

            values = [
                float(v)
                for v in values
            ]

            points = []

            for i in range(
                0,
                len(values) - 1,
                2
            ):

                x = values[i]
                y = values[i + 1]

                points.append(
                    (x, y)
                )

            if len(points) >= 2:

                lanes.append(
                    points
                )

    return lanes


# ============================================================
# TRANSFORM CULANE POINT
# ============================================================

def transform_point(x, y):

    new_x = (
        x * SCALE
        + PAD_X
    )

    new_y = (
        y * SCALE
        + PAD_Y
    )

    return (
        new_x,
        new_y
    )


# ============================================================
# INTERPOLATE X AT MODEL Y
# ============================================================

def interpolate_x(
    points,
    target_y
):

    transformed = [
        transform_point(x, y)
        for x, y in points
    ]

    transformed.sort(
        key=lambda p: p[1]
    )

    for i in range(
        len(transformed) - 1
    ):

        x1, y1 = transformed[i]
        x2, y2 = transformed[i + 1]

        if (
            y1 <= target_y <= y2
            or
            y2 <= target_y <= y1
        ):

            if abs(y2 - y1) < 1e-6:

                return (
                    x1 + x2
                ) / 2.0

            ratio = (
                target_y - y1
            ) / (
                y2 - y1
            )

            return (
                x1 +
                ratio *
                (x2 - x1)
            )

    return None


# ============================================================
# EXTRACT LANE POSITIONS
# ============================================================

def lane_positions(
    lanes,
    y_level
):

    positions = []

    for lane in lanes:

        x = interpolate_x(
            lane,
            y_level
        )

        if x is not None:

            positions.append(
                x
            )

    positions.sort()

    return positions


# ============================================================
# MATCH LANES
# ============================================================

def calculate_sample_error(
    predictions,
    lanes
):

    all_errors = []

    for y_level in Y_LEVELS:

        gt_positions = lane_positions(
            lanes,
            y_level
        )

        if len(gt_positions) == 0:
            continue

        # ----------------------------------------------------
        # Predictions are five ordered x positions.
        # ----------------------------------------------------

        pred_positions = sorted(
            predictions
        )

        # ----------------------------------------------------
        # Use only the number of lanes that can be
        # compared safely.
        # ----------------------------------------------------

        count = min(
            len(
                gt_positions
            ),
            len(
                pred_positions
            )
        )

        if count == 0:
            continue

        # ----------------------------------------------------
        # Ordered matching.
        #
        # CULane positions are sorted from left to right.
        # Our predictions are also sorted from left to right.
        #
        # This avoids arbitrary nearest-neighbour matching.
        # ----------------------------------------------------

        for i in range(count):

            error = abs(
                pred_positions[i]
                -
                gt_positions[i]
            )

            all_errors.append(
                error
            )

    return all_errors


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate():

    print()
    print("=" * 70)
    print(
        "ROBUST-LANENET V3 - "
        "CULANE EXTERNAL EVALUATION"
    )
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    print()
    print("Model:", MODEL_PATH)

    print()
    print(
        "Model input:",
        f"{IMAGE_WIDTH} x {IMAGE_HEIGHT}"
    )

    # --------------------------------------------------------
    # FIND DATA
    # --------------------------------------------------------

    samples = find_samples()

    print()
    print(
        "Valid CULane samples:",
        len(samples)
    )

    if len(samples) == 0:

        print()
        print(
            "ERROR: No valid CULane samples found."
        )

        return

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
        "Model loaded successfully."
    )

    print()
    print("-" * 70)

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    all_errors = []

    evaluated_images = 0
    skipped_images = 0

    max_samples = len(samples)

    with torch.no_grad():

        for index, (
            image_path,
            label_path
        ) in enumerate(samples):

            try:

                # ------------------------------------------------
                # LOAD IMAGE
                # ------------------------------------------------

                image = Image.open(
                    image_path
                ).convert("RGB")

                # ------------------------------------------------
                # PREPROCESS
                # ------------------------------------------------

                tensor = preprocess_image(
                    image
                )

                tensor = tensor.unsqueeze(
                    0
                ).to(
                    DEVICE
                )

                # ------------------------------------------------
                # MODEL PREDICTION
                # ------------------------------------------------

                prediction = model(
                    tensor
                )

                prediction = (
                    prediction
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )

                # Convert normalized x coordinates
                # to model pixel coordinates.
                prediction = (
                    prediction *
                    IMAGE_WIDTH
                )

                prediction = [
                    float(x)
                    for x in prediction
                ]

                # ------------------------------------------------
                # GROUND TRUTH
                # ------------------------------------------------

                lanes = read_lanes(
                    label_path
                )

                if len(lanes) == 0:

                    skipped_images += 1
                    continue

                # ------------------------------------------------
                # CALCULATE ERROR
                # ------------------------------------------------

                errors = calculate_sample_error(
                    prediction,
                    lanes
                )

                if len(errors) == 0:

                    skipped_images += 1
                    continue

                all_errors.extend(
                    errors
                )

                evaluated_images += 1

                # ------------------------------------------------
                # PROGRESS
                # ------------------------------------------------

                if (
                    evaluated_images % 100
                    == 0
                ):

                    print(
                        f"Processed: "
                        f"{evaluated_images}/"
                        f"{max_samples}"
                    )

            except Exception as error:

                skipped_images += 1

                print()
                print(
                    "Warning:",
                    image_path
                )

                print(
                    error
                )

    # --------------------------------------------------------
    # CHECK RESULTS
    # --------------------------------------------------------

    if len(all_errors) == 0:

        print()
        print(
            "ERROR: No valid comparisons."
        )

        return

    errors = np.array(
        all_errors,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    mean_error = float(
        np.mean(errors)
    )

    median_error = float(
        np.median(errors)
    )

    max_error = float(
        np.max(errors)
    )

    success_rate = float(
        np.mean(
            errors <= SUCCESS_THRESHOLD
        ) * 100.0
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "CULANE EXTERNAL EVALUATION RESULTS"
    )
    print("=" * 70)

    print()

    print(
        f"Evaluated images: "
        f"{evaluated_images}"
    )

    print(
        f"Skipped images:   "
        f"{skipped_images}"
    )

    print(
        f"Lane comparisons:  "
        f"{len(errors)}"
    )

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
    print(
        "CULANE EVALUATION COMPLETE"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    output_dir = "results/metrics"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    results = {

        "model":
            MODEL_PATH,

        "dataset":
            "CULane",

        "input_resolution":
            [
                IMAGE_WIDTH,
                IMAGE_HEIGHT
            ],

        "original_culane_resolution":
            [
                CULANE_WIDTH,
                CULANE_HEIGHT
            ],

        "resize_mode":
            "aspect_ratio_preserving_letterbox",

        "scale":
            SCALE,

        "padding":
            {
                "x": PAD_X,
                "y": PAD_Y
            },

        "evaluated_images":
            evaluated_images,

        "skipped_images":
            skipped_images,

        "lane_comparisons":
            len(errors),

        "mean_error_pixels":
            mean_error,

        "median_error_pixels":
            median_error,

        "maximum_error_pixels":
            max_error,

        "success_rate_5px":
            success_rate
    }

    output_path = os.path.join(
        output_dir,
        "culane_v3_corrected_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print()
    print(
        "Metrics saved to:"
    )

    print(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate()

