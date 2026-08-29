import os
import sys
import cv2
import json
import numpy as np

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Add project root to Python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# IMPORT ROBUST-LANENET
# ============================================================

from ai.perception import RobustLanePerception


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "robust_lane"
)

TEST_IMAGES_DIR = os.path.join(
    DATASET_DIR,
    "test",
    "images"
)

TEST_LABELS_DIR = os.path.join(
    DATASET_DIR,
    "test",
    "labels"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "robustness_evaluation"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# ROBUSTNESS SETTINGS
# ============================================================

ROBUSTNESS_CONDITIONS = [
    "CLEAN",
    "NOISE",
    "BLUR",
    "LOW_CONTRAST",
    "OCCLUSION"
]


# ============================================================
# IMAGE CORRUPTION
# ============================================================

def apply_robustness_test(image, condition):

    if image is None:
        return None

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    if condition == "CLEAN":

        return image.copy()

    # --------------------------------------------------------
    # GAUSSIAN NOISE
    # --------------------------------------------------------

    if condition == "NOISE":

        noisy = image.astype(
            np.float32
        )

        noise = np.random.normal(
            0,
            25,
            noisy.shape
        )

        noisy = noisy + noise

        noisy = np.clip(
            noisy,
            0,
            255
        )

        return noisy.astype(
            np.uint8
        )

    # --------------------------------------------------------
    # BLUR
    # --------------------------------------------------------

    if condition == "BLUR":

        return cv2.GaussianBlur(
            image,
            (21, 21),
            0
        )

    # --------------------------------------------------------
    # LOW CONTRAST
    # --------------------------------------------------------

    if condition == "LOW_CONTRAST":

        low_contrast = (
            image.astype(
                np.float32
            )
            * 0.45
            + 70
        )

        low_contrast = np.clip(
            low_contrast,
            0,
            255
        )

        return low_contrast.astype(
            np.uint8
        )

    # --------------------------------------------------------
    # OCCLUSION
    # --------------------------------------------------------

    if condition == "OCCLUSION":

        occluded = image.copy()

        height, width = (
            occluded.shape[:2]
        )

        x1 = int(
            width * 0.35
        )

        x2 = int(
            width * 0.65
        )

        y1 = int(
            height * 0.45
        )

        y2 = int(
            height * 0.75
        )

        cv2.rectangle(
            occluded,
            (x1, y1),
            (x2, y2),
            (0, 0, 0),
            -1
        )

        return occluded

    return image.copy()


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(label_path):

    if not os.path.exists(label_path):

        return None

    try:

        with open(
            label_path,
            "r"
        ) as f:

            data = json.load(f)

        # ----------------------------------------------------
        # Your labels contain the lane coordinates.
        # We try common possible field names.
        # ----------------------------------------------------

        if isinstance(data, dict):

            if "lane_x" in data:

                return np.array(
                    data["lane_x"],
                    dtype=np.float32
                )

            if "lanes" in data:

                return np.array(
                    data["lanes"],
                    dtype=np.float32
                )

            if "coordinates" in data:

                return np.array(
                    data["coordinates"],
                    dtype=np.float32
                )

        # ----------------------------------------------------
        # If the JSON itself is a list
        # ----------------------------------------------------

        if isinstance(data, list):

            return np.array(
                data,
                dtype=np.float32
            )

    except Exception as e:

        print(
            "Could not read label:",
            label_path
        )

        print(
            "Error:",
            e
        )

    return None


# ============================================================
# CALCULATE ERROR
# ============================================================

def calculate_lane_error(
    prediction,
    ground_truth
):

    if prediction is None:
        return None

    if ground_truth is None:
        return None

    prediction = np.asarray(
        prediction,
        dtype=np.float32
    )

    ground_truth = np.asarray(
        ground_truth,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Make sure both arrays have same length
    # --------------------------------------------------------

    count = min(
        len(prediction),
        len(ground_truth)
    )

    if count == 0:
        return None

    prediction = prediction[:count]

    ground_truth = ground_truth[:count]

    error = np.abs(
        prediction - ground_truth
    )

    return float(
        np.mean(error)
    )


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print(
    "ROBUST-LANENET V3 — ROBUSTNESS EVALUATION"
)
print("=" * 70)
print()

print(
    "Project root:"
)
print(
    PROJECT_ROOT
)

print()

print(
    "Test images:"
)
print(
    TEST_IMAGES_DIR
)

print()

print(
    "Test labels:"
)
print(
    TEST_LABELS_DIR
)

print()

print(
    "Results:"
)
print(
    RESULTS_DIR
)

print()


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(
    TEST_IMAGES_DIR
):

    raise FileNotFoundError(
        f"Test images directory not found:\n"
        f"{TEST_IMAGES_DIR}"
    )

if not os.path.exists(
    TEST_LABELS_DIR
):

    raise FileNotFoundError(
        f"Test labels directory not found:\n"
        f"{TEST_LABELS_DIR}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

perception = RobustLanePerception()


# ============================================================
# FIND TEST IMAGES
# ============================================================

image_files = []

for filename in os.listdir(
    TEST_IMAGES_DIR
):

    if filename.lower().endswith(
        (
            ".png",
            ".jpg",
            ".jpeg"
        )
    ):

        image_files.append(
            os.path.join(
                TEST_IMAGES_DIR,
                filename
            )
        )

image_files.sort()


print(
    "Test images found:",
    len(image_files)
)

print()


# ============================================================
# EVALUATE EACH ROBUSTNESS CONDITION
# ============================================================

all_results = {}


for condition in ROBUSTNESS_CONDITIONS:

    print()
    print("=" * 70)
    print(
        "TEST CONDITION:",
        condition
    )
    print("=" * 70)
    print()

    successful = 0
    failed = 0

    errors = []

    for index, image_path in enumerate(
        image_files
    ):

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = cv2.imread(
            image_path
        )

        if image is None:

            failed += 1

            continue

        # ----------------------------------------------------
        # APPLY CORRUPTION
        # ----------------------------------------------------

        test_image = (
            apply_robustness_test(
                image,
                condition
            )
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = (
            perception.predict(
                test_image
            )
        )

        if prediction is None:

            failed += 1

            continue

        successful += 1

        # ----------------------------------------------------
        # FIND LABEL
        # ----------------------------------------------------

        filename = os.path.basename(
            image_path
        )

        label_name = (
            os.path.splitext(
                filename
            )[0]
            + ".json"
        )

        label_path = os.path.join(
            TEST_LABELS_DIR,
            label_name
        )

        ground_truth = (
            load_ground_truth(
                label_path
            )
        )

        # ----------------------------------------------------
        # ERROR
        # ----------------------------------------------------

        error = (
            calculate_lane_error(
                prediction,
                ground_truth
            )
        )

        if error is not None:

            errors.append(
                error
            )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if index % 10 == 0:

            if error is not None:

                print(
                    f"[{index + 1}/"
                    f"{len(image_files)}] "
                    f"Error: "
                    f"{error:.2f} px"
                )

            else:

                print(
                    f"[{index + 1}/"
                    f"{len(image_files)}] "
                    f"Prediction successful"
                )

    # ========================================================
    # CONDITION RESULTS
    # ========================================================

    if len(errors) > 0:

        mean_error = float(
            np.mean(errors)
        )

        median_error = float(
            np.median(errors)
        )

    else:

        mean_error = None
        median_error = None

    all_results[condition] = {
        "total": len(image_files),
        "successful": successful,
        "failed": failed,
        "mean_error": mean_error,
        "median_error": median_error
    }

    print()
    print(
        "Condition:",
        condition
    )

    print(
        "Total:",
        len(image_files)
    )

    print(
        "Successful:",
        successful
    )

    print(
        "Failed:",
        failed
    )

    if mean_error is not None:

        print(
            f"Mean lane error: "
            f"{mean_error:.2f} pixels"
        )

        print(
            f"Median lane error: "
            f"{median_error:.2f} pixels"
        )

    else:

        print(
            "No ground-truth errors calculated."
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    RESULTS_DIR,
    "robustness_results.json"
)

with open(
    results_file,
    "w"
) as f:

    json.dump(
        all_results,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 70)
print(
    "ROBUST-LANENET V3 — FINAL ROBUSTNESS RESULTS"
)
print("=" * 70)
print()

for condition, result in (
    all_results.items()
):

    print(
        f"{condition:15s} | "
        f"Success: "
        f"{result['successful']:3d}/"
        f"{result['total']:3d} | "
        f"Failed: "
        f"{result['failed']:3d}"
    )

    if result["mean_error"] is not None:

        print(
            f"{'':15s} | "
            f"Mean error: "
            f"{result['mean_error']:.2f} px | "
            f"Median: "
            f"{result['median_error']:.2f} px"
        )

print()

print(
    "Results saved to:"
)

print(
    results_file
)

print()

print("=" * 70)
print(
    "ROBUST-LANENET V3 ROBUSTNESS EVALUATION COMPLETE"
)
print("=" * 70)
print()