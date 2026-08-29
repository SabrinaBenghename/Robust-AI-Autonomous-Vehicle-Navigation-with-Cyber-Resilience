import os
import json
import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

LABELS_DIR = "datasets/labels"

ROBUSTNESS_DIR = "results/robustness"

CONDITIONS = [
    "noise",
    "blur",
    "low_contrast",
    "occlusion"
]

LANE_NAMES = [
    "road_left",
    "lane_line_1",
    "center_line",
    "lane_line_2",
    "road_right"
]


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(image_name):

    base_name = os.path.splitext(image_name)[0]

    label_path = os.path.join(
        LABELS_DIR,
        base_name + ".json"
    )

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data["lanes"]


# ============================================================
# CANNY
# ============================================================

def preprocess(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    return edges


# ============================================================
# ROI
# ============================================================

def region_of_interest(edges):

    height, width = edges.shape

    mask = np.zeros_like(edges)

    polygon = np.array(
        [[
            (0, height),
            (width, height),
            (width, int(height * 0.35)),
            (0, int(height * 0.35))
        ]],
        dtype=np.int32
    )

    cv2.fillPoly(
        mask,
        polygon,
        255
    )

    return cv2.bitwise_and(
        edges,
        mask
    )


# ============================================================
# HOUGH
# ============================================================

def detect_lines(edges):

    return cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=40,
        maxLineGap=20
    )


# ============================================================
# VERTICAL LINE EXTRACTION
# ============================================================

def extract_vertical_positions(lines):

    if lines is None:
        return []

    positions = []

    for line in lines:

        x1, y1, x2, y2 = line[0]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dy == 0:
            continue

        if dx <= 8:

            x = (x1 + x2) / 2

            positions.append(x)

    return positions


# ============================================================
# CLUSTER POSITIONS
# ============================================================

def cluster_positions(
    positions,
    distance=20
):

    if not positions:
        return []

    positions = sorted(
        positions
    )

    clusters = [
        [positions[0]]
    ]

    for x in positions[1:]:

        if abs(
            x - np.mean(
                clusters[-1]
            )
        ) <= distance:

            clusters[-1].append(x)

        else:

            clusters.append(
                [x]
            )

    return [
        float(
            np.mean(cluster)
        )
        for cluster in clusters
    ]


# ============================================================
# ESTIMATE LANES
# ============================================================

def estimate_lanes(
    positions,
    width
):

    if not positions:
        return None

    positions = sorted(
        positions
    )

    road_left_limit = width * 0.20
    road_right_limit = width * 0.80

    positions = [
        x
        for x in positions
        if road_left_limit <= x <= road_right_limit
    ]

    if not positions:
        return None

    expected = np.array([
        width * 0.265,
        width * 0.383,
        width * 0.500,
        width * 0.617,
        width * 0.735
    ])

    result = []

    for target in expected:

        if not positions:
            return None

        nearest = min(
            positions,
            key=lambda x: abs(
                x - target
            )
        )

        result.append(
            nearest
        )

        positions.remove(
            nearest
        )

    return result


# ============================================================
# EVALUATE ONE IMAGE
# ============================================================

def evaluate_image(
    image_path,
    image_name
):

    image = cv2.imread(
        image_path
    )

    if image is None:
        return None

    ground_truth = load_ground_truth(
        image_name
    )

    edges = preprocess(
        image
    )

    roi = region_of_interest(
        edges
    )

    lines = detect_lines(
        roi
    )

    positions = extract_vertical_positions(
        lines
    )

    positions = cluster_positions(
        positions
    )

    prediction = estimate_lanes(
        positions,
        image.shape[1]
    )

    if prediction is None:
        return None

    errors = []

    for name, predicted in zip(
        LANE_NAMES,
        prediction
    ):

        actual = ground_truth[name]

        errors.append(
            abs(
                predicted - actual
            )
        )

    return np.mean(
        errors
    )


# ============================================================
# EVALUATE CONDITION
# ============================================================

def evaluate_condition(
    condition
):

    directory = os.path.join(
        ROBUSTNESS_DIR,
        condition
    )

    images = sorted([
        file
        for file in os.listdir(
            directory
        )
        if file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ])

    errors = []

    for image_name in images:

        image_path = os.path.join(
            directory,
            image_name
        )

        error = evaluate_image(
            image_path,
            image_name
        )

        if error is not None:

            errors.append(
                error
            )

    total = len(images)
    successful = len(errors)

    if errors:

        mean_error = np.mean(
            errors
        )

        median_error = np.median(
            errors
        )

    else:

        mean_error = None
        median_error = None

    return (
        total,
        successful,
        mean_error,
        median_error
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("CANNY + HOUGH ROBUSTNESS EVALUATION")
    print("=" * 70)
    print()

    print(
        f"{'Condition':<18}"
        f"{'Success':<15}"
        f"{'Mean Error':<18}"
        f"{'Median Error'}"
    )

    print("-" * 70)

    results = {}

    for condition in CONDITIONS:

        (
            total,
            successful,
            mean_error,
            median_error
        ) = evaluate_condition(
            condition
        )

        results[condition] = {
            "total": total,
            "successful": successful,
            "detection_rate": (
                successful / total * 100
                if total > 0
                else 0
            ),
            "mean_error": (
                float(mean_error)
                if mean_error is not None
                else None
            ),
            "median_error": (
                float(median_error)
                if median_error is not None
                else None
            )
        }

        if mean_error is not None:

            print(
                f"{condition:<18}"
                f"{successful}/{total:<15}"
                f"{mean_error:<18.2f}"
                f"{median_error:.2f}"
            )

        else:

            print(
                f"{condition:<18}"
                f"{successful}/{total:<15}"
                f"No detections"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    os.makedirs(
        "results/metrics",
        exist_ok=True
    )

    output_path = (
        "results/metrics/"
        "canny_hough_robustness.json"
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
    print("=" * 70)
    print("Evaluation complete.")
    print()
    print(
        f"Metrics saved to: {output_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()