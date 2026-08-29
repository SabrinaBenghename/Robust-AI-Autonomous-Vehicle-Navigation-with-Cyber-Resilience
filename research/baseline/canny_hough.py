import os
import json

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

IMAGES_DIR = "datasets/images"
LABELS_DIR = "datasets/labels"

OUTPUT_DIR = "results/canny_hough"

os.makedirs(OUTPUT_DIR, exist_ok=True)


LANE_NAMES = [
    "road_left",
    "lane_line_1",
    "center_line",
    "lane_line_2",
    "road_right"
]


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(image_name):

    base_name = os.path.splitext(image_name)[0]

    label_path = os.path.join(
        LABELS_DIR,
        base_name + ".json"
    )

    with open(label_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["lanes"]


# ============================================================
# PREPROCESSING
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
# REGION OF INTEREST
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
# HOUGH TRANSFORM
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
# EXTRACT VERTICAL LINE POSITIONS
# ============================================================

def extract_vertical_positions(lines):

    if lines is None:
        return []

    positions = []

    for line in lines:

        x1, y1, x2, y2 = line[0]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # Ignore horizontal / diagonal lines.
        if dy == 0:
            continue

        # Verticality test.
        if dx <= 8:

            x = (x1 + x2) / 2

            positions.append(x)

    return positions


# ============================================================
# GROUP NEARBY POSITIONS
# ============================================================

def cluster_positions(
    positions,
    distance=20
):

    if not positions:
        return []

    positions = sorted(positions)

    clusters = [
        [positions[0]]
    ]

    for x in positions[1:]:

        if abs(
            x - np.mean(clusters[-1])
        ) <= distance:

            clusters[-1].append(x)

        else:

            clusters.append([x])

    return [
        float(np.mean(cluster))
        for cluster in clusters
    ]


# ============================================================
# SELECT FIVE LANE BOUNDARIES
# ============================================================

def estimate_lanes(
    positions,
    width
):

    if not positions:
        return None

    positions = sorted(positions)

    # Keep positions inside the road area.
    road_left_limit = width * 0.20
    road_right_limit = width * 0.80

    positions = [
        x
        for x in positions
        if road_left_limit <= x <= road_right_limit
    ]

    if not positions:
        return None

    # We expect five boundaries.
    #
    # In our controlled environment the expected
    # approximate positions are:
    #
    # 340, 490, 640, 790, 940
    #
    # We therefore use five spatial regions.

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
            key=lambda x: abs(x - target)
        )

        result.append(nearest)

        positions.remove(nearest)

    return result


# ============================================================
# CALCULATE ERROR
# ============================================================

def calculate_error(
    prediction,
    ground_truth
):

    errors = []

    for name, predicted in zip(
        LANE_NAMES,
        prediction
    ):

        actual = ground_truth[name]

        error = abs(
            predicted - actual
        )

        errors.append(error)

    mean_error = np.mean(errors)

    return errors, mean_error


# ============================================================
# DRAW RESULTS
# ============================================================

def draw_results(
    image,
    prediction,
    ground_truth
):

    output = image.copy()

    # Ground truth = BLUE
    for name in LANE_NAMES:

        x = int(
            ground_truth[name]
        )

        cv2.line(
            output,
            (x, 0),
            (x, image.shape[0]),
            (255, 0, 0),
            2
        )

    # Prediction = GREEN
    if prediction is not None:

        for x in prediction:

            cv2.line(
                output,
                (int(x), 0),
                (
                    int(x),
                    image.shape[0]
                ),
                (0, 255, 0),
                3
            )

    return output


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(image_name):

    image_path = os.path.join(
        IMAGES_DIR,
        image_name
    )

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

    if prediction is not None:

        errors, mean_error = calculate_error(
            prediction,
            ground_truth
        )

    else:

        errors = None
        mean_error = None

    result = draw_results(
        image,
        prediction,
        ground_truth
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        image_name
    )

    cv2.imwrite(
        output_path,
        result
    )

    return prediction, mean_error


# ============================================================
# MAIN
# ============================================================

def main():

    images = sorted([
        file
        for file in os.listdir(IMAGES_DIR)
        if file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ])

    all_errors = []

    successful = 0

    print()
    print("=" * 60)
    print("CANNY + HOUGH BASELINE EVALUATION")
    print("=" * 60)
    print()

    print(
        f"Images: {len(images)}"
    )

    print()

    for index, image_name in enumerate(images):

        result = process_image(
            image_name
        )

        if result is not None:

            prediction, mean_error = result

            if mean_error is not None:

                successful += 1

                all_errors.append(
                    mean_error
                )

        if (index + 1) % 25 == 0:

            print(
                f"Processed {index + 1}/{len(images)}"
            )

    print()

    # ========================================================
    # FINAL METRICS
    # ========================================================

    if all_errors:

        mean_error = np.mean(
            all_errors
        )

        median_error = np.median(
            all_errors
        )

        print(
            f"Successful detections: "
            f"{successful}/{len(images)}"
        )

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
            "No successful lane detections."
        )

    print()

    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()


if __name__ == "__main__":
    main()