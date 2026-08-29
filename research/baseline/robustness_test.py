import os
import cv2
import numpy as np

from canny_hough import process_image


IMAGES_DIR = "datasets/images"

OUTPUT_DIR = "results/robustness"

CONDITIONS = [
    "noise",
    "blur",
    "low_contrast",
    "occlusion"
]


# ============================================================
# DEGRADATION FUNCTIONS
# ============================================================

def add_noise(image):
    noise = np.random.normal(
        0,
        25,
        image.shape
    ).astype(np.float32)

    noisy = image.astype(np.float32) + noise

    return np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)


def add_blur(image):
    return cv2.GaussianBlur(
        image,
        (15, 15),
        0
    )


def reduce_contrast(image):

    return cv2.convertScaleAbs(
        image,
        alpha=0.45,
        beta=60
    )


def add_occlusion(image):

    result = image.copy()

    height, width = image.shape[:2]

    # Occlude part of the road.
    x1 = int(width * 0.40)
    x2 = int(width * 0.60)

    y1 = int(height * 0.45)
    y2 = int(height * 0.75)

    cv2.rectangle(
        result,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        -1
    )

    return result


# ============================================================
# CONDITION SELECTOR
# ============================================================

def degrade(image, condition):

    if condition == "noise":
        return add_noise(image)

    if condition == "blur":
        return add_blur(image)

    if condition == "low_contrast":
        return reduce_contrast(image)

    if condition == "occlusion":
        return add_occlusion(image)

    return image


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

    print()
    print("=" * 65)
    print("ROBUSTNESS EXPERIMENT")
    print("=" * 65)
    print()

    print(
        f"Images: {len(images)}"
    )

    print()

    for condition in CONDITIONS:

        condition_dir = os.path.join(
            OUTPUT_DIR,
            condition
        )

        os.makedirs(
            condition_dir,
            exist_ok=True
        )

        print(
            f"Testing: {condition}"
        )

        # ----------------------------------------------------
        # Create degraded dataset
        # ----------------------------------------------------

        for index, image_name in enumerate(images):

            image_path = os.path.join(
                IMAGES_DIR,
                image_name
            )

            image = cv2.imread(
                image_path
            )

            if image is None:
                continue

            degraded_image = degrade(
                image,
                condition
            )

            output_path = os.path.join(
                condition_dir,
                image_name
            )

            cv2.imwrite(
                output_path,
                degraded_image
            )

        print(
            f"  Generated {len(images)} degraded images."
        )

    print()
    print("=" * 65)
    print("DEGRADATION DATASETS CREATED")
    print("=" * 65)
    print()

    print(
        f"Results stored in: {OUTPUT_DIR}"
    )

    print()

    print(
        "Next we will run Canny-Hough on these "
        "degraded datasets and calculate the errors."
    )


if __name__ == "__main__":
    main()