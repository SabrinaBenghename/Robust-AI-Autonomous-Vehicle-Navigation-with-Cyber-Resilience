import os
import json
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

IMAGES_DIR = "datasets/images"
LABELS_DIR = "datasets/labels"

REQUIRED_LANES = [
    "road_left",
    "lane_line_1",
    "center_line",
    "lane_line_2",
    "road_right"
]


# ============================================================
# VALIDATION
# ============================================================

valid = []
invalid = []

image_files = [
    f
    for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]


for image_name in sorted(image_files):

    base_name = os.path.splitext(image_name)[0]

    image_path = os.path.join(
        IMAGES_DIR,
        image_name
    )

    label_path = os.path.join(
        LABELS_DIR,
        base_name + ".json"
    )

    errors = []

    # --------------------------------------------------------
    # CHECK LABEL EXISTS
    # --------------------------------------------------------

    if not os.path.exists(label_path):

        errors.append(
            "Missing JSON label"
        )

    else:

        try:

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

        except Exception:

            errors.append(
                "Invalid JSON"
            )
            data = None

        # ----------------------------------------------------
        # CHECK STRUCTURE
        # ----------------------------------------------------

        if data is not None:

            if "lanes" not in data:

                errors.append(
                    "Missing 'lanes'"
                )

            else:

                lanes = data["lanes"]

                # Check all required coordinates
                for lane_name in REQUIRED_LANES:

                    if lane_name not in lanes:

                        errors.append(
                            f"Missing {lane_name}"
                        )

                # ------------------------------------------------
                # CHECK IMAGE
                # ------------------------------------------------

                try:

                    with Image.open(
                        image_path
                    ) as image:

                        image_width, image_height = (
                            image.size
                        )

                except Exception:

                    errors.append(
                        "Cannot read image"
                    )

                    image_width = 0
                    image_height = 0

                # ------------------------------------------------
                # CHECK COORDINATES
                # ------------------------------------------------

                if image_width > 0:

                    coordinates = [
                        lanes.get(name)
                        for name in REQUIRED_LANES
                    ]

                    if all(
                        isinstance(x, (int, float))
                        for x in coordinates
                    ):

                        for name, x in zip(
                            REQUIRED_LANES,
                            coordinates
                        ):

                            if (
                                x < 0
                                or x > image_width
                            ):

                                errors.append(
                                    f"{name} outside image"
                                )

                        # ----------------------------------------
                        # CHECK ORDER
                        # ----------------------------------------

                        if not (
                            lanes["road_left"]
                            < lanes["lane_line_1"]
                            < lanes["center_line"]
                            < lanes["lane_line_2"]
                            < lanes["road_right"]
                        ):

                            errors.append(
                                "Lane coordinates are not ordered"
                            )

                        # ----------------------------------------
                        # CHECK SPACING
                        # ----------------------------------------

                        spacing = [
                            lanes["lane_line_1"]
                            - lanes["road_left"],

                            lanes["center_line"]
                            - lanes["lane_line_1"],

                            lanes["lane_line_2"]
                            - lanes["center_line"],

                            lanes["road_right"]
                            - lanes["lane_line_2"]
                        ]

                        # For our current simulator,
                        # lane spacing should be approximately 150 px.

                        for value in spacing:

                            if abs(value - 150) > 5:

                                errors.append(
                                    "Unexpected lane spacing"
                                )
                                break

    # ========================================================
    # RESULT
    # ========================================================

    if errors:

        invalid.append(
            (
                image_name,
                errors
            )
        )

    else:

        valid.append(image_name)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 55)
print("DATASET VALIDATION")
print("=" * 55)
print()

print(
    f"Images found: {len(image_files)}"
)

print(
    f"Valid samples: {len(valid)}"
)

print(
    f"Invalid samples: {len(invalid)}"
)

print()

# ============================================================
# INVALID SAMPLES
# ============================================================

if invalid:

    print("INVALID SAMPLES:")
    print()

    for image_name, errors in invalid:

        print(
            f"{image_name}:"
        )

        for error in errors:

            print(
                f"    - {error}"
            )

        print()

else:

    print(
        "ALL SAMPLES PASSED VALIDATION."
    )

print()

print("=" * 55)