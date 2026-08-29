import os
import json
import random
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_IMAGES = "datasets/images"
SOURCE_LABELS = "datasets/labels"

OUTPUT = "datasets/processed"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


# ============================================================
# CREATE DIRECTORIES
# ============================================================

splits = [
    "train",
    "validation",
    "test"
]

for split in splits:

    os.makedirs(
        os.path.join(
            OUTPUT,
            split,
            "images"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            OUTPUT,
            split,
            "labels"
        ),
        exist_ok=True
    )


# ============================================================
# FIND IMAGES
# ============================================================

images = [
    file
    for file in os.listdir(SOURCE_IMAGES)
    if file.lower().endswith(".png")
]


# ============================================================
# CHECK DATA
# ============================================================

valid_images = []

for image in images:

    label_name = (
        os.path.splitext(image)[0]
        + ".json"
    )

    label_path = os.path.join(
        SOURCE_LABELS,
        label_name
    )

    if os.path.exists(label_path):

        valid_images.append(image)


print(
    f"Found {len(valid_images)} valid image/label pairs."
)


# ============================================================
# SHUFFLE
# ============================================================

random.seed(SEED)

random.shuffle(
    valid_images
)


# ============================================================
# SPLIT
# ============================================================

total = len(valid_images)

train_end = int(
    total * TRAIN_RATIO
)

val_end = train_end + int(
    total * VAL_RATIO
)

train_images = valid_images[
    :train_end
]

val_images = valid_images[
    train_end:val_end
]

test_images = valid_images[
    val_end:
]


# ============================================================
# COPY DATA
# ============================================================

def copy_split(
    image_list,
    split_name
):

    for image_name in image_list:

        label_name = (
            os.path.splitext(image_name)[0]
            + ".json"
        )

        source_image = os.path.join(
            SOURCE_IMAGES,
            image_name
        )

        source_label = os.path.join(
            SOURCE_LABELS,
            label_name
        )

        destination_image = os.path.join(
            OUTPUT,
            split_name,
            "images",
            image_name
        )

        destination_label = os.path.join(
            OUTPUT,
            split_name,
            "labels",
            label_name
        )

        shutil.copy2(
            source_image,
            destination_image
        )

        shutil.copy2(
            source_label,
            destination_label
        )


# ============================================================
# EXECUTE
# ============================================================

copy_split(
    train_images,
    "train"
)

copy_split(
    val_images,
    "validation"
)

copy_split(
    test_images,
    "test"
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("Dataset successfully prepared.")
print()

print(
    f"Training:   {len(train_images)}"
)

print(
    f"Validation: {len(val_images)}"
)

print(
    f"Testing:    {len(test_images)}"
)

print()

print(
    f"Total:      {total}"
)