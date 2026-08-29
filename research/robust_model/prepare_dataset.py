import os
import json
import random
import shutil


IMAGES_DIR = "datasets/images"
LABELS_DIR = "datasets/labels"

OUTPUT_DIR = "datasets/robust_lane"

LANE_NAMES = [
    "road_left",
    "lane_line_1",
    "center_line",
    "lane_line_2",
    "road_right"
]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


def load_samples():

    samples = []

    images = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ])

    for image_name in images:

        base_name = os.path.splitext(
            image_name
        )[0]

        label_path = os.path.join(
            LABELS_DIR,
            base_name + ".json"
        )

        if not os.path.exists(label_path):
            continue

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        lanes = data["lanes"]

        coordinates = [
            lanes[name]
            for name in LANE_NAMES
        ]

        samples.append({
            "image": image_name,
            "lanes": coordinates
        })

    return samples


def create_directories():

    for split in [
        "train",
        "val",
        "test"
    ]:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                split,
                "images"
            ),
            exist_ok=True
        )

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                split,
                "labels"
            ),
            exist_ok=True
        )


def save_sample(sample, split):

    image_name = sample["image"]

    source_image = os.path.join(
        IMAGES_DIR,
        image_name
    )

    destination_image = os.path.join(
        OUTPUT_DIR,
        split,
        "images",
        image_name
    )

    shutil.copy2(
        source_image,
        destination_image
    )

    base_name = os.path.splitext(
        image_name
    )[0]

    label_path = os.path.join(
        OUTPUT_DIR,
        split,
        "labels",
        base_name + ".json"
    )

    with open(
        label_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "image": image_name,
                "lanes": sample["lanes"]
            },
            file,
            indent=4
        )


def main():

    random.seed(SEED)

    samples = load_samples()

    random.shuffle(samples)

    total = len(samples)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = train_end + int(
        total * VAL_RATIO
    )

    train_samples = samples[:train_end]

    val_samples = samples[
        train_end:val_end
    ]

    test_samples = samples[
        val_end:
    ]

    create_directories()

    for sample in train_samples:
        save_sample(sample, "train")

    for sample in val_samples:
        save_sample(sample, "val")

    for sample in test_samples:
        save_sample(sample, "test")

    print()
    print("=" * 60)
    print("ROBUST MODEL DATASET")
    print("=" * 60)

    print()
    print(f"Total:      {total}")
    print(f"Training:   {len(train_samples)}")
    print(f"Validation: {len(val_samples)}")
    print(f"Testing:    {len(test_samples)}")

    print()
    print("Dataset created at:")
    print(OUTPUT_DIR)
    print()


if __name__ == "__main__":
    main()