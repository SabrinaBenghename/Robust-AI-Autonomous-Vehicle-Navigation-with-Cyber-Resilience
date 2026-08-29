import os
import json
import pygame


class DatasetGenerator:

    def __init__(self):

        self.image_folder = "datasets/images"
        self.label_folder = "datasets/labels"

        os.makedirs(
            self.image_folder,
            exist_ok=True
        )

        os.makedirs(
            self.label_folder,
            exist_ok=True
        )

        self.frame_id = 0

    def save_frame(self, screen, lane_data):

        # -------------------------
        # IMAGE
        # -------------------------

        image_name = (
            f"frame_{self.frame_id:05d}.png"
        )

        image_path = os.path.join(
            self.image_folder,
            image_name
        )

        pygame.image.save(
            screen,
            image_path
        )

        # -------------------------
        # LABEL
        # -------------------------

        label_name = (
            f"frame_{self.frame_id:05d}.json"
        )

        label_path = os.path.join(
            self.label_folder,
            label_name
        )

        label = {
            "image": image_name,
            "lanes": lane_data
        }

        with open(
            label_path,
            "w"
        ) as file:

            json.dump(
                label,
                file,
                indent=4
            )

        self.frame_id += 1