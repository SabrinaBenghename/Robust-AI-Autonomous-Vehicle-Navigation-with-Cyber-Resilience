import torch
import torch.nn as nn


class LaneLoss(nn.Module):

    def __init__(
        self,
        geometry_weight=0.10
    ):
        super().__init__()

        self.coordinate_loss = nn.SmoothL1Loss()

        self.geometry_weight = geometry_weight

    def forward(
        self,
        predictions,
        targets
    ):

        # ====================================================
        # 1. COORDINATE LOSS
        # ====================================================
        #
        # Measures how close each predicted lane coordinate
        # is to the ground-truth coordinate.
        #

        coordinate_loss = self.coordinate_loss(
            predictions,
            targets
        )

        # ====================================================
        # 2. ORDERING LOSS
        # ====================================================
        #
        # We want:
        #
        # road_left
        #     <
        # lane_line_1
        #     <
        # center_line
        #     <
        # lane_line_2
        #     <
        # road_right
        #
        # Penalize predictions when this order is violated.
        #

        order_loss = (
            torch.relu(
                predictions[:, 0]
                - predictions[:, 1]
            )
            +
            torch.relu(
                predictions[:, 1]
                - predictions[:, 2]
            )
            +
            torch.relu(
                predictions[:, 2]
                - predictions[:, 3]
            )
            +
            torch.relu(
                predictions[:, 3]
                - predictions[:, 4]
            )
        ).mean()

        # ====================================================
        # 3. SPACING CONSISTENCY LOSS
        # ====================================================
        #
        # Calculate the spacing between neighboring lines.
        #

        pred_spacing = predictions[:, 1:] - predictions[:, :-1]

        target_spacing = targets[:, 1:] - targets[:, :-1]

        spacing_loss = self.coordinate_loss(
            pred_spacing,
            target_spacing
        )

        # ====================================================
        # 4. GEOMETRY LOSS
        # ====================================================

        geometry_loss = (
            order_loss
            +
            spacing_loss
        )

        # ====================================================
        # 5. TOTAL LOSS
        # ====================================================

        total_loss = (
            coordinate_loss
            +
            self.geometry_weight
            * geometry_loss
        )

        return total_loss