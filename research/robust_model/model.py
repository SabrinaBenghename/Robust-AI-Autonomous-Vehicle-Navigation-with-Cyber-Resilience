import torch
import torch.nn as nn

from blocks import DepthwiseSeparableConv, WRFEB


class RobustLaneNet(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------

        self.backbone = nn.Sequential(

            DepthwiseSeparableConv(
                3,
                32,
                stride=2
            ),

            DepthwiseSeparableConv(
                32,
                64,
                stride=2
            ),

            DepthwiseSeparableConv(
                64,
                128,
                stride=2
            )
        )

        # ----------------------------------------
        # WEATHER-ROBUST FEATURE ENHANCEMENT
        # ----------------------------------------

        self.wrfeb = WRFEB(
            128
        )

        # ----------------------------------------
        # GLOBAL FEATURE REPRESENTATION
        # ----------------------------------------

        self.pool = nn.AdaptiveAvgPool2d(
            1
        )

        # ----------------------------------------
        # LANE PREDICTION HEAD
        # ----------------------------------------

        self.head = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                64,
                5
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        # Extract visual features
        x = self.backbone(x)

        # Enhance robust features
        x = self.wrfeb(x)

        # Compress spatial information
        x = self.pool(x)

        # Predict five normalized lane positions
        x = self.head(x)

        return x


# ============================================================
# TEST THE MODEL
# ============================================================

if __name__ == "__main__":

    model = RobustLaneNet()

    print()
    print("=" * 60)
    print("ROBUST-LANENET")
    print("=" * 60)

    print(model)

    # Fake image:
    # batch = 2
    # channels = 3
    # height = 720
    # width = 1280

    test_input = torch.randn(
        2,
        3,
        720,
        1280
    )

    output = model(
        test_input
    )

    print()
    print(
        "Input shape:",
        test_input.shape
    )

    print(
        "Output shape:",
        output.shape
    )

    print()
    print("Example prediction:")
    print(output[0])

    print()
    print("=" * 60)
    print("FORWARD PASS SUCCESSFUL")
    print("=" * 60)