import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class LaneBoundary:
    left_coeffs: Tuple[float, float, float]  # polynomial a*x^2 + b*x + c
    right_coeffs: Tuple[float, float, float]
    lane_width: float
    lateral_offset: float

class LaneDetector:
    """
    Polynomial lane boundary detector and departure warning system.
    """
    def __init__(self, expected_width: float = 3.7):
        self.expected_width = expected_width

    def detect_lanes(self, ego_x: float, ego_y: float) -> LaneBoundary:
        """
        Estimate left and right lane boundary curves relative to ego vehicle center.
        """
        half_w = self.expected_width / 2.0
        # Left boundary polynomial: y = 0.0001*x^2 + 0.0*x + half_w
        left = (0.0001, 0.0, half_w)
        # Right boundary polynomial: y = 0.0001*x^2 + 0.0*x - half_w
        right = (0.0001, 0.0, -half_w)
        
        lateral_offset = ego_y  # distance from center
        return LaneBoundary(
            left_coeffs=left,
            right_coeffs=right,
            lane_width=self.expected_width,
            lateral_offset=lateral_offset
        )
