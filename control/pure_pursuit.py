import numpy as np
from typing import List

class PurePursuitController:
    """
    Geometric Pure Pursuit Controller for lateral path tracking and steering command computation.
    """
    def __init__(self, wheelbase: float = 2.7, lookahead_distance: float = 6.0):
        self.wheelbase = wheelbase
        self.lookahead_distance = lookahead_distance

    def compute_steering_angle(self, vehicle_x: float, vehicle_y: float, vehicle_yaw: float, trajectory: List) -> float:
        """
        Calculates front wheel steering angle delta needed to track target trajectory.
        """
        if not trajectory:
            return 0.0

        # Find target lookahead point along trajectory
        target_pt = None
        for pt in trajectory:
            dx = pt.x - vehicle_x
            dy = pt.y - vehicle_y
            dist = np.hypot(dx, dy)
            if dist >= self.lookahead_distance:
                target_pt = pt
                break

        if target_pt is None:
            target_pt = trajectory[-1]

        # Transform target point to vehicle relative coordinate frame
        dx = target_pt.x - vehicle_x
        dy = target_pt.y - vehicle_y
        
        # Vehicle frame rotation
        alpha = np.arctan2(dy, dx) - vehicle_yaw
        
        # Pure pursuit curvature formula: k = 2 * sin(alpha) / Ld
        lookahead_actual = max(np.hypot(dx, dy), 0.1)
        curvature = (2.0 * np.sin(alpha)) / lookahead_actual
        steering_angle = np.arctan(self.wheelbase * curvature)

        # Clamp max steer angle to +- 30 degrees (0.52 rad)
        return float(np.clip(steering_angle, -0.5236, 0.5236))
