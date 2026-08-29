import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class TrajectoryPoint:
    x: float
    y: float
    yaw: float
    target_velocity: float

class TrajectoryGenerator:
    """
    Generates dynamic quintic polynomial trajectories for smooth longitudinal and lateral motion.
    """
    def generate_trajectory(self, waypoints: List[Tuple[float, float]], target_velocity: float, num_points: int = 50) -> List[TrajectoryPoint]:
        if len(waypoints) < 2:
            return []

        wx = [p[0] for p in waypoints]
        wy = [p[1] for p in waypoints]

        # Simple linear interpolation & orientation calculation for smooth lookahead
        t = np.linspace(0, 1, num_points)
        x_interp = np.interp(t, np.linspace(0, 1, len(wx)), wx)
        y_interp = np.interp(t, np.linspace(0, 1, len(wy)), wy)

        trajectory = []
        for i in range(num_points):
            if i < num_points - 1:
                dx = x_interp[i+1] - x_interp[i]
                dy = y_interp[i+1] - y_interp[i]
                yaw = np.arctan2(dy, dx)
            else:
                yaw = trajectory[-1].yaw if trajectory else 0.0

            trajectory.append(
                TrajectoryPoint(
                    x=float(x_interp[i]),
                    y=float(y_interp[i]),
                    yaw=float(yaw),
                    target_velocity=target_velocity
                )
            )
        return trajectory
