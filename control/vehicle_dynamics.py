import numpy as np
from dataclasses import dataclass

@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    velocity: float = 0.0
    steering_angle: float = 0.0
    acceleration: float = 0.0

class KinematicBicycleModel:
    """
    Nonlinear Kinematic Bicycle Vehicle Dynamics Model.
    """
    def __init__(self, wheelbase: float = 2.7, dt: float = 0.1):
        self.wheelbase = wheelbase
        self.dt = dt

    def step(self, state: VehicleState, throttle_brake: float, steer_cmd: float) -> VehicleState:
        """
        Updates vehicle state over time delta dt.
        """
        # Clamp inputs
        accel = np.clip(throttle_brake, -8.0, 3.5)
        steer = np.clip(steer_cmd, -0.5236, 0.5236)

        # Update kinematic differential equations
        new_x = state.x + state.velocity * np.cos(state.yaw) * self.dt
        new_y = state.y + state.velocity * np.sin(state.yaw) * self.dt
        new_yaw = state.yaw + (state.velocity / self.wheelbase) * np.tan(steer) * self.dt
        new_velocity = max(0.0, state.velocity + accel * self.dt)

        # Normalize yaw angle to [-pi, pi]
        new_yaw = np.arctan2(np.sin(new_yaw), np.cos(new_yaw))

        return VehicleState(
            x=float(new_x),
            y=float(new_y),
            yaw=float(new_yaw),
            velocity=float(new_velocity),
            steering_angle=float(steer),
            acceleration=float(accel)
        )
