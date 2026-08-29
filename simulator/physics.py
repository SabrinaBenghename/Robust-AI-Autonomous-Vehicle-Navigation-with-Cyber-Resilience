import math


class PhysicsEngine:
    """
    Simulates vehicle kinematics, aerodynamic drag, rolling resistance, and collision checks.
    """

    def __init__(self, gravity: float = 9.81, air_density: float = 1.225):
        self.gravity = gravity
        self.air_density = air_density

    def calculate_forces(
        self,
        speed: float,
        throttle: float,
        brake: float,
        mass: float = 1500.0,
        drag_coeff: float = 0.3,
        frontal_area: float = 2.2,
        rolling_friction: float = 0.015,
    ) -> float:
        """
        Calculates net longitudinal forceacting on vehicle (Engine Force - Drag - Rolling Resistance - Braking).
        """
        max_engine_force = 4000.0
        engine_force = throttle * max_engine_force

        drag_force = 0.5 * self.air_density * drag_coeff * frontal_area * (speed ** 2)
        rolling_force = rolling_friction * mass * self.gravity
        braking_force = brake * 8000.0

        net_force = engine_force - drag_force - rolling_force - braking_force
        return net_force

    def update_kinematics(
        self,
        x: float,
        y: float,
        speed: float,
        angle_deg: float,
        steering_angle_deg: float,
        dt: float,
        wheelbase: float = 2.7,
    ) -> tuple[float, float, float, float]:
        """
        Bicycle model state update (X, Y, Speed, Heading Angle).
        """
        rad_heading = math.radians(angle_deg)
        rad_steering = math.radians(steering_angle_deg)

        dx = speed * math.sin(rad_heading) * dt
        dy = -speed * math.cos(rad_heading) * dt

        # Angular velocity based on kinematic bicycle model
        d_angle = math.degrees((speed / wheelbase) * math.tan(rad_steering)) * dt

        new_x = x + dx
        new_y = y + dy
        new_angle = (angle_deg + d_angle) % 360.0

        return new_x, new_y, speed, new_angle

    @staticmethod
    def check_bounding_box_collision(
        box1: tuple[float, float, float, float],
        box2: tuple[float, float, float, float]
    ) -> bool:
        """
        AABB Collision check (x, y, width, height).
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        return (
            x1 < x2 + w2 and
            x1 + w1 > x2 and
            y1 < y2 + h2 and
            y1 + h1 > y2
        )
