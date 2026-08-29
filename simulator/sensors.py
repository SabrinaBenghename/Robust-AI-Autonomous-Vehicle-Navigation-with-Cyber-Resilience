import math


class DistanceSensor:

    def __init__(
        self,
        angle_offset,
        max_distance=1500.0,
        detection_angle=30.0
    ):

        self.angle_offset = float(
            angle_offset
        )

        self.max_distance = float(
            max_distance
        )

        self.detection_angle = float(
            detection_angle
        )

        self.distance = (
            self.max_distance
        )

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        vehicle,
        obstacles
    ):

        sensor_angle = (
            float(vehicle.angle)
            + self.angle_offset
        )

        origin_x = float(
            vehicle.x
        )

        origin_y = float(
            vehicle.y
        )

        closest = (
            self.max_distance
        )

        for obstacle in obstacles:

            dx = (
                float(obstacle.x)
                - origin_x
            )

            dy = (
                float(obstacle.y)
                - origin_y
            )

            distance = math.hypot(
                dx,
                dy
            )

            if distance > self.max_distance:

                continue

            # =================================================
            # DIRECTION TO OBSTACLE
            # =================================================

            obstacle_angle = math.degrees(
                math.atan2(
                    dx,
                    -dy
                )
            )

            difference = (
                obstacle_angle
                - sensor_angle
            )

            while difference > 180.0:

                difference -= 360.0

            while difference < -180.0:

                difference += 360.0

            # =================================================
            # SENSOR CONE
            # =================================================

            if (
                abs(difference)
                <= self.detection_angle
            ):

                closest = min(
                    closest,
                    distance
                )

        self.distance = closest

    # ========================================================
    # GET
    # ========================================================

    def get_measurement(self):

        return self.distance


class SensorSuite:

    def __init__(self):

        # ====================================================
        # FRONT
        # ====================================================

        self.front = DistanceSensor(
            angle_offset=0.0,
            max_distance=1500.0,
            detection_angle=30.0
        )

        # ====================================================
        # LEFT
        # ====================================================

        self.left = DistanceSensor(
            angle_offset=-50.0,
            max_distance=1500.0,
            detection_angle=34.0
        )

        # ====================================================
        # RIGHT
        # ====================================================

        self.right = DistanceSensor(
            angle_offset=50.0,
            max_distance=1500.0,
            detection_angle=34.0
        )

        self.sensors = [
            self.front,
            self.left,
            self.right
        ]

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        vehicle,
        obstacles
    ):

        for sensor in self.sensors:

            sensor.update(
                vehicle,
                obstacles
            )

    # ========================================================
    # DATA
    # ========================================================

    def get_data(self):

        return {
            "front": (
                self.front.get_measurement()
            ),
            "left": (
                self.left.get_measurement()
            ),
            "right": (
                self.right.get_measurement()
            )
        }