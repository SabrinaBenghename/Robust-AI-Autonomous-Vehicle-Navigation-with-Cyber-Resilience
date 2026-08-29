import math


class VehicleController:

    def __init__(self):

        # ====================================================
        # SPEED
        # ====================================================

        self.target_speed = 8.8

        self.avoidance_speed = 7.4

        self.slow_speed = 3.2

        # ====================================================
        # LANE FOLLOWING
        # ====================================================

        self.lookahead_base = 155.0

        self.max_target_heading = 14.0

        self.heading_gain = 0.052

        self.max_lane_steering = 0.55

        self.lane_deadband = 1.5

        # ====================================================
        # FILTERING
        # ====================================================

        self.target_filter = 0.45

        self.steering_filter = 0.45

        self.max_steering_change = 0.075

        self.smoothed_target_x = None

        self.last_steering = 0.0

        # ====================================================
        # DEBUG
        # ====================================================

        self.last_lane_error = 0.0

        self.last_desired_heading = 0.0

        self.last_heading_error = 0.0

    # ========================================================
    # CLAMP
    # ========================================================

    @staticmethod
    def clamp(
        value,
        minimum,
        maximum
    ):

        return max(
            minimum,
            min(
                value,
                maximum
            )
        )

    # ========================================================
    # NORMALIZE ANGLE
    # ========================================================

    @staticmethod
    def normalize_angle(
        angle
    ):

        while angle > 180.0:

            angle -= 360.0

        while angle < -180.0:

            angle += 360.0

        return angle

    # ========================================================
    # TARGET FILTER
    # ========================================================

    def filter_target(
        self,
        target_x
    ):

        target_x = float(
            target_x
        )

        if self.smoothed_target_x is None:

            self.smoothed_target_x = (
                target_x
            )

        else:

            self.smoothed_target_x += (
                self.target_filter
                * (
                    target_x
                    - self.smoothed_target_x
                )
            )

        return (
            self.smoothed_target_x
        )

    # ========================================================
    # LANE CONTROL
    # ========================================================

    def lane_control(
        self,
        vehicle,
        target_x
    ):

        target_x = (
            self.filter_target(
                target_x
            )
        )

        # ====================================================
        # LATERAL ERROR
        # ====================================================

        error = (
            target_x
            - float(vehicle.x)
        )

        self.last_lane_error = (
            error
        )

        # ====================================================
        # LOOKAHEAD
        # ====================================================

        lookahead = (
            self.lookahead_base
            + abs(
                float(vehicle.speed)
            ) * 8.0
        )

        # ====================================================
        # TARGET HEADING
        # ====================================================

        if (
            abs(error)
            <= self.lane_deadband
        ):

            desired_heading = 0.0

        else:

            desired_heading = math.degrees(
                math.atan2(
                    error,
                    lookahead
                )
            )

        desired_heading = self.clamp(
            desired_heading,
            -self.max_target_heading,
            self.max_target_heading
        )

        self.last_desired_heading = (
            desired_heading
        )

        # ====================================================
        # HEADING ERROR
        # ====================================================

        heading_error = (
            desired_heading
            - float(vehicle.angle)
        )

        heading_error = (
            self.normalize_angle(
                heading_error
            )
        )

        self.last_heading_error = (
            heading_error
        )

        # ====================================================
        # STEERING
        # ====================================================

        desired_steering = (
            heading_error
            * self.heading_gain
        )

        desired_steering = self.clamp(
            desired_steering,
            -self.max_lane_steering,
            self.max_lane_steering
        )

        # ====================================================
        # SOFTEN CLOSE TO LANE CENTER
        # ====================================================

        if abs(error) < 25.0:

            scale = max(
                0.22,
                abs(error) / 25.0
            )

            desired_steering *= (
                scale
            )

        # ====================================================
        # CENTERED
        # ====================================================

        if (
            abs(error)
            <= self.lane_deadband
            and abs(
                float(vehicle.angle)
            ) < 0.25
        ):

            desired_steering = 0.0

        # ====================================================
        # FILTER
        # ====================================================

        filtered = (
            self.last_steering
            + self.steering_filter
            * (
                desired_steering
                - self.last_steering
            )
        )

        # ====================================================
        # RATE LIMIT
        # ====================================================

        change = (
            filtered
            - self.last_steering
        )

        change = self.clamp(
            change,
            -self.max_steering_change,
            self.max_steering_change
        )

        steering = (
            self.last_steering
            + change
        )

        self.last_steering = (
            steering
        )

        return steering

    # ========================================================
    # SPEED CONTROL
    # ========================================================

    def speed_control(
        self,
        vehicle,
        desired_speed
    ):

        speed = abs(
            float(vehicle.speed)
        )

        throttle = 0.0
        brake = 0.0

        if (
            speed
            < desired_speed - 0.30
        ):

            throttle = 1.0

        elif (
            speed
            > desired_speed + 0.40
        ):

            brake = 0.24

        else:

            throttle = 0.14

        return (
            throttle,
            brake
        )

    # ========================================================
    # MAIN CONTROL
    # ========================================================

    def control(
        self,
        decision,
        vehicle,
        target_x=None
    ):

        # ====================================================
        # SPEED MODE
        # ====================================================

        if decision in (
            "LEFT",
            "RIGHT"
        ):

            desired_speed = (
                self.avoidance_speed
            )

        elif decision == "SLOW":

            desired_speed = (
                self.slow_speed
            )

        else:

            desired_speed = (
                self.target_speed
            )

        throttle, brake = (
            self.speed_control(
                vehicle,
                desired_speed
            )
        )

        # ====================================================
        # LANE TRACKING
        # ====================================================

        if target_x is not None:

            steering = (
                self.lane_control(
                    vehicle,
                    target_x
                )
            )

        else:

            steering = 0.0

        # ====================================================
        # EMERGENCY STOP
        # ====================================================

        if decision == "STOP":

            throttle = 0.0
            brake = 1.0

        return {

            "throttle": self.clamp(
                throttle,
                -1.0,
                1.0
            ),

            "brake": self.clamp(
                brake,
                0.0,
                1.0
            ),

            "steering": self.clamp(
                steering,
                -1.0,
                1.0
            )
        }