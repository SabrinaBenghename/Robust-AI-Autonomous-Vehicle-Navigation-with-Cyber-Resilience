import math
import pygame


class Vehicle:

    def __init__(self):

        # ====================================================
        # POSITION
        # ====================================================

        self.x = 715.0
        self.y = 450.0

        # ====================================================
        # SIZE
        # ====================================================

        self.width = 40
        self.height = 70

        # ====================================================
        # VEHICLE GEOMETRY
        # ====================================================

        # Slightly shorter wheelbase = quicker lane changes
        self.wheelbase = 82.0

        # ====================================================
        # SPEED
        # ====================================================

        self.speed = 0.0

        self.max_speed = 11.0
        self.max_reverse_speed = 3.0

        # Faster acceleration
        self.acceleration = 0.30

        self.brake_strength = 0.42

        self.drag = 0.03

        # ====================================================
        # STEERING
        # ====================================================

        self.steering_angle = 0.0

        # More physical steering range
        self.max_steering_angle = 15.0

        # Faster wheel movement
        self.steering_speed = 0.90

        # ====================================================
        # HEADING
        # ====================================================

        self.angle = 0.0

        # ====================================================
        # COLLISION
        # ====================================================

        self.collided = False

        self.collision_cooldown = 0

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
    # COLLISION
    # ========================================================

    def register_collision(
        self,
        frames=30
    ):

        self.speed = 0.0

        self.collided = True

        self.collision_cooldown = max(
            self.collision_cooldown,
            int(frames)
        )

    # ========================================================
    # APPLY CONTROL
    # ========================================================

    def apply_control(
        self,
        throttle,
        brake,
        steering
    ):

        throttle = self.clamp(
            float(throttle),
            -1.0,
            1.0
        )

        brake = self.clamp(
            float(brake),
            0.0,
            1.0
        )

        steering = self.clamp(
            float(steering),
            -1.0,
            1.0
        )

        # ====================================================
        # COLLISION HOLD
        # ====================================================

        if self.collision_cooldown > 0:

            self.collision_cooldown -= 1

            throttle = 0.0
            brake = 1.0

        else:

            self.collided = False

        # ====================================================
        # THROTTLE
        # ====================================================

        if throttle != 0.0:

            self.speed += (
                throttle
                * self.acceleration
            )

        # ====================================================
        # BRAKING
        # ====================================================

        if brake > 0.0:

            brake_force = (
                brake
                * self.brake_strength
            )

            if self.speed > 0.0:

                self.speed -= min(
                    self.speed,
                    brake_force
                )

            elif self.speed < 0.0:

                self.speed += min(
                    abs(self.speed),
                    brake_force
                )

        # ====================================================
        # DRAG
        # ====================================================

        if self.speed > 0.0:

            self.speed -= min(
                self.speed,
                self.drag
            )

        elif self.speed < 0.0:

            self.speed += min(
                abs(self.speed),
                self.drag
            )

        # ====================================================
        # SPEED LIMIT
        # ====================================================

        self.speed = self.clamp(
            self.speed,
            -self.max_reverse_speed,
            self.max_speed
        )

        # ====================================================
        # TARGET WHEEL ANGLE
        # ====================================================

        target_wheel_angle = (
            steering
            * self.max_steering_angle
        )

        steering_difference = (
            target_wheel_angle
            - self.steering_angle
        )

        steering_difference = self.clamp(
            steering_difference,
            -self.steering_speed,
            self.steering_speed
        )

        self.steering_angle += (
            steering_difference
        )

        if (
            abs(self.steering_angle)
            < 0.005
        ):

            self.steering_angle = 0.0

        # ====================================================
        # KINEMATIC BICYCLE MODEL
        # ====================================================

        if abs(self.speed) > 0.001:

            wheel_rad = math.radians(
                self.steering_angle
            )

            yaw_rate = (
                self.speed
                / self.wheelbase
                * math.tan(
                    wheel_rad
                )
            )

            self.angle += math.degrees(
                yaw_rate
            )

            # ------------------------------------------------
            # NORMALIZE
            # ------------------------------------------------

            while self.angle > 180.0:

                self.angle -= 360.0

            while self.angle < -180.0:

                self.angle += 360.0

            # ------------------------------------------------
            # POSITION
            # ------------------------------------------------

            heading_rad = math.radians(
                self.angle
            )

            self.x += (
                math.sin(
                    heading_rad
                )
                * self.speed
            )

            self.y -= (
                math.cos(
                    heading_rad
                )
                * self.speed
            )

    # ========================================================
    # MANUAL
    # ========================================================

    def update_manual(self):

        keys = pygame.key.get_pressed()

        throttle = 0.0
        brake = 0.0
        steering = 0.0

        if (
            keys[pygame.K_UP]
            or keys[pygame.K_w]
        ):

            throttle = 1.0

        if (
            keys[pygame.K_DOWN]
            or keys[pygame.K_s]
        ):

            brake = 1.0

        if keys[pygame.K_LEFT]:

            steering = -1.0

        elif keys[pygame.K_RIGHT]:

            steering = 1.0

        self.apply_control(
            throttle,
            brake,
            steering
        )

    # ========================================================
    # RECT
    # ========================================================

    def get_rect(self):

        return pygame.Rect(
            int(
                self.x
                - self.width / 2
            ),
            int(
                self.y
                - self.height / 2
            ),
            self.width,
            self.height
        )

    # ========================================================
    # DRAW
    # ========================================================

    def draw(
        self,
        screen,
        camera
    ):

        vehicle_surface = pygame.Surface(
            (
                self.width,
                self.height
            ),
            pygame.SRCALPHA
        )

        if self.collided:

            color = (
                255,
                120,
                20
            )

        else:

            color = (
                230,
                70,
                20
            )

        pygame.draw.rect(
            vehicle_surface,
            color,
            (
                0,
                0,
                self.width,
                self.height
            ),
            border_radius=7
        )

        pygame.draw.rect(
            vehicle_surface,
            (
                35,
                45,
                65
            ),
            (
                5,
                8,
                self.width - 10,
                17
            ),
            border_radius=4
        )

        rotated = pygame.transform.rotate(
            vehicle_surface,
            -self.angle
        )

        screen_x, screen_y = camera.apply(
            (
                self.x,
                self.y
            )
        )

        rect = rotated.get_rect(
            center=(
                int(screen_x),
                int(screen_y)
            )
        )

        screen.blit(
            rotated,
            rect
        )