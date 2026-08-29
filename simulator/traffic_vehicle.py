import pygame


class TrafficVehicle:

    def __init__(
        self,
        x,
        y,
        speed,
        oncoming=True
    ):

        # Center coordinates.

        self.x = float(x)
        self.y = float(y)

        self.width = 44
        self.height = 74

        self.speed = float(
            speed
        )

        self.oncoming = bool(
            oncoming
        )

        self.collision_cooldown = 0

    # ========================================================
    # UPDATE
    # ========================================================

    def update(self):

        if self.collision_cooldown > 0:

            self.collision_cooldown -= 1
            return

        if self.oncoming:

            # Ego travels toward negative Y.
            #
            # Oncoming vehicle travels toward positive Y.
            self.y += self.speed

        else:

            self.y -= self.speed

    # ========================================================
    # COLLISION
    # ========================================================

    def register_collision(
        self,
        frames=30
    ):

        self.collision_cooldown = max(
            self.collision_cooldown,
            int(frames)
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

        surface = pygame.Surface(
            (
                self.width,
                self.height
            ),
            pygame.SRCALPHA
        )

        pygame.draw.rect(
            surface,
            (
                75,
                75,
                80
            ),
            (
                0,
                0,
                self.width,
                self.height
            ),
            border_radius=6
        )

        # ----------------------------------------------------
        # WINDSHIELD
        # ----------------------------------------------------

        if self.oncoming:

            windshield_y = (
                self.height - 25
            )

        else:

            windshield_y = 9

        pygame.draw.rect(
            surface,
            (
                130,
                205,
                225
            ),
            (
                7,
                windshield_y,
                self.width - 14,
                16
            ),
            border_radius=3
        )

        screen_x, screen_y = (
            camera.apply(
                (
                    self.x,
                    self.y
                )
            )
        )

        rect = surface.get_rect(
            center=(
                int(screen_x),
                int(screen_y)
            )
        )

        screen.blit(
            surface,
            rect
        )