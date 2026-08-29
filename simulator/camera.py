class Camera:

    def __init__(
        self,
        width,
        height
    ):

        self.width = width
        self.height = height

        # ====================================================
        # IMPORTANT COORDINATE RULE
        # ====================================================
        #
        # Horizontal camera movement is DISABLED.
        #
        # This means:
        #
        # screen_x == world_x
        #
        # Therefore:
        #
        # V3 lane x
        # vehicle.x
        # traffic.x
        # road x
        #
        # all share the same horizontal coordinate system.
        # ====================================================

        self.x = 0.0
        self.y = 0.0

    # ========================================================
    # FOLLOW
    # ========================================================

    def follow(
        self,
        target
    ):

        # Never move horizontally.
        self.x = 0.0

        # Keep ego vehicle around 75% down the screen.
        # This gives more visibility ahead.
        self.y = (
            target.y
            - self.height * 0.75
        )

    # ========================================================
    # WORLD -> SCREEN
    # ========================================================

    def apply(
        self,
        position
    ):

        world_x, world_y = position

        screen_x = (
            world_x
            - self.x
        )

        screen_y = (
            world_y
            - self.y
        )

        return (
            screen_x,
            screen_y
        )

    # ========================================================
    # SCREEN -> WORLD
    # ========================================================

    def screen_to_world(
        self,
        position
    ):

        screen_x, screen_y = position

        return (
            screen_x + self.x,
            screen_y + self.y
        )