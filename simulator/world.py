import pygame


class World:

    def __init__(self):

        # -------------------------
        # ROAD
        # -------------------------

        self.road_width = 600
        self.lane_count = 4
        self.lane_width = self.road_width / self.lane_count

        # -------------------------
        # VISUAL SETTINGS
        # -------------------------

        self.grass_color = (45, 120, 45)
        self.road_color = (55, 55, 55)
        self.shoulder_color = (120, 120, 120)
        self.lane_color = (235, 235, 235)
        self.center_line_color = (255, 210, 0)

        # -------------------------
        # LANE MARKINGS
        # -------------------------

        self.dash_length = 30
        self.dash_gap = 20
        self.line_width = 4

    def get_road_x(self, screen_width):

        return (
            screen_width - self.road_width
        ) / 2

    def get_lane_positions(self, screen_width):

        road_x = self.get_road_x(
            screen_width
        )

        center_x = screen_width / 2

        return [
            road_x,
            road_x + self.lane_width,
            center_x,
            road_x + self.lane_width * 3,
            road_x + self.road_width
        ]

    def get_lane_centers(self, screen_width):

        boundaries = self.get_lane_positions(
            screen_width
        )

        centers = []

        for i in range(len(boundaries) - 1):

            centers.append(
                (boundaries[i] + boundaries[i + 1]) / 2
            )

        return centers

    def get_lane_ground_truth(
        self,
        screen_width,
        screen_height
    ):

        lane_positions = self.get_lane_positions(
            screen_width
        )

        return {
            "road_left": lane_positions[0],
            "lane_line_1": lane_positions[1],
            "center_line": lane_positions[2],
            "lane_line_2": lane_positions[3],
            "road_right": lane_positions[4],
            "image_height": screen_height
        }

    def draw(self, screen, camera):

        width = screen.get_width()
        height = screen.get_height()

        # -------------------------
        # GRASS
        # -------------------------

        screen.fill(self.grass_color)

        # -------------------------
        # ROAD POSITION
        # -------------------------

        road_x = self.get_road_x(width)

        # -------------------------
        # ROAD SHOULDER
        # -------------------------

        pygame.draw.rect(
            screen,
            self.shoulder_color,
            (
                road_x - 8,
                0,
                self.road_width + 16,
                height
            )
        )

        # -------------------------
        # ROAD
        # -------------------------

        pygame.draw.rect(
            screen,
            self.road_color,
            (
                road_x,
                0,
                self.road_width,
                height
            )
        )

        # -------------------------
        # ROAD BOUNDARIES
        # -------------------------

        pygame.draw.line(
            screen,
            self.lane_color,
            (road_x, 0),
            (road_x, height),
            self.line_width
        )

        pygame.draw.line(
            screen,
            self.lane_color,
            (
                road_x + self.road_width,
                0
            ),
            (
                road_x + self.road_width,
                height
            ),
            self.line_width
        )

        # -------------------------
        # CENTER LINE
        # -------------------------

        center_x = width / 2

        pygame.draw.line(
            screen,
            self.center_line_color,
            (center_x, 0),
            (center_x, height),
            self.line_width
        )

        # -------------------------
        # DASHED LANE LINES
        # -------------------------

        lane_positions = [
            road_x + self.lane_width,
            road_x + self.lane_width * 3
        ]

        pattern_length = (
            self.dash_length
            + self.dash_gap
        )

        start_y = (
            -(camera.y % pattern_length)
        )

        for lane_x in lane_positions:

            y = start_y

            while y < height:

                pygame.draw.line(
                    screen,
                    self.lane_color,
                    (lane_x, y),
                    (
                        lane_x,
                        y + self.dash_length
                    ),
                    self.line_width
                )

                y += pattern_length