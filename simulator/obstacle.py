import pygame


class Obstacle:

    def __init__(
        self,
        x,
        y,
        width=40,
        height=40
    ):

        self.x = x
        self.y = y

        self.width = width
        self.height = height

    def draw(self, screen, camera):

        screen_x, screen_y = camera.apply(
            (self.x, self.y)
        )

        pygame.draw.rect(
            screen,
            (30, 30, 30),
            (
                screen_x - self.width / 2,
                screen_y - self.height / 2,
                self.width,
                self.height
            ),
            border_radius=5
        )