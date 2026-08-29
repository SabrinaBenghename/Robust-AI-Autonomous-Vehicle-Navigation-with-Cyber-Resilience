import pygame
import math
import random
from typing import List


class Pedestrian:
    """
    Simulates a pedestrian agent with crosswalk navigation and collision bounding circle.
    """

    def __init__(self, agent_id: int, x: float, y: float, speed: float = 1.0):
        self.id = agent_id
        self.x = x
        self.y = y
        self.radius = 8
        self.speed = speed
        self.direction = random.choice([0, math.pi / 2, math.pi, 3 * math.pi / 2])
        self.color = (240, 200, 80)

    def update(self):
        """
        Update pedestrian walking movement.
        """
        self.x += math.cos(self.direction) * self.speed
        self.y += math.sin(self.direction) * self.speed

        # Random direction adjustment
        if random.random() < 0.02:
            self.direction += random.uniform(-0.5, 0.5)

    def draw(self, screen: pygame.Surface):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        # Head highlight
        pygame.draw.circle(screen, (255, 255, 255), (int(self.x), int(self.y)), self.radius - 3)


class PedestrianManager:
    """
    Spawns and manages pedestrian agents near crosswalks and sidewalks.
    """

    def __init__(self, count: int = 4):
        self.pedestrians: List[Pedestrian] = []
        for i in range(count):
            x = random.randint(200, 1000)
            y = random.randint(100, 600)
            self.pedestrians.append(Pedestrian(i, x, y))

    def update(self):
        for p in self.pedestrians:
            p.update()

    def draw(self, screen: pygame.Surface):
        for p in self.pedestrians:
            p.draw(screen)
