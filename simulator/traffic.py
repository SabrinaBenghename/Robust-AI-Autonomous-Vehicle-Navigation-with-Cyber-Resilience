import pygame
import random
from typing import List


class NPCDriver:
    """
    Non-Player Character vehicle representing ambient road traffic.
    """

    def __init__(self, vehicle_id: int, x: float, y: float, speed: float = 3.0, color=(50, 100, 220)):
        self.id = vehicle_id
        self.x = x
        self.y = y
        self.width = 38
        self.height = 65
        self.speed = speed
        self.color = color

    def update(self):
        """
        Move vehicle forward smoothly down the road.
        """
        self.y -= self.speed
        if self.y < -100:
            self.y = 800  # Respawn at bottom of screen loop

    def draw(self, screen: pygame.Surface):
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(surface, self.color, (0, 0, self.width, self.height), border_radius=6)
        screen.blit(surface, (self.x - self.width / 2, self.y - self.height / 2))


class TrafficManager:
    """
    Manages spawning, updating, and collision detection for background traffic vehicles.
    """

    def __init__(self, num_vehicles: int = 5):
        self.vehicles: List[NPCDriver] = []
        self.spawn_traffic(num_vehicles)

    def spawn_traffic(self, count: int):
        lanes_x = [450, 570, 710, 830]
        for i in range(count):
            x = random.choice(lanes_x)
            y = random.randint(-400, 600)
            speed = random.uniform(2.5, 4.5)
            color = random.choice([(40, 120, 200), (200, 100, 40), (100, 200, 80), (180, 50, 180)])
            self.vehicles.append(NPCDriver(i, x, y, speed, color))

    def update(self):
        for vehicle in self.vehicles:
            vehicle.update()

    def draw(self, screen: pygame.Surface):
        for vehicle in self.vehicles:
            vehicle.draw(screen)
