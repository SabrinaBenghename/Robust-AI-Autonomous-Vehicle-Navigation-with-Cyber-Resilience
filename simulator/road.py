import pygame
from typing import List, Tuple


class Waypoint:
    def __init__(self, x: float, y: float, speed_limit: float = 50.0, lane_id: int = 0):
        self.x = x
        self.y = y
        self.speed_limit = speed_limit
        self.lane_id = lane_id


class RoadSegment:
    """
    Represents a road section with lanes, boundaries, and navigation waypoints.
    """

    def __init__(
        self,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        lanes: int = 2,
        lane_width: float = 60.0
    ):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.lanes = lanes
        self.lane_width = lane_width
        self.waypoints: List[Waypoint] = []
        self._generate_waypoints()

    def _generate_waypoints(self):
        """
        Generate discrete navigation waypoints along center lines of lanes.
        """
        num_points = 20
        dx = (self.end_pos[0] - self.start_pos[0]) / num_points
        dy = (self.end_pos[1] - self.start_pos[1]) / num_points

        for i in range(num_points + 1):
            for lane in range(self.lanes):
                lane_offset = (lane - (self.lanes - 1) / 2.0) * self.lane_width
                wp_x = self.start_pos[0] + i * dx + lane_offset
                wp_y = self.start_pos[1] + i * dy
                self.waypoints.append(Waypoint(wp_x, wp_y, lane_id=lane))

    def draw(self, screen: pygame.Surface):
        """
        Render road network surface, lane dividers, and markings.
        """
        total_width = self.lanes * self.lane_width
        road_rect = pygame.Rect(
            min(self.start_pos[0], self.end_pos[0]) - total_width / 2,
            min(self.start_pos[1], self.end_pos[1]),
            total_width,
            abs(self.end_pos[1] - self.start_pos[1]) or 800
        )
        # Asphalt
        pygame.draw.rect(screen, (50, 50, 55), road_rect)

        # Draw lane separators
        for i in range(1, self.lanes):
            offset = -total_width / 2 + i * self.lane_width
            start_x = self.start_pos[0] + offset
            end_x = self.end_pos[0] + offset
            pygame.draw.line(screen, (240, 240, 240), (start_x, self.start_pos[1]), (end_x, self.end_pos[1]), 2)
