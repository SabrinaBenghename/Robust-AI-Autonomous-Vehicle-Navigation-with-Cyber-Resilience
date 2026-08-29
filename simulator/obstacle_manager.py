import random

from simulator.obstacle import Obstacle


class ObstacleManager:

    def __init__(self, road_width):

        self.road_width = road_width

        self.obstacles = []

        self.spawn_distance = 800
        self.remove_distance = 500

    def update(self, vehicle):

        # ------------------------------------------------
        # REMOVE OBSTACLES THAT ARE FAR BEHIND THE VEHICLE
        # ------------------------------------------------

        remaining_obstacles = []

        for obstacle in self.obstacles:

            if obstacle.y < vehicle.y + self.remove_distance:

                remaining_obstacles.append(obstacle)

        self.obstacles = remaining_obstacles

        # ------------------------------------------------
        # FIND THE FURTHEST OBSTACLE
        # ------------------------------------------------

        if self.obstacles:

            furthest_y = min(
                obstacle.y
                for obstacle in self.obstacles
            )

        else:

            furthest_y = vehicle.y

        # ------------------------------------------------
        # SPAWN NEW OBSTACLES
        # ------------------------------------------------

        while furthest_y > vehicle.y - self.spawn_distance:

            self.spawn_obstacle(
                furthest_y - random.randint(250, 450)
            )

            furthest_y = min(
                obstacle.y
                for obstacle in self.obstacles
            )

    def spawn_obstacle(self, y):

        # Four lanes
        lane_width = self.road_width / 4

        # Random lane
        lane = random.randint(0, 3)

        road_center = 640

        x = (
            road_center
            - self.road_width / 2
            + lane_width * lane
            + lane_width / 2
        )

        obstacle = Obstacle(
            x,
            y
        )

        self.obstacles.append(
            obstacle
        )

    def get_obstacles(self):

        return self.obstacles