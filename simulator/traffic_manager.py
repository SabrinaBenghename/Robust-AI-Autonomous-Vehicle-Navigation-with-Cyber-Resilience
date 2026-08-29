import random

from simulator.settings import WINDOW_WIDTH
from simulator.traffic_vehicle import TrafficVehicle


class TrafficManager:

    def __init__(
        self,
        road_width
    ):

        # ====================================================
        # ROAD GEOMETRY
        # ====================================================

        self.road_width = float(
            road_width
        )

        self.road_left = (
            WINDOW_WIDTH
            - self.road_width
        ) / 2.0

        self.road_right = (
            self.road_left
            + self.road_width
        )

        self.lane_count = 4

        self.lane_width = (
            self.road_width
            / self.lane_count
        )

        # ====================================================
        # LANE CENTERS
        # ====================================================

        self.lane_centers = []

        for lane_index in range(
            self.lane_count
        ):

            center = (
                self.road_left
                + lane_index * self.lane_width
                + self.lane_width / 2.0
            )

            self.lane_centers.append(
                center
            )

        # ====================================================
        # TRAFFIC DENSITY
        # ====================================================

        self.vehicles = []

        # Keep the amount of traffic you liked.
        self.max_vehicles = 7

        self.spawn_interval = 45

        self.spawn_timer = 0

        # ====================================================
        # IMPORTANT:
        # SPAWN MUCH FARTHER AHEAD
        # ====================================================
        #
        # Before:
        #
        # 350px could appear suddenly ahead.
        #
        # At ego speed ~9 plus oncoming traffic speed,
        # that gave almost no time to change lane.
        #
        # Now obstacles appear early enough for planning.
        # ====================================================

        self.spawn_min_distance = 850.0

        self.spawn_max_distance = 1450.0

        # ====================================================
        # ONCOMING SPEED
        # ====================================================

        self.min_speed = 2.0

        self.max_speed = 3.8

        # ====================================================
        # VEHICLE SPACING
        # ====================================================

        self.min_same_lane_gap = 260.0

        self.min_general_gap = 150.0

        # ====================================================
        # COLLISIONS
        # ====================================================

        self.collision_count = 0

        # Prevent one physical collision from becoming
        # 5, 10, 20 collision events.
        self.recorded_collision_ids = set()

    # ========================================================
    # GET LANE CENTER
    # ========================================================

    def get_lane_center(
        self,
        lane_index
    ):

        lane_index = max(
            0,
            min(
                int(lane_index),
                self.lane_count - 1
            )
        )

        return self.lane_centers[
            lane_index
        ]

    # ========================================================
    # CHECK SPAWN POSITION
    # ========================================================

    def can_spawn_at(
        self,
        x,
        y
    ):

        for vehicle in self.vehicles:

            dx = abs(
                float(vehicle.x)
                - float(x)
            )

            dy = abs(
                float(vehicle.y)
                - float(y)
            )

            # ------------------------------------------------
            # SAME LANE
            # ------------------------------------------------

            if (
                dx
                < self.lane_width * 0.42
            ):

                if (
                    dy
                    < self.min_same_lane_gap
                ):

                    return False

            # ------------------------------------------------
            # GENERAL ANTI-OVERLAP
            # ------------------------------------------------

            if (
                dx < 65.0
                and dy < self.min_general_gap
            ):

                return False

        return True

    # ========================================================
    # SPAWN RANDOM VEHICLE
    # ========================================================

    def spawn_vehicle(
        self,
        ego_vehicle
    ):

        for _ in range(30):

            # =================================================
            # RANDOM LANE
            # =================================================

            lane_index = random.randint(
                0,
                self.lane_count - 1
            )

            lane_center = (
                self.get_lane_center(
                    lane_index
                )
            )

            # =================================================
            # SMALL RANDOM OFFSET
            # =================================================

            x = (
                lane_center
                + random.uniform(
                    -10.0,
                    10.0
                )
            )

            # =================================================
            # FORCE WHOLE CAR ON GREY ROAD
            # =================================================

            half_width = 24.0

            minimum_x = (
                self.road_left
                + half_width
                + 6.0
            )

            maximum_x = (
                self.road_right
                - half_width
                - 6.0
            )

            x = max(
                minimum_x,
                min(
                    x,
                    maximum_x
                )
            )

            # =================================================
            # RANDOM POSITION AHEAD
            # =================================================

            distance = random.uniform(
                self.spawn_min_distance,
                self.spawn_max_distance
            )

            y = (
                float(ego_vehicle.y)
                - distance
            )

            # =================================================
            # CHECK SPACING
            # =================================================

            if not self.can_spawn_at(
                x,
                y
            ):

                continue

            # =================================================
            # RANDOM ONCOMING SPEED
            # =================================================

            speed = random.uniform(
                self.min_speed,
                self.max_speed
            )

            # =================================================
            # CREATE
            # =================================================

            vehicle = TrafficVehicle(
                x=x,
                y=y,
                speed=speed,
                oncoming=True
            )

            self.vehicles.append(
                vehicle
            )

            return True

        return False

    # ========================================================
    # COLLISION CHECK
    # ========================================================

    def check_collisions(
        self,
        ego_vehicle
    ):

        ego_rect = (
            ego_vehicle.get_rect()
        )

        collided_vehicles = []

        for traffic in self.vehicles:

            traffic_rect = (
                traffic.get_rect()
            )

            if not ego_rect.colliderect(
                traffic_rect
            ):

                continue

            collision_id = id(
                traffic
            )

            # =================================================
            # ALREADY RECORDED
            # =================================================

            if (
                collision_id
                in self.recorded_collision_ids
            ):

                continue

            # =================================================
            # RECORD EXACTLY ONCE
            # =================================================

            self.recorded_collision_ids.add(
                collision_id
            )

            self.collision_count += 1

            print()
            print("=" * 60)
            print("COLLISION DETECTED")
            print(
                "TOTAL COLLISIONS:",
                self.collision_count
            )
            print("=" * 60)
            print()

            # =================================================
            # SHORT EGO STOP
            # =================================================

            if hasattr(
                ego_vehicle,
                "register_collision"
            ):

                ego_vehicle.register_collision(
                    12
                )

            else:

                ego_vehicle.speed = 0.0

            # =================================================
            # IMPORTANT:
            # DO NOT PUSH THE CAR SIDEWAYS
            # ====================================================
            #
            # The old collision correction could move ego
            # 40+ pixels sideways after every repeated hit,
            # destroying lane error.
            #
            # We simply record the failed avoidance and remove
            # that obstacle from the experiment.
            # =================================================

            collided_vehicles.append(
                traffic
            )

        # ====================================================
        # REMOVE COLLIDED VEHICLES
        # ====================================================

        if collided_vehicles:

            self.vehicles = [

                vehicle

                for vehicle
                in self.vehicles

                if vehicle
                not in collided_vehicles
            ]

    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        ego_vehicle
    ):

        # ====================================================
        # MOVE TRAFFIC
        # ====================================================

        for vehicle in self.vehicles:

            vehicle.update()

        # ====================================================
        # REMOVE FAR VEHICLES
        # ====================================================

        remaining = []

        for vehicle in self.vehicles:

            if (
                ego_vehicle.y - 1750.0
                < vehicle.y
                < ego_vehicle.y + 900.0
            ):

                remaining.append(
                    vehicle
                )

        self.vehicles = remaining

        # ====================================================
        # SPAWN TIMER
        # ====================================================

        self.spawn_timer += 1

        if (
            self.spawn_timer
            >= self.spawn_interval
        ):

            self.spawn_timer = 0

            # ------------------------------------------------
            # ONLY ONE NEW VEHICLE AT A TIME
            # ------------------------------------------------

            if (
                len(self.vehicles)
                < self.max_vehicles
            ):

                self.spawn_vehicle(
                    ego_vehicle
                )

    # ========================================================
    # GET VEHICLES
    # ========================================================

    def get_vehicles(self):

        return self.vehicles