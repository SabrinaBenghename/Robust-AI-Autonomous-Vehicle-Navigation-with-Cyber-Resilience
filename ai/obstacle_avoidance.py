import math


class ObstacleAvoidance:

    def __init__(self):

        # ====================================================
        # HIGH-SPEED PLANNING
        # ====================================================

        self.trigger_distance = 1150.0

        self.trigger_ttc = 95.0

        # ====================================================
        # NO SAFE ESCAPE
        # ====================================================

        self.slow_ttc = 48.0

        self.emergency_ttc = 22.0

        # ====================================================
        # TARGET LANE SAFETY
        # ====================================================

        self.minimum_ahead_gap = 420.0

        self.minimum_target_ttc = 75.0

        self.abort_target_ttc = 28.0

        # ====================================================
        # RETURN
        # ====================================================

        self.return_minimum_gap = 650.0

        self.return_minimum_ttc = 90.0

        self.return_clear_frames = 24

        self.return_clear_counter = 0

        # ====================================================
        # DEBUG
        # ====================================================

        self.last_reason = "START"

        self.last_lane_gaps = {}

        self.last_lane_ttc = {}

    # ========================================================
    # CLOSEST LANE
    # ========================================================

    @staticmethod
    def closest_lane(
        x,
        lane_centers
    ):

        return min(
            range(
                len(lane_centers)
            ),
            key=lambda lane_index: abs(
                float(x)
                - float(
                    lane_centers[
                        lane_index
                    ]
                )
            )
        )

    # ========================================================
    # DIRECTION
    # ========================================================

    @staticmethod
    def direction_to_lane(
        vehicle,
        lane_index,
        lane_centers
    ):

        target_x = float(
            lane_centers[
                lane_index
            ]
        )

        error = (
            target_x
            - float(vehicle.x)
        )

        if error > 6.0:

            return "RIGHT"

        if error < -6.0:

            return "LEFT"

        return "FORWARD"

    # ========================================================
    # CLOSING SPEED
    # ========================================================

    @staticmethod
    def closing_speed(
        vehicle,
        obstacle
    ):

        ego_speed = abs(
            float(vehicle.speed)
        )

        obstacle_speed = abs(
            float(
                getattr(
                    obstacle,
                    "speed",
                    0.0
                )
            )
        )

        oncoming = bool(
            getattr(
                obstacle,
                "oncoming",
                True
            )
        )

        if oncoming:

            return max(
                0.01,
                ego_speed
                + obstacle_speed
            )

        return max(
            0.01,
            ego_speed
            - obstacle_speed
        )

    # ========================================================
    # LANE STATE
    # ========================================================

    def lane_state(
        self,
        lane_index,
        vehicle,
        traffic,
        lane_centers,
        lane_width
    ):

        lane_center = float(
            lane_centers[
                lane_index
            ]
        )

        nearest_ahead_gap = float(
            "inf"
        )

        nearest_ahead_ttc = float(
            "inf"
        )

        nearest_behind_gap = float(
            "inf"
        )

        ego_half_length = (
            float(vehicle.height)
            / 2.0
        )

        for obstacle in traffic:

            lateral_distance = abs(
                float(obstacle.x)
                - lane_center
            )

            if (
                lateral_distance
                > lane_width * 0.44
            ):

                continue

            obstacle_half_length = (
                float(
                    getattr(
                        obstacle,
                        "height",
                        70.0
                    )
                )
                / 2.0
            )

            center_difference = (
                float(vehicle.y)
                - float(obstacle.y)
            )

            body_length = (
                ego_half_length
                + obstacle_half_length
            )

            # =================================================
            # AHEAD
            # =================================================

            if center_difference >= 0.0:

                gap = max(
                    0.0,
                    center_difference
                    - body_length
                )

                closing = (
                    self.closing_speed(
                        vehicle,
                        obstacle
                    )
                )

                ttc = (
                    gap
                    / closing
                )

                nearest_ahead_gap = min(
                    nearest_ahead_gap,
                    gap
                )

                nearest_ahead_ttc = min(
                    nearest_ahead_ttc,
                    ttc
                )

            # =================================================
            # BEHIND
            # =================================================

            else:

                gap = max(
                    0.0,
                    -center_difference
                    - body_length
                )

                nearest_behind_gap = min(
                    nearest_behind_gap,
                    gap
                )

        return {

            "ahead_gap": (
                nearest_ahead_gap
            ),

            "ahead_ttc": (
                nearest_ahead_ttc
            ),

            "behind_gap": (
                nearest_behind_gap
            )
        }

    # ========================================================
    # TARGET LANE SAFE
    # ========================================================

    def target_lane_safe(
        self,
        lane_index,
        vehicle,
        traffic,
        lane_centers,
        lane_width,
        required_gap=None,
        required_ttc=None
    ):

        if required_gap is None:

            required_gap = (
                self.minimum_ahead_gap
            )

        if required_ttc is None:

            required_ttc = (
                self.minimum_target_ttc
            )

        state = self.lane_state(
            lane_index,
            vehicle,
            traffic,
            lane_centers,
            lane_width
        )

        safe = (
            state["ahead_gap"]
            >= required_gap
            and state["ahead_ttc"]
            >= required_ttc
        )

        return (
            safe,
            state
        )

    # ========================================================
    # CHOOSE BEST ADJACENT LANE
    # ========================================================

    def choose_escape_lane(
        self,
        current_lane,
        cruise_lane,
        vehicle,
        traffic,
        lane_centers,
        lane_width
    ):

        candidates = []

        if current_lane - 1 >= 0:

            candidates.append(
                current_lane - 1
            )

        if (
            current_lane + 1
            < len(lane_centers)
        ):

            candidates.append(
                current_lane + 1
            )

        best_lane = None

        best_score = float(
            "-inf"
        )

        for lane_index in candidates:

            safe, state = (
                self.target_lane_safe(
                    lane_index,
                    vehicle,
                    traffic,
                    lane_centers,
                    lane_width
                )
            )

            if not safe:

                continue

            ahead_gap = (
                state[
                    "ahead_gap"
                ]
            )

            ahead_ttc = (
                state[
                    "ahead_ttc"
                ]
            )

            # =================================================
            # SCORE
            # =================================================

            if math.isinf(
                ahead_gap
            ):

                gap_score = 1800.0

            else:

                gap_score = min(
                    ahead_gap,
                    1800.0
                )

            if math.isinf(
                ahead_ttc
            ):

                ttc_score = 150.0

            else:

                ttc_score = min(
                    ahead_ttc,
                    150.0
                )

            score = (
                gap_score
                + ttc_score * 6.0
            )

            # Prefer staying on right side of road.

            if (
                cruise_lane >= 2
                and lane_index >= 2
            ):

                score += 180.0

            score -= (
                abs(
                    lane_index
                    - cruise_lane
                )
                * 25.0
            )

            if score > best_score:

                best_score = score

                best_lane = lane_index

        return best_lane

    # ========================================================
    # MAIN PLANNER
    # ========================================================

    def plan(
        self,
        sensor_data,
        vehicle,
        traffic,
        lane_centers,
        target_lane,
        cruise_lane,
        lane_width
    ):

        # ====================================================
        # PHYSICAL LANE
        # ====================================================

        current_lane = (
            self.closest_lane(
                vehicle.x,
                lane_centers
            )
        )

        # ====================================================
        # ALL LANE STATES
        # ====================================================

        lane_states = {}

        for lane_index in range(
            len(lane_centers)
        ):

            state = self.lane_state(
                lane_index,
                vehicle,
                traffic,
                lane_centers,
                lane_width
            )

            lane_states[
                lane_index
            ] = state

        self.last_lane_gaps = {

            lane_index: {

                "ahead": (
                    state[
                        "ahead_gap"
                    ]
                ),

                "behind": (
                    state[
                        "behind_gap"
                    ]
                )
            }

            for lane_index, state
            in lane_states.items()
        }

        self.last_lane_ttc = {

            lane_index: (
                state[
                    "ahead_ttc"
                ]
            )

            for lane_index, state
            in lane_states.items()
        }

        # ====================================================
        # CURRENT THREAT
        # ====================================================

        current_state = (
            lane_states[
                current_lane
            ]
        )

        current_gap = (
            current_state[
                "ahead_gap"
            ]
        )

        current_ttc = (
            current_state[
                "ahead_ttc"
            ]
        )

        sensor_front = float(
            sensor_data.get(
                "front",
                1500.0
            )
        )

        front_distance = min(
            sensor_front,
            current_gap
        )

        # ====================================================
        # CURRENTLY MOVING TOWARD TARGET?
        # ====================================================

        target_center = float(
            lane_centers[
                target_lane
            ]
        )

        target_error = abs(
            float(vehicle.x)
            - target_center
        )

        changing_lane = (
            target_error > 22.0
        )

        # ====================================================
        # COMPLETE CURRENT CHANGE
        # ====================================================

        if changing_lane:

            target_state = (
                lane_states[
                    target_lane
                ]
            )

            target_ttc = (
                target_state[
                    "ahead_ttc"
                ]
            )

            # ------------------------------------------------
            # TARGET BECAME DANGEROUS
            # ------------------------------------------------

            if (
                target_ttc
                < self.abort_target_ttc
            ):

                current_safe = (
                    current_state[
                        "ahead_ttc"
                    ]
                    > self.abort_target_ttc
                    + 15.0
                )

                if current_safe:

                    self.last_reason = (
                        "ABORT UNSAFE TARGET"
                    )

                    return {

                        "decision": (
                            self.direction_to_lane(
                                vehicle,
                                current_lane,
                                lane_centers
                            )
                        ),

                        "target_lane": (
                            current_lane
                        ),

                        "front_distance": (
                            front_distance
                        ),

                        "reason": (
                            self.last_reason
                        )
                    }

                self.last_reason = (
                    "EMERGENCY DURING LANE CHANGE"
                )

                return {

                    "decision": "STOP",

                    "target_lane": (
                        target_lane
                    ),

                    "front_distance": (
                        front_distance
                    ),

                    "reason": (
                        self.last_reason
                    )
                }

            # ------------------------------------------------
            # CONTINUE
            # ------------------------------------------------

            self.last_reason = (
                "COMPLETING LANE CHANGE"
            )

            return {

                "decision": (
                    self.direction_to_lane(
                        vehicle,
                        target_lane,
                        lane_centers
                    )
                ),

                "target_lane": (
                    target_lane
                ),

                "front_distance": (
                    front_distance
                ),

                "reason": (
                    self.last_reason
                )
            }

        # ====================================================
        # THREAT?
        # ====================================================

        threat = (
            current_gap
            <= self.trigger_distance
            or current_ttc
            <= self.trigger_ttc
        )

        if threat:

            self.return_clear_counter = 0

            # =================================================
            # CHOOSE SAFE LANE
            # =================================================

            best_lane = (
                self.choose_escape_lane(
                    current_lane,
                    cruise_lane,
                    vehicle,
                    traffic,
                    lane_centers,
                    lane_width
                )
            )

            if best_lane is not None:

                self.last_reason = (
                    "PREDICTIVE AVOIDANCE -> LANE "
                    + str(best_lane)
                )

                return {

                    "decision": (
                        self.direction_to_lane(
                            vehicle,
                            best_lane,
                            lane_centers
                        )
                    ),

                    "target_lane": (
                        best_lane
                    ),

                    "front_distance": (
                        front_distance
                    ),

                    "reason": (
                        self.last_reason
                    )
                }

            # =================================================
            # NO SAFE LANE
            # =================================================

            if (
                current_ttc
                <= self.emergency_ttc
            ):

                self.last_reason = (
                    "EMERGENCY BRAKE"
                )

                return {

                    "decision": "STOP",

                    "target_lane": (
                        current_lane
                    ),

                    "front_distance": (
                        front_distance
                    ),

                    "reason": (
                        self.last_reason
                    )
                }

            if (
                current_ttc
                <= self.slow_ttc
            ):

                self.last_reason = (
                    "SLOW: NO SAFE LANE"
                )

                return {

                    "decision": "SLOW",

                    "target_lane": (
                        current_lane
                    ),

                    "front_distance": (
                        front_distance
                    ),

                    "reason": (
                        self.last_reason
                    )
                }

            self.last_reason = (
                "MONITORING BLOCKED LANES"
            )

            return {

                "decision": "FORWARD",

                "target_lane": (
                    current_lane
                ),

                "front_distance": (
                    front_distance
                ),

                "reason": (
                    self.last_reason
                )
            }

        # ====================================================
        # RETURN TO CRUISE
        # ====================================================

        if current_lane != cruise_lane:

            cruise_state = (
                lane_states[
                    cruise_lane
                ]
            )

            cruise_safe = (
                cruise_state[
                    "ahead_gap"
                ]
                >= self.return_minimum_gap

                and

                cruise_state[
                    "ahead_ttc"
                ]
                >= self.return_minimum_ttc
            )

            current_lane_clear = (
                current_state[
                    "ahead_ttc"
                ]
                > self.return_minimum_ttc
            )

            if (
                cruise_safe
                and current_lane_clear
            ):

                self.return_clear_counter += 1

            else:

                self.return_clear_counter = 0

            if (
                self.return_clear_counter
                >= self.return_clear_frames
            ):

                self.return_clear_counter = 0

                self.last_reason = (
                    "RETURNING TO CRUISE LANE"
                )

                return {

                    "decision": (
                        self.direction_to_lane(
                            vehicle,
                            cruise_lane,
                            lane_centers
                        )
                    ),

                    "target_lane": (
                        cruise_lane
                    ),

                    "front_distance": (
                        front_distance
                    ),

                    "reason": (
                        self.last_reason
                    )
                }

            self.last_reason = (
                "HOLDING SAFE LANE"
            )

            return {

                "decision": "FORWARD",

                "target_lane": (
                    current_lane
                ),

                "front_distance": (
                    front_distance
                ),

                "reason": (
                    self.last_reason
                )
            }

        # ====================================================
        # CRUISE
        # ====================================================

        self.return_clear_counter = 0

        self.last_reason = (
            "ROAD CLEAR"
        )

        return {

            "decision": "FORWARD",

            "target_lane": (
                cruise_lane
            ),

            "front_distance": (
                front_distance
            ),

            "reason": (
                self.last_reason
            )
        }

    # ========================================================
    # LEGACY API
    # ========================================================

    def decide(
        self,
        sensor_data
    ):

        front = float(
            sensor_data.get(
                "front",
                1500.0
            )
        )

        left = float(
            sensor_data.get(
                "left",
                1500.0
            )
        )

        right = float(
            sensor_data.get(
                "right",
                1500.0
            )
        )

        if (
            front
            >= self.trigger_distance
        ):

            return "FORWARD"

        if right >= left:

            return "RIGHT"

        return "LEFT"