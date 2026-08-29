from enum import Enum
from typing import List

class BehaviorState(Enum):
    CRUISE = "CRUISE"
    FOLLOW_VEHICLE = "FOLLOW_VEHICLE"
    LANE_CHANGE_LEFT = "LANE_CHANGE_LEFT"
    LANE_CHANGE_RIGHT = "LANE_CHANGE_RIGHT"
    EMERGENCY_STOP = "EMERGENCY_STOP"

class BehaviorPlanner:
    """
    Finite State Machine (FSM) for high-level tactical behavioral decision making.
    """
    def __init__(self, target_speed: float = 20.0, safe_distance: float = 15.0):
        self.target_speed = target_speed
        self.safe_distance = safe_distance
        self.state = BehaviorState.CRUISE

    def update(self, ego_speed: float, obstacles: List) -> BehaviorState:
        """
        Evaluate surroundings and determine tactical maneuver state.
        """
        closest_dist = float('inf')
        for obs in obstacles:
            if hasattr(obs, 'x') and obs.x > 0:
                dist = obs.x
                if dist < closest_dist:
                    closest_dist = dist

        if closest_dist < self.safe_distance * 0.4:
            self.state = BehaviorState.EMERGENCY_STOP
        elif closest_dist < self.safe_distance:
            self.state = BehaviorState.FOLLOW_VEHICLE
        else:
            self.state = BehaviorState.CRUISE

        return self.state
