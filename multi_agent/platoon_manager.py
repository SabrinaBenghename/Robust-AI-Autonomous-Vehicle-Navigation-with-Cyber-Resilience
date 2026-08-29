from typing import List, Dict, Any


class PlatoonVehicle:
    def __init__(self, vehicle_id: str, is_leader: bool = False):
        self.vehicle_id = vehicle_id
        self.is_leader = is_leader
        self.target_gap_m = 10.0


class PlatoonManager:
    """
    Cooperative Adaptive Cruise Control (CACC) manager for autonomous vehicle platooning.
    """

    def __init__(self, platoon_id: str):
        self.platoon_id = platoon_id
        self.members: List[PlatoonVehicle] = []

    def add_vehicle(self, vehicle_id: str, is_leader: bool = False):
        vehicle = PlatoonVehicle(vehicle_id, is_leader)
        if is_leader:
            self.members.insert(0, vehicle)
        else:
            self.members.append(vehicle)

    def calculate_platoon_gap(self, follower_idx: int, leader_speed: float) -> float:
        """
        Calculates dynamic target gap distance based on leader speed and time headway.
        """
        time_headway = 0.5  # seconds
        min_gap = 5.0  # meters
        return min_gap + leader_speed * time_headway

    def get_status(self) -> Dict[str, Any]:
        return {
            "platoon_id": self.platoon_id,
            "vehicle_count": len(self.members),
            "leader_id": self.members[0].vehicle_id if self.members else None
        }
