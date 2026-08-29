import time
from typing import List, Dict, Any


class V2XNetworkMesh:
    """
    Simulates V2V (Vehicle-to-Vehicle) and V2I (Vehicle-to-Infrastructure) radio message broadcasting.
    """

    def __init__(self, broadcast_range_m: float = 300.0):
        self.broadcast_range_m = broadcast_range_m
        self.connected_nodes: List[Dict[str, Any]] = []

    def register_node(self, node_id: str, node_type: str = "VEHICLE"):
        self.connected_nodes.append({"id": node_id, "type": node_type, "last_seen": time.time()})

    def broadcast_bsm(self, sender_id: str, position: tuple, speed: float, heading: float) -> Dict[str, Any]:
        """
        Broadcast Basic Safety Message (BSM) across the mesh network.
        """
        message = {
            "msg_type": "BSM",
            "sender_id": sender_id,
            "timestamp": time.time(),
            "position": position,
            "speed_m_s": speed,
            "heading_deg": heading
        }
        return message
