import unittest
from multi_agent.v2x_network import V2XNetworkMesh
from multi_agent.platoon_manager import PlatoonManager


class TestMultiAgent(unittest.TestCase):

    def test_v2x_mesh(self):
        mesh = V2XNetworkMesh()
        mesh.register_node("VEH_01")
        msg = mesh.broadcast_bsm("VEH_01", (0, 0), 20.0, 90.0)
        self.assertEqual(msg["msg_type"], "BSM")

    def test_platoon_manager(self):
        platoon = PlatoonManager("PLATOON_1")
        platoon.add_vehicle("LEADER_01", is_leader=True)
        platoon.add_vehicle("FOLLOWER_01", is_leader=False)
        status = platoon.get_status()
        self.assertEqual(status["vehicle_count"], 2)
        self.assertEqual(status["leader_id"], "LEADER_01")


if __name__ == "__main__":
    unittest.main()
