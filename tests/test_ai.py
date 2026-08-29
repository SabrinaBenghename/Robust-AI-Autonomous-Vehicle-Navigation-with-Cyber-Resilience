import unittest
from ai.perception import AIPerceptionPipeline
from ai.planning import AIPlannerEngine
from ai.controller import AIControllerSuite
from ai.navigation import AINavigationSystem
from control.vehicle_dynamics import VehicleState
from planning.trajectory_generator import TrajectoryPoint


class TestAI(unittest.TestCase):

    def test_ai_perception(self):
        pipeline = AIPerceptionPipeline()
        res = pipeline.process_frame(None, [10.0] * 36, [])
        self.assertEqual(res["status"], "TRACKING_ACTIVE")

    def test_ai_planner(self):
        planner = AIPlannerEngine()
        plan = planner.plan_step((0, 0, 0), (100, 100), [], 10.0)
        self.assertIn("behavior_state", plan)

    def test_ai_controller(self):
        controller = AIControllerSuite()
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=10.0)
        waypoints = [TrajectoryPoint(x=10.0, y=0.0, yaw=0.0, target_velocity=15.0)]
        cmds = controller.compute_commands(state, target_speed=15.0, waypoints=waypoints)
        self.assertIn("throttle", cmds)
        self.assertIn("steering_angle_deg", cmds)

    def test_ai_navigation(self):
        nav = AINavigationSystem()
        nav.set_destination((0, 0), (500, 500))
        guidance = nav.get_guidance(100, 100)
        self.assertEqual(guidance["status"], "NAVIGATING")


if __name__ == "__main__":
    unittest.main()
