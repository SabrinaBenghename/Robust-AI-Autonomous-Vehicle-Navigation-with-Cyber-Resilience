import unittest
from simulator.vehicle import Vehicle
from simulator.camera import Camera
from simulator.physics import PhysicsEngine
from simulator.sensors import SensorSuite


class TestSimulator(unittest.TestCase):

    def test_vehicle_initialization(self):
        v = Vehicle()
        self.assertEqual(v.x, 640)
        self.assertEqual(v.y, 550)
        self.assertEqual(v.speed, 0)

    def test_camera_tracking(self):
        cam = Camera(1280, 720)
        cam.follow(640, 550)
        sx, sy = cam.apply(640, 550)
        self.assertGreaterEqual(round(sx), 0)

    def test_physics_engine(self):
        phys = PhysicsEngine()
        force = phys.calculate_forces(speed=10.0, throttle=0.5, brake=0.0)
        self.assertGreater(force, 0)

    def test_sensors(self):
        suite = SensorSuite()
        lidar = suite.get_lidar_scan(0, 0, 0, [(10, 10)])
        self.assertEqual(len(lidar), 36)


if __name__ == "__main__":
    unittest.main()
