import unittest
from digital_twin.telemetry_sync import TelemetrySynchronizer
from digital_twin.predictive_maintenance import PredictiveMaintenanceEngine


class TestDigitalTwin(unittest.TestCase):

    def test_telemetry_sync(self):
        sync = TelemetrySynchronizer()
        packet = sync.capture_telemetry(speed_mps=25.0, x=10.0, y=20.0)
        self.assertEqual(packet.speed_kmh, 90.0)

    def test_predictive_maintenance(self):
        engine = PredictiveMaintenanceEngine()
        health = engine.evaluate_health(speed_mps=15.0, brake_active=True)
        self.assertIsNotNone(health)


if __name__ == "__main__":
    unittest.main()
