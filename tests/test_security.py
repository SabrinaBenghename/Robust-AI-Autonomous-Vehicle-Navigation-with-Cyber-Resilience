import unittest
from security import (
    IntegrityChecker,
    IntegrityReport,
    SecurityLogger,
    SecurityEvent,
    SecuritySeverity,
    SecurityMonitor,
    ThreatLevel,
)


class TestSecurityModule(unittest.TestCase):

    def setUp(self):
        self.checker = IntegrityChecker()
        self.logger = SecurityLogger()
        self.monitor = SecurityMonitor(logger=self.logger, checker=self.checker)

    def test_integrity_checker_payload_valid(self):
        payload = "autonomous_vehicle_telemetry_data"
        expected_hash = self.checker.calculate_hash(payload, "sha256")
        report = self.checker.verify_payload("TELEMETRY_01", payload, expected_hash)
        self.assertTrue(report.is_valid)
        self.assertEqual(report.status, "VALID")

    def test_integrity_checker_payload_tampered(self):
        payload = "autonomous_vehicle_telemetry_data"
        tampered_hash = "0" * 64
        report = self.checker.verify_payload("TELEMETRY_01", payload, tampered_hash)
        self.assertFalse(report.is_valid)
        self.assertEqual(report.status, "TAMPERED")

    def test_security_logger_events(self):
        event = self.logger.log_info("CAN_NODE", "STARTUP", "CAN Node active")
        self.assertEqual(event.severity, SecuritySeverity.INFO)
        self.assertEqual(event.source, "CAN_NODE")

        self.logger.log_intrusion("CAN_BUS", "Flooding attack detected")
        events = self.logger.get_events(severity=SecuritySeverity.CRITICAL)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "INTRUSION_DETECTED")

    def test_security_monitor_threat_assessment(self):
        # Initial threat level should be NOMINAL
        self.assertEqual(self.monitor.threat_level, ThreatLevel.NOMINAL)

        # Trigger CAN frame failures
        self.monitor.process_can_frame_result(False, {"arb_id": "0x123"})
        self.assertEqual(self.monitor.threat_level, ThreatLevel.ELEVATED)

        # Trigger multiple V2X failures to hit HIGH threat level
        self.monitor.process_v2x_message_result(False, "VEH_99", "INVALID_SIG")
        self.monitor.process_v2x_message_result(False, "VEH_99", "INVALID_SIG")
        self.assertEqual(self.monitor.threat_level, ThreatLevel.HIGH)

        # Trigger further failures to hit CRITICAL threat level and activate fail safe
        self.monitor.process_v2x_message_result(False, "VEH_99", "INVALID_SIG")
        self.monitor.process_v2x_message_result(False, "VEH_99", "INVALID_SIG")
        self.assertEqual(self.monitor.threat_level, ThreatLevel.CRITICAL)
        self.assertTrue(self.monitor.is_fail_safe_active)

    def test_security_monitor_reset(self):
        self.monitor.process_can_frame_result(False)
        self.assertEqual(self.monitor.threat_level, ThreatLevel.ELEVATED)
        self.monitor.reset_threat_level()
        self.assertEqual(self.monitor.threat_level, ThreatLevel.NOMINAL)
        self.assertFalse(self.monitor.is_fail_safe_active)


if __name__ == "__main__":
    unittest.main()
