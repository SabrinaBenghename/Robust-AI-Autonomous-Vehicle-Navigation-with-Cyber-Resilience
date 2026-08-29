import unittest
from cybersecurity.can_ids import CANIntrusionDetector, CANFrame
from cybersecurity.v2x_crypto import V2XCryptoProvider


class TestCybersecurity(unittest.TestCase):

    def test_can_ids(self):
        ids = CANIntrusionDetector()
        frame = CANFrame(arbitration_id=0x123, data=bytes([0] * 8), timestamp=100.0)
        status = ids.inspect_frame(frame)
        self.assertTrue(status)

    def test_v2x_crypto(self):
        crypto = V2XCryptoProvider()
        msg = crypto.sign_message(sender_id="VEH_01", payload="SPEED_50")
        self.assertIsNotNone(msg.signature)


if __name__ == "__main__":
    unittest.main()
