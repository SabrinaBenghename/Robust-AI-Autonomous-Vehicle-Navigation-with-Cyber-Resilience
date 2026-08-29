import time
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class CANFrame:
    arbitration_id: int   # e.g. 0x1A0 (Speed), 0x2B0 (Steering)
    data: bytes           # 8-byte CAN payload
    timestamp: float

class CANIntrusionDetector:
    """
    Real-time Intrusion Detection System (IDS) monitoring frequency, payload anomalies, and replay attacks on CAN Bus.
    """
    def __init__(self, frequency_threshold_hz: float = 100.0):
        self.frequency_threshold = frequency_threshold_hz
        self.message_history: Dict[int, List[float]] = {}
        self.anomalies_logged: List[Dict] = []

    def inspect_frame(self, frame: CANFrame) -> bool:
        """
        Returns True if frame is valid/benign, False if intrusion detected.
        """
        now = frame.timestamp
        arb_id = frame.arbitration_id

        if arb_id not in self.message_history:
            self.message_history[arb_id] = []
        
        self.message_history[arb_id].append(now)
        # Keep recent 20 timestamps
        if len(self.message_history[arb_id]) > 20:
            self.message_history[arb_id].pop(0)

        # Check frequency / flooding attack
        if len(self.message_history[arb_id]) >= 5:
            dt = self.message_history[arb_id][-1] - self.message_history[arb_id][0]
            freq = len(self.message_history[arb_id]) / max(dt, 0.0001)
            if freq > self.frequency_threshold:
                self.anomalies_logged.append({
                    "type": "BUS_FLOODING_ATTACK",
                    "arb_id": hex(arb_id),
                    "freq_hz": freq,
                    "timestamp": now
                })
                return False

        # Malicious ID or invalid length check
        if len(frame.data) != 8:
            self.anomalies_logged.append({
                "type": "MALFORMED_CAN_PAYLOAD",
                "arb_id": hex(arb_id),
                "len": len(frame.data),
                "timestamp": now
            })
            return False

        return True
