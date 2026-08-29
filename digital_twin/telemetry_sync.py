import time
from dataclasses import dataclass, asdict
from typing import List, Dict

@dataclass
class TelemetryPacket:
    timestamp: float
    vehicle_id: str
    speed_kmh: float
    battery_percentage: float
    tire_pressure_psi: Dict[str, float]
    brake_temp_celsius: float
    latitude: float
    longitude: float

class TelemetrySynchronizer:
    """
    Simulates high-frequency IoT streaming telemetry bridge to edge cloud digital twin.
    """
    def __init__(self, vehicle_id: str = "AV-EGO-01"):
        self.vehicle_id = vehicle_id
        self.buffer: List[TelemetryPacket] = []

    def capture_telemetry(self, speed_mps: float, x: float, y: float) -> TelemetryPacket:
        packet = TelemetryPacket(
            timestamp=time.time(),
            vehicle_id=self.vehicle_id,
            speed_kmh=round(speed_mps * 3.6, 2),
            battery_percentage=94.5,
            tire_pressure_psi={"fl": 32.1, "fr": 32.0, "rl": 31.8, "rr": 31.9},
            brake_temp_celsius=65.4,
            latitude=37.7749 + (y * 0.00001),
            longitude=-122.4194 + (x * 0.00001)
        )
        self.buffer.append(packet)
        return packet

    def sync_to_cloud(self) -> int:
        count = len(self.buffer)
        self.buffer.clear()
        return count
