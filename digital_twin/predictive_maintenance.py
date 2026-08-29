from dataclasses import dataclass

@dataclass
class ComponentHealth:
    brake_pad_wear_pct: float     # 0.0 (new) to 100.0 (worn)
    tire_wear_pct: float
    battery_soh_pct: float         # State of Health
    anomaly_detected: bool
    recommended_action: str

class PredictiveMaintenanceEngine:
    """
    Evaluates real-time sensor history to project remaining useful life (RUL) and schedule maintenance.
    """
    def __init__(self):
        self.total_distance_km = 0.0
        self.brake_cycles = 0

    def evaluate_health(self, speed_mps: float, brake_active: bool, dt: float = 0.1) -> ComponentHealth:
        dist_delta = (speed_mps * dt) / 1000.0
        self.total_distance_km += dist_delta
        if brake_active:
            self.brake_cycles += 1

        brake_wear = min(100.0, (self.brake_cycles / 50000.0) * 100.0)
        tire_wear = min(100.0, (self.total_distance_km / 40000.0) * 100.0)
        battery_soh = max(70.0, 100.0 - (self.total_distance_km / 100000.0) * 10.0)

        anomaly = brake_wear > 85.0 or tire_wear > 90.0
        action = "Inspection Recommended" if anomaly else "Normal Operation"

        return ComponentHealth(
            brake_pad_wear_pct=round(brake_wear, 2),
            tire_wear_pct=round(tire_wear, 2),
            battery_soh_pct=round(battery_soh, 2),
            anomaly_detected=anomaly,
            recommended_action=action
        )
